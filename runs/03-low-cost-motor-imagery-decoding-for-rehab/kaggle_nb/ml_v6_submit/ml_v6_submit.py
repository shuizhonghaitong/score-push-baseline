"""LCMID single-subject ML pipeline (competition-rule compliant ML track).
Per-subject: band power + Riemannian tangent space + CSP features,
LogReg/LDA ensemble, probability averaging. Writes submission.csv (auto-submitted)."""
import os, glob, json, time, warnings
os.environ.setdefault('OMP_NUM_THREADS', os.environ.get('LCMID_THREADS', '2'))
os.environ.setdefault('OPENBLAS_NUM_THREADS', os.environ.get('LCMID_THREADS', '2'))
os.environ.setdefault('MKL_NUM_THREADS', os.environ.get('LCMID_THREADS', '2'))
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from scipy import signal
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.covariance import LedoitWolf
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

def _find_data():
    slug = 'low-cost-motor-imagery-decoding-for-rehab'
    cands = [
        '/kaggle/input/competitions/' + slug,
        '/kaggle/input/' + slug,
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data'),
    ]
    if os.environ.get('LCMID_DATA'):
        cands = [os.environ['LCMID_DATA']] + cands
    for p in cands:
        if os.path.isdir(p) and os.path.isdir(os.path.join(p, 'Training set')):
            return p
    if os.path.isdir('/kaggle/input'):
        for root, dirs, files in os.walk('/kaggle/input'):
            if 'Training set' in dirs:
                return root
    raise SystemExit('DATA_NOT_FOUND. /kaggle/input: %r' % (os.listdir('/kaggle/input') if os.path.isdir('/kaggle/input') else 'none'))

DATA = _find_data()
FS = 250.0
CHANNELS = ['Fz', 'C3', 'Cz', 'C4', 'PO7', 'Pz', 'PO8', 'Oz']
CUE_LEN, BASE_LEN = 5.0, 3.0
TCUE, TBASE = int(CUE_LEN * FS), int(BASE_LEN * FS)
TEST_SUBJECTS = ['S%03d' % i for i in [1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 14, 16, 17, 18, 19, 20]]
SEED = 42
BANDSETS = {
    'A': ([(4, 8), (8, 12), (12, 16), (16, 20), (20, 25), (25, 30)], ((8, 12), (12, 16))),
    'B': ([(4, 6), (6, 8), (8, 10), (10, 12), (12, 16), (16, 20), (20, 30)], ((8, 12), (12, 16), (16, 20))),
}
OFF = 0.0  # unused; per-config offset

# ---------------- signal ----------------
_NOTCH = signal.iirnotch(50.0 / (FS / 2.0), 30.0)
_BUTTER = {}
def band_filter(x, l, h):
    key = (float(l), float(h))
    if key not in _BUTTER:
        nyq = FS / 2.0
        _BUTTER[key] = signal.butter(4, [max(0.01, l / nyq), min(0.99, h / nyq)], btype='bandpass')
    b, a = _BUTTER[key]
    axis = 1 if x.ndim == 3 else 0
    return signal.filtfilt(b, a, x, axis=axis)

def read_epochs(path, labeled=True):
    df = pd.read_csv(path, low_memory=False)
    x = df[CHANNELS].to_numpy(np.float64)
    m = df['Marker_val'].astype(str)
    x = x - x.mean(axis=1, keepdims=True)
    x = signal.filtfilt(_NOTCH[0], _NOTCH[1], x, axis=0)
    if labeled:
        idx = np.flatnonzero(m.isin(['rest', 'move']).to_numpy())
        y = np.array([1 if v == 'move' else 0 for v in m.iloc[idx]])
    else:
        idx = np.flatnonzero((m == 'cue_start').to_numpy())
        y = None
    cues, bases = [], []
    for i in idx:
        if i + TCUE > len(x):
            continue
        cues.append(x[i:i + TCUE])
        b0 = max(0, i - TBASE)
        bases.append(x[b0:i][-TBASE:])
    return np.stack(cues), np.stack(bases), y

def load_subject(sub):
    runs = {}
    for f in sorted(glob.glob(os.path.join(DATA, 'Training set', f'{sub}-*_eeg.csv'))):
        run = os.path.basename(f).split('-')[1][:3]
        runs[run] = read_epochs(f, labeled=True)
    tc, tb, _ = read_epochs(os.path.join(DATA, 'Single subject test set', f'{sub}_test.csv'), labeled=False)
    return runs, tc, tb

# ---------------- features ----------------
def band_power(cue, base, bands, off):
    if off > 0:
        k = int(off * FS)
        cue = cue[:, k:, :]
    N = cue.shape[0]
    feats = []
    for (l, h) in bands:
        cf = band_filter(cue, l, h)
        bf = band_filter(base, l, h)
        pc = np.log(np.maximum((cf ** 2).mean(axis=1), 1e-12))
        pb = np.log(np.maximum((bf ** 2).mean(axis=1), 1e-12))
        feats.append(pc)
        feats.append(pc - pb)
    return np.concatenate(feats, axis=1)

def _eigh_op(A, p):
    w, v = np.linalg.eigh(A)
    return v @ np.diag(w ** p) @ v.T

def _mlog(A):
    return _eigh_op(A, 1.0)

def _mlog_stable(A):
    w, v = np.linalg.eigh(A)
    return v @ np.diag(np.log(np.maximum(w, 1e-12))) @ v.T

def rts_fit(cue, bands, off):
    est = LedoitWolf()
    covs = []
    if off > 0:
        k = int(off * FS)
        cue = cue[:, k:, :]
    for (l, h) in bands:
        cf = band_filter(cue, l, h)
        covs.append(np.array([est.fit(x).covariance_ for x in cf]))
    ref = np.stack([np.mean(c, axis=0) for c in covs])
    return covs, ref

def rts_project(covs, ref):
    feats = []
    for ci, r in zip(covs, ref):
        Ginv_sqrt = _eigh_op(r, -0.5)
        f = []
        for C in ci:
            W = Ginv_sqrt @ C @ Ginv_sqrt
            L = _mlog_stable(W)
            idx = np.triu_indices(8)
            v = L[idx].copy()
            v[1:] *= np.sqrt(2.0)
            f.append(v)
        feats.append(np.stack(f))
    return np.concatenate(feats, axis=1)

def csp_fit(cue, y, bands, off):
    est = LedoitWolf()
    if off > 0:
        k = int(off * FS)
        cue = cue[:, k:, :]
    filters = []
    for (l, h) in bands:
        cf = band_filter(cue, l, h)
        C1 = np.mean([np.cov(x.T) for x in cf[y == 1]], axis=0)
        C0 = np.mean([np.cov(x.T) for x in cf[y == 0]], axis=0)
        C1, C0 = est.fit(C1).covariance_, est.fit(C0).covariance_
        w, V = np.linalg.eigh(np.linalg.pinv(C0) @ C1)
        V = V[:, np.argsort(w)]
        F = np.concatenate([V[:, -4:], V[:, :4]], axis=1)
        filters.append(F)
    return filters

def csp_transform(cue, filters, bands, off):
    if off > 0:
        k = int(off * FS)
        cue = cue[:, k:, :]
    feats = []
    for (l, h), F in zip(bands, filters):
        cf = band_filter(cue, l, h)
        d = np.log(np.maximum(np.array([np.diag(np.cov((x @ F).T)) for x in cf]), 1e-12))
        feats.append(d)
    return np.concatenate(feats, axis=1)

# ---------------- models ----------------
def make_model(kind):
    if kind.startswith('lr_'):
        return LogisticRegression(C=float(kind[3:]), max_iter=2000)
    if kind.startswith('lda_'):
        return LinearDiscriminantAnalysis(solver='lsqr', shrinkage=float(kind[4:]))
    raise ValueError(kind)

# (feat, model, offset_s, bandset)
CONFIGS = [
    ('all', 'lr_0.1', 0.0, 'A'), ('all', 'lr_1', 0.0, 'A'), ('all', 'lr_10', 0.0, 'A'),
    ('rts', 'lr_0.1', 0.0, 'A'), ('rts', 'lr_1', 0.0, 'A'),
    ('bp_rts', 'lr_0.1', 0.0, 'A'),
    ('all', 'lda_0.5', 0.0, 'A'), ('rts', 'lda_0.5', 0.0, 'A'),
    ('all', 'lr_0.1', 0.5, 'B'),
    ('rts', 'lr_0.1', 0.5, 'B'),
    ('bp_rts', 'lr_0.1', 0.5, 'B'),
    ('all', 'lr_0.1', 0.25, 'A'), ('rts', 'lr_0.1', 0.25, 'A'), ('bp_rts', 'lr_0.1', 0.25, 'A'),
    ('all', 'lr_0.1', 0.75, 'A'), ('rts', 'lr_0.1', 0.75, 'A'), ('bp_rts', 'lr_0.1', 0.75, 'A'),
    ('all', 'lr_0.1', 0.0, 'A', 1000), ('rts', 'lr_0.1', 0.0, 'A', 1000), ('bp_rts', 'lr_0.1', 0.0, 'A', 1000),
    ('all', 'lr_0.1', 1.0, 'A'), ('rts', 'lr_0.1', 1.0, 'A'), ('bp_rts', 'lr_0.1', 1.0, 'A'),
    ('all', 'lr_0.1', 0.25, 'B'), ('rts', 'lr_0.1', 0.25, 'B'), ('bp_rts', 'lr_0.1', 0.25, 'B'),
    ('all', 'lr_0.1', 0.75, 'B'), ('rts', 'lr_0.1', 0.75, 'B'), ('bp_rts', 'lr_0.1', 0.75, 'B'),
    ('all', 'lr_0.3', 0.0, 'A'), ('rts', 'lr_0.3', 0.0, 'A'), ('bp_rts', 'lr_0.3', 0.0, 'A'),
    ('all', 'lda_0.3', 0.0, 'A'), ('rts', 'lda_0.3', 0.0, 'A'), ('bp_rts', 'lda_0.3', 0.0, 'A'),
]
CONFIGS = [c + (None,) * (5 - len(c)) for c in CONFIGS]

def featurize(name, cues, bases, y, state, off=0.0, bs='A', cue_end=None):
    if cue_end:
        cues = cues[:, :cue_end]
    bands, cbands = BANDSETS[bs]
    if state is None:
        if name == 'rts':
            covs, ref = rts_fit(cues, bands, off)
            return rts_project(covs, ref), ref
        if name == 'bp_rts':
            covs, ref = rts_fit(cues, bands, off)
            return np.concatenate([band_power(cues, bases, bands, off), rts_project(covs, ref)], 1), ref
        if name == 'all':
            covs, ref = rts_fit(cues, bands, off)
            filt = csp_fit(cues, y, cbands, off)
            return (np.concatenate([band_power(cues, bases, bands, off), rts_project(covs, ref),
                                     csp_transform(cues, filt, cbands, off)], 1), (ref, filt))
        raise ValueError(name)
    ref, filt = (state, None) if name != 'all' else state
    x_rts = rts_project(rts_fit(cues, bands, off)[0], ref)
    if name == 'rts':
        return x_rts
    if name == 'bp_rts':
        return np.concatenate([band_power(cues, bases, bands, off), x_rts], 1)
    c = csp_transform(cues, filt, cbands, off)
    return np.concatenate([band_power(cues, bases, bands, off), x_rts, c], 1)

W3 = 3  # replicate run-3 trials W3 times when fitting (same session as test)

def run3_rep(n, r3):
    idx = list(range(n))
    for i in r3:
        idx.extend([i] * (W3 - 1))
    return np.array(idx)

def fit_model(mk, Xtr, y, r3):
    rep = run3_rep(len(y), r3)
    m = make_model(mk)
    sc = StandardScaler()
    m.fit(sc.fit_transform(Xtr[rep]), y[rep])
    return lambda X: m.predict_proba(sc.transform(X))[:, 1]

def predict_subject(sub, configs=CONFIGS, weights=None):
    runs, tc, tb = load_subject(sub)
    cues = np.concatenate([runs[r][0] for r in sorted(runs)])
    bases = np.concatenate([runs[r][1] for r in sorted(runs)])
    y = np.concatenate([runs[r][2] for r in sorted(runs)])
    pos, r3 = 0, []
    for r in sorted(runs):
        n = len(runs[r][2])
        if r == '003':
            r3 = list(range(pos, pos + n))
        pos += n
    probs = []
    for fn, mk, off, bs, ce in configs:
        Xtr, state = featurize(fn, cues, bases, y, None, off, bs, ce)
        Xva = featurize(fn, tc, tb, None, state, off, bs, ce)
        probs.append(fit_model(mk, Xtr, y, r3)(Xva))
    probs = np.array(probs)
    if weights is None:
        weights = np.ones(len(configs))
    return probs, np.asarray(weights) / np.asarray(weights).sum()

def oof_matrix(sub, configs=CONFIGS, n_folds=5, seed=SEED):
    runs, tc, tb = load_subject(sub)
    cues = np.concatenate([runs[r][0] for r in sorted(runs)])
    bases = np.concatenate([runs[r][1] for r in sorted(runs)])
    y = np.concatenate([runs[r][2] for r in sorted(runs)])
    r3_set = set()
    pos = 0
    for r in sorted(runs):
        n = len(runs[r][2])
        if r == '003':
            r3_set = set(range(pos, pos + n))
        pos += n
    skf = StratifiedKFold(n_folds, shuffle=True, random_state=seed)
    oof = np.zeros((len(y), len(configs)))
    for tr_i, va_i in skf.split(cues, y):
        tr3 = [int(np.searchsorted(tr_i, g)) for g in sorted(r3_set)
               if np.searchsorted(tr_i, g) < len(tr_i) and tr_i[np.searchsorted(tr_i, g)] == g]
        for j, (fn, mk, off, bs, ce) in enumerate(configs):
            Xtr, state = featurize(fn, cues[tr_i], bases[tr_i], y[tr_i], None, off, bs, ce)
            Xva = featurize(fn, cues[va_i], bases[va_i], None, state, off, bs, ce)
            oof[va_i, j] = fit_model(mk, Xtr, y[tr_i], tr3)(Xva)
    return oof, y

def main():
    t0 = time.time()
    rows = []
    i = 0
    diag = {}
    from sklearn.metrics import accuracy_score as _acc
    for sub in TEST_SUBJECTS:
        oof, y = oof_matrix(sub)
        acc5 = np.array([_acc(y, (oof[:, j] >= 0.5).astype(int)) for j in range(oof.shape[1])])
        topk = np.argsort(acc5)[::-1][:5]
        w = np.zeros(len(acc5))
        w[topk] = 1.0
        w = w / w.sum()
        P = oof @ w
        best_t, best_a = 0.5, _acc(y, (P >= 0.5).astype(int))
        for t in np.arange(0.40, 0.601, 0.01):
            a = _acc(y, (P >= t).astype(int))
            if a > best_a:
                best_a, best_t = a, t
        probs, _ = predict_subject(sub, weights=w)
        p = probs.T @ w
        diag[sub] = {'cv5_u': round(float(_acc(y, (oof.mean(axis=1) >= 0.5).astype(int))), 4),
                     'cv5_w': round(float(best_a), 4),
                     'thr': round(float(best_t), 2),
                     'n_move_pred': int((p >= best_t).sum())}
        for q in p:
            rows.append((i, 'move' if q >= best_t else 'rest'))
            i += 1
        print(sub, diag[sub], flush=True)
    assert i == 680, i
    df = pd.DataFrame(rows, columns=['ID', 'TARGET'])
    df.to_csv('submission.csv', index=False)
    json.dump({'diag': diag, 'elapsed_s': round(time.time() - t0, 1)}, open('diag.json', 'w'), indent=1)
    print('SUBMISSION_WRITTEN', len(df), df['TARGET'].value_counts().to_dict(), flush=True)


if __name__ == '__main__':
    main()

# ---- in-kernel submission ----
def submit_submission():
    import pandas as pd
    SLUG = 'low-cost-motor-imagery-decoding-for-rehab'
    df = pd.read_csv('submission.csv')
    assert list(df.columns) == ['ID', 'TARGET'] and len(df) == 680, (list(df.columns), len(df))
    assert (df['ID'].values == list(range(680))).all()
    assert set(df['TARGET'].unique()) <= {'rest', 'move'}
    print('schema OK:', df['TARGET'].value_counts().to_dict(), flush=True)
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    res = api.competition_submit('submission.csv',
        'ML v6: 35cfg ensemble, per-subject top5 by OOF acc, w3=3, OOF threshold', SLUG, quiet=True)
    ref = getattr(res, 'ref', None)
    print('SUBMIT_RECEIPT ref=%s' % ref, flush=True)
    if not (isinstance(ref, int) and ref > 0):
        raise SystemExit('SUBMIT_FAILED: %r' % (res,))

if os.environ.get('LCMID_SUBMIT') == '1' or not os.environ.get('LCMID_LOCAL'):
    submit_submission()
