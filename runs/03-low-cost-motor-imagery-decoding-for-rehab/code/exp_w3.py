"""Test run-3 sample weighting on the strong ML configs."""
import sys, os, json, time, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('OMP_NUM_THREADS', '1')
from common import load_subject, TEST_SUBJECTS
import features as F
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score

BANDS = F.BANDS

def featurize(name, cues, bases, y):
    if name == 'rts':
        x, ref = F.riemannian_features(cues)
        return x, ref
    if name == 'all':
        x, ref = F.riemannian_features(cues)
        c, filt = F.csp_features(cues, y)
        return np.concatenate([F.band_power_features(cues, bases), x, c], 1), (ref, filt)
    raise ValueError(name)

def featurize_va(name, cues, bases, state):
    ref, filt = (state, None) if name == 'rts' else state
    x, _ = F.riemannian_features(cues, ref=ref)
    if name == 'rts':
        return x
    c = F.csp_transform(cues, filt)
    return np.concatenate([F.band_power_features(cues, bases), x, c], 1)

def make_model(kind, w3):
    def fit(X, y, w):
        # integer-weight emulation by row replication (works for both models)
        rep = []
        for i, wi in enumerate(w):
            rep.extend([i] * int(round(wi)))
        Xr, yr = X[rep], y[rep]
        if kind.startswith('lr'):
            m = LogisticRegression(C=float(kind[2:]), max_iter=2000)
        else:
            m = LinearDiscriminantAnalysis(solver='lsqr', shrinkage=float(kind[3:]))
        m.fit(Xr, yr)
        return m
    return fit

def eval_subject(sub, w3, configs):
    d = load_subject(sub)
    runs = [k for k in ('tr01', 'tr02', 'tr03') if k in d]
    cues = np.concatenate([d[k][0] for k in runs])
    bases = np.concatenate([d[k][1] for k in runs])
    y = np.concatenate([d[k][2] for k in runs])
    pos, r3 = 0, []
    for k in runs:
        n = len(d[k][2])
        if k == 'tr03':
            r3 = list(range(pos, pos + n))
        pos += n
    w = np.ones(len(y))
    w[r3] = w3
    skf = StratifiedKFold(5, shuffle=True, random_state=42)
    oof = {}
    for fn, mk in configs:
        o = np.zeros(len(y))
        for tr_i, va_i in skf.split(cues, y):
            Xtr, state = featurize(fn, cues[tr_i], bases[tr_i], y[tr_i])
            Xva = featurize_va(fn, cues[va_i], bases[va_i], state)
            pipe_sc = StandardScaler()
            Xtr_s = pipe_sc.fit_transform(Xtr)
            Xva_s = pipe_sc.transform(Xva)
            m = make_model(mk, w3)(Xtr_s, y[tr_i], w[tr_i])
            o[va_i] = m.predict_proba(Xva_s)[:, 1]
        oof[(fn, mk)] = o
    # ensemble prob
    ens = np.mean(list(oof.values()), axis=0)
    acc5 = accuracy_score(y, (ens >= 0.5).astype(int))
    # run3 holdout ensemble
    r3a = np.array(r3)
    if len(r3a) >= 4 and np.bincount(y[r3a]).min() >= 2:
        mask = np.ones(len(y), bool); mask[r3a] = False
        ps = []
        for fn, mk in configs:
            Xtr, state = featurize(fn, cues[mask], bases[mask], y[mask])
            Xva = featurize_va(fn, cues[r3a], bases[r3a], state)
            sc = StandardScaler().fit(Xtr)
            m = make_model(mk, w3)(sc.transform(Xtr), y[mask], w[mask])
            ps.append(m.predict_proba(sc.transform(Xva))[:, 1])
        p = np.mean(ps, axis=0)
        acc3 = accuracy_score(y[r3a], (p >= 0.5).astype(int))
    else:
        acc3 = np.nan
    # threshold tuning on OOF (argmax accuracy)
    best_t, best_a = 0.5, acc5
    for t in np.arange(0.35, 0.66, 0.01):
        a = accuracy_score(y, (ens >= t).astype(int))
        if a > best_a:
            best_a, best_t = a, t
    return acc5, acc3, best_a, best_t

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--w3', type=float, default=3.0)
    ap.add_argument('--subjects', nargs='+', default=TEST_SUBJECTS)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    configs = [('all', 'lr0.1'), ('all', 'lr1'), ('rts', 'lr0.1'), ('rts', 'lda0.5')]
    per = {}
    for sub in args.subjects:
        a5, a3, bt, t = eval_subject(sub, args.w3, configs)
        per[sub] = (a5, a3, bt, t)
        print(sub, f'w3={args.w3}', f'ens5fold={a5:.3f} run3={a3 if a3==a3 else float("nan"):.3f} bestT={bt:.3f}@{t:.2f}', flush=True)
    json.dump(per, open(args.out, 'w'))
