"""TTM (Riemannian test-time mean) validation on run3 holdout (proxy for test session)."""
import sys, os, time, json
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('OMP_NUM_THREADS', '1')
from common import load_subject, TEST_SUBJECTS
import features as F
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score

def _mexp(A):
    w, V = np.linalg.eigh(A)
    return (V * np.exp(w)) @ V.T

def ttm_mix(ref_train, ref_test, lam):
    out = []
    for A, B in zip(ref_train, ref_test):
        M = (1 - lam) * F._matrix_log(A) + lam * F._matrix_log(B)
        out.append(_mexp(M))
    return np.stack(out)

BANDS = F.BANDS
CB = ((8, 12), (12, 16))

def rts(cues, lam, ref=None):
    covs, ref_t = F.riemannian_features(cues, bands=BANDS, off=0.0)
    if ref is None:
        ref = ref_t
    else:
        if lam:
            ref = ttm_mix(ref, ref_t, lam)
    return F.riemannian_features(cues, bands=BANDS, ref=ref, off=0.0)[0], ref

def feats(name, cues, bases, y, ref=None, filt=None, lam=0.0):
    x, ref = rts(cues, lam, ref)
    if name == 'rts':
        return x, ref
    if name == 'bp_rts':
        return np.concatenate([F.band_power_features(cues, bases, bands=BANDS), x], 1), ref
    if ref is None or filt is None:
        filt = F.csp_features(cues, y, bands=CB)[1]
    c = F.csp_transform(cues, filt, bands=CB)
    return np.concatenate([F.band_power_features(cues, bases, bands=BANDS), x, c], 1), ref

def make_model(kind):
    if kind.startswith('lr_'):
        return LogisticRegression(C=float(kind[3:]), max_iter=2000)
    return LinearDiscriminantAnalysis(solver='lsqr', shrinkage=float(kind[4:]))

def subj(sub):
    d = load_subject(sub)
    tr_keys = [k for k in ('tr01', 'tr02') if k in d]
    cues = np.concatenate([d[k][0] for k in tr_keys])
    bases = np.concatenate([d[k][1] for k in tr_keys])
    y = np.concatenate([d[k][2] for k in tr_keys])
    r3 = d.get('tr03')
    if r3 is None or len(r3[2]) < 4 or np.bincount(r3[2]).min() < 2:
        return None
    rc, rb, ry = r3[0], r3[1], r3[2]
    out = {}
    for name in ('all', 'rts', 'bp_rts'):
        Xtr, ref = feats(name, cues, bases, y, None, None, 0.0)
        for lam in (0.0, 0.5, 1.0):
            for mk in ('lr_0.1',):
                Xva, _ = feats(name, rc, rb, ry, ref, None, lam)
                m = make_pipeline(StandardScaler(), make_model(mk))
                m.fit(Xtr, y)
                a = accuracy_score(ry, (m.predict_proba(Xva)[:, 1] >= 0.5).astype(int))
                out['%s|%s|lam%.1f' % (name, mk, lam)] = float(a)
    return out

if __name__ == '__main__':
    t0 = time.time()
    per = {}
    for s in TEST_SUBJECTS:
        r = subj(s)
        if r:
            per[s] = r
            print(s, r, flush=True)
    keys = sorted({k for r in per.values() for k in r})
    print('\n=== run3 holdout mean (n=%d subjects) ===' % len(per))
    for k in keys:
        print('%-22s %.4f' % (k, float(np.mean([r[k] for r in per.values()]))))
    json.dump(per, open('state/cv_ttm.json', 'w'), indent=1)
    print('elapsed %.0fs' % (time.time() - t0))
