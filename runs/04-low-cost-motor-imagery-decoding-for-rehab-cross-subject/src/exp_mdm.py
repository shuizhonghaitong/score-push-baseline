"""MDM (Riemannian min-distance) + FBCSP study. LOSO, per-file balanced + 0.5 eval."""
import os, sys, re, time, glob
import numpy as np
for v in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(v, '4')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import read_epochs, DATA
from fast_riemannian import BANDS, band_filter, _triu_vec
from features import band_power_features, csp_features, csp_transform
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

t0 = time.time()
def log(*a): print('[%6.1fs]' % (time.time()-t0), *a, flush=True)

files = sorted(glob.glob(DATA + '/Training set/*.csv'))
cues, bases, y, sub = [], [], [], []
for f in files:
    c, b, lab = read_epochs(f, labeled=True)
    cues.append(c); bases.append(b); y.append(lab)
    sub.append(np.full(len(lab), re.search(r'(S\d+)-', f).group(1)))
cues = np.concatenate(cues).astype(np.float32); bases = np.concatenate(bases)
y = np.concatenate(y); sub = np.concatenate(sub)
subjects = sorted(set(sub))
t, c = cues.shape[1], cues.shape[2]
log('loaded', cues.shape)

Xbf6 = np.stack([band_filter(cues, l, h).astype(np.float32) for (l, h) in BANDS])
log('band-filtered Xbf6', Xbf6.shape)

def covs(Xb6):
    return [np.einsum('nta,ntb->nab', Xb6[bi], Xb6[bi]) / (t - 1) for bi in range(6)]

def logmap(C, R):
    w, v = np.linalg.eigh(R)
    G = v @ np.diag(np.maximum(w, 1e-12) ** -0.5) @ v.T
    W = np.einsum('ab,nbc,cd->nad', G, C, G)
    w2, v2 = np.linalg.eigh(W)
    L = (v2 * np.log(np.maximum(w2, 1e-12))[:, None, :]) @ v2.transpose(0, 2, 1)
    return np.stack([_triu_vec(L[i]) for i in range(len(L))])

def lr(cC): return make_pipeline(StandardScaler(), LogisticRegression(C=cC, max_iter=5000))
def lda(): return LinearDiscriminantAnalysis(solver='lsqr')
def acc05(P, yy): return ((P > 0.5).astype(int) == yy).mean()
def acc_bal(P, yy):
    k = len(yy) // 2
    pred = np.zeros(len(yy), int); pred[np.argpartition(-P, k)[:k]] = 1
    return (pred == yy).mean()

bp_all = band_power_features(cues, bases).astype(np.float32)
log('bp done')

N = len(y)
Pmdm_raw = np.zeros(N); Pmdm_lr = np.zeros(N); Pfbcsp = np.zeros(N)
for s in subjects:
    va = sub == s; tr = ~va
    Ctr, Cva = covs(Xbf6[:, tr]), covs(Xbf6[:, va])
    R1 = [np.mean(Ctr[bi][y[tr] == 1], 0) for bi in range(6)]
    R0 = [np.mean(Ctr[bi][y[tr] == 0], 0) for bi in range(6)]
    fva1 = np.concatenate([logmap(Cva[bi], R1[bi]) for bi in range(6)], 1)
    fva0 = np.concatenate([logmap(Cva[bi], R0[bi]) for bi in range(6)], 1)
    ftr1 = np.concatenate([logmap(Ctr[bi], R1[bi]) for bi in range(6)], 1)
    ftr0 = np.concatenate([logmap(Ctr[bi], R0[bi]) for bi in range(6)], 1)
    d1 = np.linalg.norm(fva1, axis=1); d0 = np.linalg.norm(fva0, axis=1)
    Pmdm_raw[va] = 1 / (1 + np.exp(-(d0 - d1)))
    Ftr = np.concatenate([ftr1, ftr0], 1); Fva = np.concatenate([fva1, fva0], 1)
    Pmdm_lr[va] = lr(1.0).fit(Ftr, y[tr]).predict_proba(Fva)[:, 1]
    ftr_b, fva_b = [], []
    for (l, h) in BANDS:
        cft, filt = csp_features(cues[tr], y[tr], bands=((l, h),), k=4)
        cfv = csp_transform(cues[va], filt, bands=((l, h),))
        ftr_b.append(cft); fva_b.append(cfv)
    Ffb_tr = np.concatenate(ftr_b, 1); Ffb_va = np.concatenate(fva_b, 1)
    Pfbcsp[va] = np.mean([lr(0.1).fit(Ffb_tr, y[tr]).predict_proba(Ffb_va)[:, 1],
                          lr(1.0).fit(Ffb_tr, y[tr]).predict_proba(Ffb_va)[:, 1],
                          lda().fit(Ffb_tr, y[tr]).predict_proba(Ffb_va)[:, 1]], axis=0)
    log('subject', s, 'done')

def report(name, P):
    a05, abal = acc05(P, y), acc_bal(P, y)
    log('%-10s overall 0.5=%.4f bal=%.4f' % (name, a05, abal))
    return a05, abal

log('=== results (LOSO) ===')
report('mdm_raw', Pmdm_raw)
report('mdm_lr', Pmdm_lr)
report('fbcsp', Pfbcsp)
report('mdm+fbcsp', 0.5 * Pmdm_lr + 0.5 * Pfbcsp)
report('all3', (Pmdm_raw + Pmdm_lr + Pfbcsp) / 3)
log('MDM STUDY DONE')
