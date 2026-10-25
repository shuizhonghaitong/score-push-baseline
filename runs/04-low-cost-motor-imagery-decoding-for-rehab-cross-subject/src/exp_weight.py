"""Does upweighting run-3 (neurofeedback) trials help? fast-w3 recipe, LOSO, per-file balanced."""
import os, sys, re, time, glob
import numpy as np
for v in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
    os.environ.setdefault(v, '4')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import read_epochs, DATA
from fast_riemannian import fast_riemannian_features
from features import band_power_features, csp_features, csp_transform
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

t0 = time.time()
def log(*a): print('[%6.1fs]' % (time.time()-t0), *a, flush=True)

files = sorted(glob.glob(DATA + '/Training set/*.csv'))
cues, bases, y, sub, run = [], [], [], [], []
for f in files:
    c, b, lab = read_epochs(f, labeled=True)
    cues.append(c); bases.append(b); y.append(lab)
    sub.append(np.full(len(lab), re.search(r'(S\d+)-', f).group(1)))
    run.append(np.full(len(lab), re.search(r'-\d+_', f).group(0).strip('-_')))
cues = np.concatenate(cues).astype(np.float32); bases = np.concatenate(bases)
y = np.concatenate(y); sub = np.concatenate(sub); run = np.concatenate(run)
subjects = sorted(set(sub))
log('loaded', cues.shape)
bp_all = band_power_features(cues, bases).astype(np.float32)

def lr(cC): return make_pipeline(StandardScaler(), LogisticRegression(C=cC, max_iter=5000, class_weight=None))
def lda(): return LinearDiscriminantAnalysis(solver='lsqr')
def acc_bal(P, yy):
    k = len(yy) // 2
    pred = np.zeros(len(yy), int); pred[np.argpartition(-P, k)[:k]] = 1
    return (pred == yy).mean()

is_r3 = (run == '003')
res = {}
for w in (1.0, 2.0, 4.0):
    P = np.zeros(len(y))
    for s in subjects:
        va = sub == s; tr = ~va
        rts_tr, ref = fast_riemannian_features(cues[tr])
        rts_va, _ = fast_riemannian_features(cues[va], ref=ref)
        cft, filt = csp_features(cues[tr], y[tr])
        cfv = csp_transform(cues[va], filt)
        Ftr = np.concatenate([bp_all[tr], rts_tr, cft], 1)
        Fva = np.concatenate([bp_all[va], rts_va, cfv], 1)
        sw = 1.0 + (w - 1.0) * is_r3[tr]
        rep = np.repeat(np.arange(len(y[tr])), (sw).astype(int))
        P[va] = np.mean([lr(0.1).fit(Ftr, y[tr], logisticregression__sample_weight=sw).predict_proba(Fva)[:, 1],
                         lr(1.0).fit(Ftr, y[tr], logisticregression__sample_weight=sw).predict_proba(Fva)[:, 1],
                         lda().fit(Ftr[rep], y[tr][rep]).predict_proba(Fva)[:, 1]], axis=0)
    r3f, oth = [], []
    for s in subjects:
        for r in ('001', '002', '003'):
            m = (sub == s) & (run == r)
            if m.sum() < 10: continue
            (r3f if r == '003' else oth).append(acc_bal(P[m], y[m]))
    res[w] = (np.mean(r3f), np.mean(oth), np.mean(r3f + oth))
    log('w_r3=%.0f: run3-files=%.4f (n=%d) other-files=%.4f (n=%d) all=%.4f' % (w, res[w][0], len(r3f), res[w][1], len(oth), res[w][2]))
log('WEIGHT STUDY DONE')
