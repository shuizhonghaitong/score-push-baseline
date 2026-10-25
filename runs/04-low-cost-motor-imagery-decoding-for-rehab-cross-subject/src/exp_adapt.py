"""Subject-adaptive pseudo-labeling, evaluated by LOSO.

Protocol per held-out subject (simulates the test subject):
  1. Cross-subject probability P0 (w3 features).
  2. Per-subject standardized band-power features (unlabeled use).
  3. K-means k=2; direction = cluster with higher mean P0 (or ERD).
  4. Per-subject LDA on pseudo-labels; iterate; final blend with P0.
"""
import sys, glob, re, time
import numpy as np
sys.path.insert(0, '.')
from common import FS, read_epochs, DATA
from features import (band_power_features, riemannian_features,
                      csp_features, csp_transform)
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.cluster import KMeans

files = sorted(glob.glob(DATA + '/Training set/*.csv'))
cues, bases, y, sub = [], [], [], []
for f in files:
    c, b, lab = read_epochs(f, labeled=True)
    subj = re.search(r'(S\d+)-', f).group(1)
    cues.append(c); bases.append(b); y.append(lab); sub.append(np.full(len(lab), subj))
cues = np.concatenate(cues); bases = np.concatenate(bases)
y = np.concatenate(y); sub = np.concatenate(sub)
subjects = sorted(set(sub))
bp_all = band_power_features(cues, bases)
print('data ready', bp_all.shape)

# precompute per-subject z-scored bp (only subject's own statistics)
bp_subj = np.zeros_like(bp_all)
for s in subjects:
    m = sub == s
    mu, sd = bp_all[m].mean(axis=0), bp_all[m].std(axis=0) + 1e-8
    bp_subj[m] = (bp_all[m] - mu) / sd

def w3_feats_fold(tr, va):
    rts_tr, ref = riemannian_features(cues[tr])
    cspf_tr, f0 = csp_features(cues[tr], y[tr])
    rts_va, _ = riemannian_features(cues[va], ref=ref)
    cspf_va = csp_transform(cues[va], f0)
    return (np.concatenate([bp_all[tr], rts_tr, cspf_tr], 1),
            np.concatenate([bp_all[va], rts_va, cspf_va], 1))

def run_variant(name, iters=1, blend=0.5, direction='p'):
    accs = []
    for s in subjects:
        va = sub == s; tr = ~va
        Ftr, Fva = w3_feats_fold(tr, va)
        P0 = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=5000)).fit(Ftr, y[tr]).predict_proba(Fva)[:, 1]
        xv = bp_subj[va]
        for it in range(iters):
            km = KMeans(2, n_init=10, random_state=0).fit(xv)
            lab = km.labels_
            if direction == 'p':
                d = P0[lab == 0].mean() > P0[lab == 1].mean()
            else:  # erd: use erd columns of bp (cols 8..24 of w3 bp = ratio block? keep simple: use P0)
                d = True
            plab = (lab == (0 if d else 1)).astype(int)
            if it < iters - 1:
                P0 = plab.astype(float)  # sharpen for next direction
            try:
                lda = LinearDiscriminantAnalysis(solver='lsqr').fit(xv, plab)
                Pv = lda.predict_proba(xv)[:, 1]
            except Exception:
                Pv = plab.astype(float)
        Pfin = (1 - blend) * Pv + blend * P0
        accs.append(((Pfin > 0.5).astype(int) == y[va]).mean())
    print(f'{name:28s} mean={np.mean(accs):.4f}  {np.round(accs, 2)}')
    return np.mean(accs)

t0 = time.time()
# baselines
accs = []
for s in subjects:
    va = sub == s; tr = ~va
    Ftr, Fva = w3_feats_fold(tr, va)
    P0 = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=5000)).fit(Ftr, y[tr]).predict_proba(Fva)[:, 1]
    accs.append(((P0 > 0.5).astype(int) == y[va]).mean())
print(f'{"baseline w3 lr_0.1":28s} mean={np.mean(accs):.4f}')

# pure per-subject LDA pseudo-labels (1 iter, no blend)
run_variant('adapt 1it blend=0.0', iters=1, blend=0.0)
run_variant('adapt 1it blend=0.5', iters=1, blend=0.5)
run_variant('adapt 1it blend=0.8', iters=1, blend=0.8)
run_variant('adapt 3it blend=0.0', iters=3, blend=0.0)
run_variant('adapt 3it blend=0.5', iters=3, blend=0.5)
print('total', round(time.time() - t0, 1))
