"""Ensemble evaluation for w3 feature set (+ cross-feature ensemble with v1)."""
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
Xv1 = np.load('../outputs/train_feats.npy')

models = {
    'lr_0.1': lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=5000)),
    'lr_1': lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000)),
    'lr_10': lambda: make_pipeline(StandardScaler(), LogisticRegression(C=10.0, max_iter=5000)),
    'lda': lambda: LinearDiscriminantAnalysis(solver='lsqr'),
}

def fold_probs(model_names, use_v1=False):
    """Return proba matrix (n, len(model_names)) via LOSO."""
    n = len(y)
    P = np.zeros((n, len(model_names)))
    for k, name in enumerate(model_names):
        for s in subjects:
            va = sub == s; tr = ~va
            if use_v1:
                Ftr, Fva = Xv1[tr], Xv1[va]
            else:
                rts_tr, ref = riemannian_features(cues[tr])
                cspf_tr, filters0 = csp_features(cues[tr], y[tr])
                rts_va, _ = riemannian_features(cues[va], ref=ref)
                cspf_va = csp_transform(cues[va], filters0)
                Ftr = np.concatenate([bp_all[tr], rts_tr, cspf_tr], axis=1)
                Fva = np.concatenate([bp_all[va], rts_va, cspf_va], axis=1)
            m = models[name]()
            m.fit(Ftr, y[tr])
            P[va, k] = m.predict_proba(Fva)[:, 1]
    return P

def ens_acc(P, y, thr=0.5):
    return ((P.mean(axis=1) > thr).astype(int) == y).mean()

P_w3 = fold_probs(['lr_0.1','lr_1','lr_10','lda'], use_v1=False)
print('w3 4-model prob ensemble LOSO: %.4f' % ens_acc(P_w3, y))
for sub_names in (['lr_0.1','lr_1'], ['lr_0.1','lr_1','lr_10'], ['lr_0.1','lda']):
    idx = [list(models).index(n) for n in sub_names]
    print(f'w3 {sub_names}: {ens_acc(P_w3[:, idx], y):.4f}')

P_v1 = fold_probs(['lr_0.1','lr_1'], use_v1=True)
print('v1 2-model prob ensemble LOSO: %.4f' % ens_acc(P_v1, y))
Pmix = np.concatenate([P_w3, P_v1], axis=1)
print('mixed w3+v1 6-model: %.4f' % ens_acc(Pmix, y))
print('mixed w3+v1 w3-first (avg of two group means):', 
      (((P_w3.mean(1)+P_v1.mean(1))/2) > 0.5).astype(int).__eq__(y).mean())
np.save('../outputs/P_w3_loso.npy', P_w3)
np.save('../outputs/P_v1_loso.npy', P_v1)
