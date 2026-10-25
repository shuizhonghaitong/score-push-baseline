import sys, time
import numpy as np
sys.path.insert(0, '.')
from modelzoo import make_model
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

X = np.load('../outputs/train_feats.npy')
y = np.load('../outputs/train_labels.npy')
sub = np.load('../outputs/train_subjects.npy')
subjects = sorted(set(sub))
FIDX = {name: i for i, name in enumerate(
    [f'logpow_{b}_{c}' for c in ['Fz','C3','Cz','C4','PO7','Pz','PO8','Oz'] for b in ['delta','theta','mu','beta','beta2']]
    + [f'erd_{b}_{c}' for b in ['mu','beta','beta2'] for c in ['Fz','C3','Cz','C4','PO7','Pz','PO8','Oz']]
    + [f'rel_{b}' for b in ['delta','theta','mu','beta','beta2']]
    + [f'dyn_{b}_{c}' for b in ['mu','beta'] for c in ['C3','Cz','C4']]
    + ['cent_post_mu'])}

def loso_feats(feats, model_fn):
    accs = []
    for s in subjects:
        va = sub == s; tr = ~va
        m = model_fn()
        m.fit(feats[tr], y[tr])
        accs.append((m.predict(feats[va]) == y[va]).mean())
    return np.mean(accs), np.array(accs)

# 1. C sweep
for C in [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]:
    mean, _ = loso_feats(X, lambda: make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000)))
    print(f'v1 lr C={C:<6} mean={mean:.4f}')

# 2. feature subsets
erdc = [FIDX[f'erd_{b}_{c}'] for b in ['mu','beta','beta2'] for c in ['C3','Cz','C4']]
erdall = [FIDX[f'erd_{b}_{c}'] for b in ['mu','beta','beta2'] for c in ['Fz','C3','Cz','C4','PO7','Pz','PO8','Oz']]
logpw_c = [FIDX[f'logpow_{b}_{c}'] for b in ['mu','beta','beta2'] for c in ['C3','Cz','C4']]
rel = [FIDX[f'rel_{b}'] for b in ['delta','theta','mu','beta','beta2']]
dyn = [FIDX[f'dyn_{b}_{c}'] for b in ['mu','beta'] for c in ['C3','Cz','C4']]
for name, idx in [('erd_central', erdc), ('erd_all', erdall),
                  ('erd_central+rel', erdc+rel), ('erd_central+dyn', erdc+dyn),
                  ('erd_all+logpw_c+rel+dyn', erdall+logpw_c+rel+dyn)]:
    mean, accs = loso_feats(X[:, idx], lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000)))
    print(f'{name:<24} mean={mean:.4f}')
