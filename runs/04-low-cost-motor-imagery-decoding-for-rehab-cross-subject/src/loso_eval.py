"""Leave-one-subject-out evaluation with pluggable features and models."""
from __future__ import annotations

import sys, time
import numpy as np
sys.path.insert(0, '.')
from modelzoo import make_model
from features2 import csp_transform

Xv1 = np.load('../outputs/train_feats.npy')
Xv2 = np.load('../outputs/feats_v2.npy')
Xcsp = np.load('../outputs/Xcsp.npy')
y = np.load('../outputs/y.npy')
sub = np.load('../outputs/sub.npy')
subjects = sorted(set(sub))

def get_feats(kind, tr_idx, va_idx):
    if kind == 'v1':
        return Xv1[tr_idx], Xv1[va_idx]
    if kind == 'v2':
        return Xv2[tr_idx], Xv2[va_idx]
    if kind == 'csp':
        F = csp_transform(Xcsp[tr_idx], y[tr_idx])
        Fv = csp_transform(Xcsp[va_idx], y[va_idx])  # transform only; y unused in transform math
        return F, Fv
    if kind == 'v2+csp':
        F = np.concatenate([Xv2[tr_idx], csp_transform(Xcsp[tr_idx], y[tr_idx])], 1)
        Fv = np.concatenate([Xv2[va_idx], csp_transform(Xcsp[va_idx], y[va_idx])], 1)
        return F, Fv
    if kind == 'v1+csp':
        F = np.concatenate([Xv1[tr_idx], csp_transform(Xcsp[tr_idx], y[tr_idx])], 1)
        Fv = np.concatenate([Xv1[va_idx], csp_transform(Xcsp[va_idx], y[va_idx])], 1)
        return F, Fv
    raise ValueError(kind)


def loso(fkind, mkind, **mp):
    per_sub = {}
    for s in subjects:
        va = sub == s
        tr = ~va
        Ftr, Fva = get_feats(fkind, np.where(tr)[0], np.where(va)[0])
        m = make_model(mkind, **mp)
        m.fit(Ftr, y[tr])
        pred = m.predict(Fva)
        per_sub[s] = (pred == y[va]).mean()
    vals = list(per_sub.values())
    return np.mean(vals), per_sub


if __name__ == '__main__':
    configs = [
        ('v1', 'lr'), ('v2', 'lr'), ('v1', 'xgb'), ('v2', 'xgb'),
        ('v1', 'lgbm'), ('v2', 'lgbm'), ('v2', 'svm'),
        ('csp', 'lr'), ('v2+csp', 'lr'), ('v2+csp', 'xgb'), ('v2+csp', 'lgbm'),
    ]
    if len(sys.argv) > 1:
        configs = [tuple(sys.argv[1:3])]
    results = {}
    for fkind, mkind in configs:
        t0 = time.time()
        mean, per_sub = loso(fkind, mkind)
        results[(fkind, mkind)] = (mean, per_sub)
        best = sorted(per_sub, key=per_sub.get)
        print(f'{fkind:8s} {mkind:5s} mean={mean:.4f} ({time.time()-t0:.0f}s) '
              f'worst={best[0]}:{per_sub[best[0]]:.2f} {best[1]}:{per_sub[best[1]]:.2f} '
              f'best={best[-1]}:{per_sub[best[-1]]:.2f}')
    np.save('../outputs/loso_results.npy', {k: v for k, v in results.items()}, allow_pickle=True)
