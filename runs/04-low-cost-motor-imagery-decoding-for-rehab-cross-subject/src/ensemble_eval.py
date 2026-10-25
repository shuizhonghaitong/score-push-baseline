import sys
import numpy as np
sys.path.insert(0, '.')
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import xgboost as xgb
import lightgbm as lgb

X = np.load('../outputs/train_feats.npy')
y = np.load('../outputs/train_labels.npy')
sub = np.load('../outputs/train_subjects.npy')
subjects = sorted(set(sub))

def make_models():
    return [
        ('lr01', make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=3000))),
        ('lr03', make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=3000))),
        ('lr1', make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000))),
        ('xgb', xgb.XGBClassifier(n_estimators=200, learning_rate=0.03, max_depth=2,
                                  subsample=0.8, colsample_bytree=0.6, reg_lambda=5.0,
                                  reg_alpha=1.0, min_child_weight=5,
                                  eval_metric='logloss', n_jobs=4, random_state=0)),
        ('lgbm', lgb.LGBMClassifier(n_estimators=200, learning_rate=0.03, num_leaves=7,
                                    subsample=0.8, colsample_bytree=0.6, reg_lambda=5.0,
                                    min_child_samples=30, n_jobs=4, random_state=0, verbose=-1)),
    ]

# LOSO with probability ensemble
accs_ind = {name: [] for name, _ in make_models()}
accs_ens = []
for s in subjects:
    va = sub == s; tr = ~va
    models = make_models()
    probs = []
    for name, m in models:
        m.fit(X[tr], y[tr])
        p = m.predict_proba(X[va])[:, 1]
        probs.append(p)
        accs_ind[name].append((p > 0.5).astype(int)[(p > 0.5).astype(int) == y[va]].sum() / len(y[va]))
    P = np.mean(probs, axis=0)
    accs_ens.append(((P > 0.5).astype(int) == y[va]).mean())
print('individual LOSO:', {k: round(np.mean(v), 4) for k, v in accs_ind.items()})
print('ensemble LOSO: %.4f' % np.mean(accs_ens))

# subsets of the ensemble
subsets = {
    'lr-only': [0, 1, 2],
    'lr+xgb': [0, 1, 2, 3],
    'lr+lgbm': [0, 1, 2, 4],
    'xgb+lgbm': [3, 4],
}
for name, idx in subsets.items():
    accs = []
    for s in subjects:
        va = sub == s; tr = ~va
        models = make_models()
        P = np.mean([m.fit(X[tr], y[tr]).predict_proba(X[va])[:, 1] for _, m in [models[i] for i in idx]], axis=0)
        accs.append(((P > 0.5).astype(int) == y[va]).mean())
    print(f'{name:<10} ensemble LOSO: {np.mean(accs):.4f}')
