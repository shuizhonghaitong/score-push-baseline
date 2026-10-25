import pandas as pd, numpy as np
from sklearn.metrics import average_precision_score as ap
import lightgbm as lgb, xgboost as xgb
exec(open("/mnt/disk3/liuyutong_home/workspace/skill/score-push/runs/17-tabular-time-series-trading-signal-classification/code/cv2.py").read().split('print("features"')[0])

folds = []
start0 = int(N * 0.40)
K = 10
seglen = (N - start0) // K
for i in range(K):
    a = start0 + i * seglen
    b = N if i == K - 1 else start0 + (i + 1) * seglen
    folds.append((a, b))

CFGS = {
 "lgb_l7c40": lambda: lgb.LGBMClassifier(n_estimators=1200, learning_rate=0.02, num_leaves=7, min_child_samples=40, subsample=0.9, subsample_freq=1, colsample_bytree=0.7, reg_lambda=3.0, random_state=42, verbose=-1, n_jobs=4),
 "lgb_l5c60": lambda: lgb.LGBMClassifier(n_estimators=1500, learning_rate=0.01, num_leaves=5, min_child_samples=60, subsample=0.9, subsample_freq=1, colsample_bytree=0.6, reg_lambda=5.0, random_state=42, verbose=-1, n_jobs=4),
 "lgb_l7c40b": lambda: lgb.LGBMClassifier(n_estimators=1500, learning_rate=0.01, num_leaves=7, min_child_samples=40, subsample=0.9, subsample_freq=1, colsample_bytree=0.7, reg_lambda=3.0, random_state=7, verbose=-1, n_jobs=4),
 "xgb_d3": lambda: xgb.XGBClassifier(n_estimators=800, learning_rate=0.02, max_depth=3, subsample=0.9, colsample_bytree=0.7, reg_lambda=2.0, n_jobs=4, random_state=42, eval_metric="auc", tree_method="hist", verbosity=0),
}
def preds_for(factory):
    out = []
    for (a, b) in folds:
        Xtr = make_X(0, a); Xva = make_X(a, b)
        m = factory(); m.fit(Xtr, Y[:a])
        out.append(m.predict_proba(Xva)[:, 1])
    return out

results = {}
for name, f in CFGS.items():
    prs = preds_for(f)
    aps = [ap(Y[a:b], pr) for (a, b), pr in zip(folds, prs)]
    results[name] = prs
    print(f"{name:10s} mean={np.mean(aps):.4f} last3={[round(x,4) for x in aps[-3:]]}")

for blend, names in [("b_l7_xgb", ["lgb_l7c40", "xgb_d3"]),
                     ("b_l7_l5_xgb", ["lgb_l7c40", "lgb_l5c60", "xgb_d3"]),
                     ("b_all4", ["lgb_l7c40", "lgb_l5c60", "lgb_l7c40b", "xgb_d3"])]:
    prs = [np.mean([results[nm][i] for nm in names], axis=0) for i in range(K)]
    aps = [ap(Y[a:b], pr) for (a, b), pr in zip(folds, prs)]
    print(f"{blend:10s} mean={np.mean(aps):.4f} last3={[round(x,4) for x in aps[-3:]]}")
