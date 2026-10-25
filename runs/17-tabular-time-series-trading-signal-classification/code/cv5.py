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

def lgbf(seed):
    return lgb.LGBMClassifier(n_estimators=1500, learning_rate=0.01, num_leaves=5, min_child_samples=60,
                              subsample=0.9, subsample_freq=1, colsample_bytree=0.6, reg_lambda=5.0,
                              random_state=seed, verbose=-1, n_jobs=4)
def xgbf(seed):
    return xgb.XGBClassifier(n_estimators=800, learning_rate=0.02, max_depth=3, subsample=0.9,
                             colsample_bytree=0.7, reg_lambda=2.0, n_jobs=4, random_state=seed,
                             eval_metric="auc", tree_method="hist", verbosity=0)

def seedavg(kind, seeds):
    out = []
    for (a, b) in folds:
        Xtr = make_X(0, a); Xva = make_X(a, b)
        prs = []
        for s in seeds:
            m = (lgbf if kind == "lgb" else xgbf)(s)
            m.fit(Xtr, Y[:a])
            prs.append(m.predict_proba(Xva)[:, 1])
        out.append(np.mean(prs, axis=0))
    return out

res_l5 = seedavg("lgb", [42, 7, 123])
res_x3 = seedavg("xgb", [42, 7])
def report(name, prs):
    aps = [ap(Y[a:b], pr) for (a, b), pr in zip(folds, prs)]
    print(f"{name:18s} mean={np.mean(aps):.4f} last3={[round(x,4) for x in aps[-3:]]} all={[round(x,3) for x in aps]}")
    return aps
report("lgb_l5 3seed", res_l5)
report("xgb_d3 2seed", res_x3)
blend_p = [0.5 * a + 0.5 * b for a, b in zip(res_l5, res_x3)]
report("prob_blend", blend_p)
# rank blend
import scipy.stats as st
rankb = []
for (a, b), pa, pb in zip(folds, res_l5, res_x3):
    ra = st.rankdata(pa); rb = st.rankdata(pb)
    pr = (ra + rb) / 2
    rankb.append(pr)
report("rank_blend", rankb)
