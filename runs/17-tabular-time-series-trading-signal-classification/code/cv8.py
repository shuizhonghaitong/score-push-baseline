import pandas as pd, numpy as np
from sklearn.metrics import average_precision_score as ap
import lightgbm as lgb, xgboost as xgb
import scipy.stats as st
exec(open("/mnt/disk3/liuyutong_home/workspace/skill/score-push/runs/17-tabular-time-series-trading-signal-classification/code/cv2.py").read().split('print("features"')[0])

folds = []
start0 = int(N * 0.40)
K = 10
seglen = (N - start0) // K
for i in range(K):
    a = start0 + i * seglen
    b = N if i == K - 1 else start0 + (i + 1) * seglen
    folds.append((a, b))

def lgbf(s):
    return lgb.LGBMClassifier(n_estimators=1500, learning_rate=0.01, num_leaves=5, min_child_samples=60, subsample=0.9, subsample_freq=1, colsample_bytree=0.6, reg_lambda=5.0, random_state=s, verbose=-1, n_jobs=4)
def xgbf(s):
    return xgb.XGBClassifier(n_estimators=800, learning_rate=0.02, max_depth=3, subsample=0.9, colsample_bytree=0.7, reg_lambda=2.0, n_jobs=4, random_state=s, eval_metric="auc", tree_method="hist", verbosity=0)

model_rank = []
for (a, b) in folds:
    Xtr = make_X(0, a).fillna(-9.0); Xva = make_X(a, b).fillna(-9.0)
    prs = [lgbf(s).fit(Xtr, Y[:a]).predict_proba(Xva)[:, 1] for s in (42, 7, 123)]
    prs += [xgbf(s).fit(Xtr, Y[:a]).predict_proba(Xva)[:, 1] for s in (42, 7)]
    l5 = np.mean(prs[:3], axis=0); x3 = np.mean(prs[3:], axis=0)
    model_rank.append((st.rankdata(l5) + st.rankdata(x3)) / 2)

vol24 = R.rolling(24).std().values
def hard_gate_within(within_fn):
    aps = []
    for (a, b), mr in zip(folds, model_rank):
        g = (LR[a:b] >= PROM).astype(float)
        w = within_fn(a, b)
        score = g * 1000 + w * g
        aps.append(ap(Y[a:b], score))
    return aps

def rep(name, aps):
    print(f"{name:26s} mean={np.mean(aps):.4f} last3={[round(x,4) for x in aps[-3:]]}")

rep("within=lr", hard_gate_within(lambda a, b: LR[a:b]))
rep("within=lr_z", hard_gate_within(lambda a, b: LR[a:b] / (vol24[a:b] + 1e-9)))
rep("within=-|r1|", hard_gate_within(lambda a, b: -np.abs(R.values[a:b])))
rep("within=model", hard_gate_within(lambda a, b: model_rank[-1]))
# blend model rank + lr rank within gate
def hybrid():
    aps = []
    for (a, b), mr in zip(folds, model_rank):
        g = (LR[a:b] >= PROM).astype(float)
        rl = st.rankdata(LR[a:b]).astype(float)
        rm = st.rankdata(mr).astype(float)
        for w in (0.5, 0.6, 0.7):
            pass
        w = w * rl + (1 - w) * rm
        aps.append(ap(Y[a:b], g * 1000 + w * g))
    return aps
w = 0.6
aps = []
for (a, b), mr in zip(folds, model_rank):
    g = (LR[a:b] >= PROM).astype(float)
    rl = st.rankdata(LR[a:b]).astype(float)
    rm = st.rankdata(mr).astype(float)
    wgt = w * rl + (1 - w) * rm
    aps.append(ap(Y[a:b], g * 1000 + wgt * g))
print(f"{'hybrid .6lr/.4model':26s} mean={np.mean(aps):.4f} last3={[round(x,4) for x in aps[-3:]]}")
