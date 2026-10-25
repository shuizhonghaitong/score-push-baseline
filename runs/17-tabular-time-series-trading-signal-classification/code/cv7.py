import pandas as pd, numpy as np
from sklearn.metrics import average_precision_score as ap
import lightgbm as lgb, xgboost as xgb
import scipy.stats as st
exec(open("/mnt/disk3/liuyutong_home/workspace/skill/score-push/runs/17-tabular-time-series-trading-signal-classification/code/cv2.py").read().split('print("features"')[0])

# ---- outcome times for right-side condition (train target only) ----
def outcome_times(p, prom):
    n = len(p)
    up = np.full(n, n); dn = np.full(n, n)
    for t in range(n):
        pt = p[t]
        seg = p[t+1:]
        upv = seg >= pt * np.exp(prom)
        if upv.any(): up[t] = t + 1 + int(np.argmax(upv))
        dnv = seg < pt
        if dnv.any(): dn[t] = t + 1 + int(np.argmax(dnv))
    return up, dn
UP, DN = outcome_times(P, PROM)
RR = (UP < DN)

# extra causal features
vol24 = R.rolling(24).std().values
vol48 = R.rolling(48).std().values
newlow6 = ((S <= S.rolling(6, min_periods=1).min())).astype(float)
NNEW6 = newlow6.rolling(6).sum()
def hst(sb):
    out = np.full(len(sb), np.nan); last = -1
    for i, b in enumerate(sb):
        if b: last = i
        out[i] = (i - last) if last >= 0 else np.nan
    return out
NEWLOW24 = (S <= S.rolling(24, min_periods=1).min()).astype(bool).values
HSD_NL24 = hst(NEWLOW24)
# consecutive hours at/below recent-low episode length: walk length of the LR walk minus 1
WALK = np.zeros(len(P))
for t in range(len(P)):
    pt = P[t]; i = t
    while i > 0 and P[i-1] >= pt: i -= 1
    WALK[t] = t - i
# dip: how far below the walk-reference max is the min of the segment (shape of the drop)
tkv = allf["taker_buy_base_asset_volume"].values; v = allf["volume"].values
vv = pd.Series(v); tv = pd.Series(tkv)
vspike = (vv / vv.rolling(24, min_periods=1).mean())

def make_X2(a, b):
    X = make_X(a, b)
    X["lr_z"] = X["lr"] / (vol24[a:b] + 1e-9)
    X["r1_z"] = X["r1"] / (vol24[a:b] + 1e-9)
    X["nnew6"] = NNEW6.values[a:b]
    X["hsd_nl24"] = HSD_NL24[a:b]
    X["walk"] = WALK[a:b]
    X["walklog"] = np.log1p(WALK[a:b])
    X["vz1"] = vspike.values[a:b]
    X["taker1z"] = (tv / vv).values[a:b] / (tv / vv).rolling(168).std().values[a:b]
    return X

folds = []
start0 = int(N * 0.40)
K = 10
seglen = (N - start0) // K
for i in range(K):
    a = start0 + i * seglen
    b = N if i == K - 1 else start0 + (i + 1) * seglen
    folds.append((a, b))

def lgbf(s, **kw):
    base = dict(n_estimators=1500, learning_rate=0.01, num_leaves=5, min_child_samples=60, subsample=0.9, subsample_freq=1, colsample_bytree=0.6, reg_lambda=5.0, random_state=s, verbose=-1, n_jobs=4)
    base.update(kw)
    return lgb.LGBMClassifier(**base)

def cv1(factory, mem=False):
    aps = []
    for (a, b) in folds:
        Xtr = (make_X2 if mem else make_X)(0, a).fillna(-9.0)
        Xva = (make_X2 if mem else make_X)(a, b).fillna(-9.0)
        m = factory(); m.fit(Xtr, Y[:a])
        aps.append(ap(Y[a:b], m.predict_proba(Xva)[:, 1]))
    return aps

def cv2stage(factory):
    aps = []
    for (a, b) in folds:
        Xtr = make_X2(0, a).fillna(-9.0); Xva = make_X2(a, b).fillna(-9.0)
        gtr = LR[:a] >= PROM; gva = LR[a:b] >= PROM
        mr = factory(); mr.fit(Xtr[gtr], RR[:a][gtr])
        q = mr.predict_proba(Xva)[:, 1]
        pr = np.where(gva, q, 0.0)
        aps.append(ap(Y[a:b], pr))
    return aps

def rep(name, aps):
    print(f"{name:24s} mean={np.mean(aps):.4f} last3={[round(x,4) for x in aps[-3:]]}")

rep("base l5 (v2 feats)", cv1(lambda: lgbf(42)))
rep("l5 + new feats", cv1(lambda: lgbf(42), mem=True))
rep("l5b + new feats", cv1(lambda: lgbf(7), mem=True))
rep("2stage right l5", cv2stage(lambda: lgbf(42)))
rep("2stage right l5b", cv2stage(lambda: lgbf(7)))
rep("2stage right l7c40", cv2stage(lambda: lgbf(42, num_leaves=7, min_child_samples=40, reg_lambda=3.0, colsample_bytree=0.7)))
