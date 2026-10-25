import pandas as pd, numpy as np
from sklearn.metrics import average_precision_score as ap
import lightgbm as lgb, xgboost as xgb
exec(open("/mnt/disk3/liuyutong_home/workspace/skill/score-push/runs/17-tabular-time-series-trading-signal-classification/code/cv2.py").read().split('print("features"')[0])

# ---- right-side outcome times (up_time, dn_time) for every row ----
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
RR = (UP < DN)  # true right-side condition
print("RR rate overall:", RR.mean(), " among LR>=PROM:", RR[LR>=PROM].mean())

def memory_features(p, up, dn, lr, r1, W, state=None, smooth=20):
    """causal: for each t, outcome rate among past W rows whose outcome is known by t."""
    n = len(p)
    out = np.full(n, np.nan)
    for t in range(W + 1, n):
        idx = np.arange(t - W, t)
        known = (up[idx] < t) | (dn[idx] < t)
        if state is not None:
            mstate = state(idx)
        else:
            mstate = np.ones(len(idx), bool)
        sel = known & mstate
        tot = sel.sum()
        succ = sel & (up[idx] < dn[idx])
        out[t] = (succ.sum() + smooth * RR[:t].mean() if t > 0 else 0) / (tot + smooth)
    return out

r1v = R.values
gate_state = lambda idx: LR[idx] >= PROM
flush_state = lambda idx: r1v[idx] < -0.005

MEM = {}
for W in (72, 168, 336):
    MEM[f"memg{W}"] = memory_features(P, UP, DN, LR, r1v, W, gate_state)
    MEM[f"memf{W}"] = memory_features(P, UP, DN, LR, r1v, W, flush_state)
    MEM[f"memall{W}"] = memory_features(P, UP, DN, LR, r1v, W, None)
# hours since last full positive (known by t: up<dn<t)
def hsd_positive(up, dn):
    n = len(up); out = np.full(n, np.nan); last = -1
    for t in range(n):
        mask = (up < dn) & (dn < t) | ((up < dn) & (up < t))
        m2 = (up < dn) & (np.minimum(up, dn) < t)
        if m2.any(): last = t - 1 - int(np.argmax(m2[::-1]))
        out[t] = (t - last) if last >= 0 else np.nan
    return out
HSP = hsd_positive(UP, DN)

def make_X2(a, b):
    X = make_X(a, b)
    for k, v in MEM.items():
        X[k] = v[a:b]
    X["hsp"] = HSP[a:b]
    return X

folds = []
start0 = int(N * 0.40)
K = 10
seglen = (N - start0) // K
for i in range(K):
    a = start0 + i * seglen
    b = N if i == K - 1 else start0 + (i + 1) * seglen
    folds.append((a, b))

def eval_cfg(name, factory, mem=False, oracle=False):
    aps = []
    for (a, b) in folds:
        Xtr = make_X2(0, a) if mem else make_X(0, a)
        Xva = make_X2(a, b) if mem else make_X(a, b)
        if oracle:
            nxt = np.log(P[1:] / P[:-1])
            Xtr = Xtr.copy(); Xva = Xva.copy()
            Xtr["ORAC_next1"] = nxt[:a]
            Xva["ORAC_next1"] = nxt[a:b]
            Xtr["ORAC_rr"] = RR[:a]; Xva["ORAC_rr"] = RR[a:b]
        m = factory(); m.fit(Xtr, Y[:a])
        aps.append(ap(Y[a:b], m.predict_proba(Xva)[:, 1]))
    print(f"{name:22s} mean={np.mean(aps):.4f} last3={[round(x,4) for x in aps[-3:]]}")
    return aps

L5 = lambda: lgb.LGBMClassifier(n_estimators=1500, learning_rate=0.01, num_leaves=5, min_child_samples=60, subsample=0.9, subsample_freq=1, colsample_bytree=0.6, reg_lambda=5.0, random_state=42, verbose=-1, n_jobs=4)
X3 = lambda: xgb.XGBClassifier(n_estimators=800, learning_rate=0.02, max_depth=3, subsample=0.9, colsample_bytree=0.7, reg_lambda=2.0, n_jobs=4, random_state=42, eval_metric="auc", tree_method="hist", verbosity=0)

print("== baseline (no mem) ==")
eval_cfg("lgb_l5c60", L5)
eval_cfg("xgb_d3", X3)
print("== with causal memory features ==")
eval_cfg("lgb_l5c60+mem", L5, mem=True)
eval_cfg("xgb_d3+mem", X3, mem=True)
print("== oracle ceiling (leaked future) ==")
eval_cfg("lgb_l5c60+ORACLE", L5, oracle=True)
