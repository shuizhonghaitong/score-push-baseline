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

def lgbf(s): return lgb.LGBMClassifier(n_estimators=1500, learning_rate=0.01, num_leaves=5, min_child_samples=60, subsample=0.9, subsample_freq=1, colsample_bytree=0.6, reg_lambda=5.0, random_state=s, verbose=-1, n_jobs=4)
def xgbf(s): return xgb.XGBClassifier(n_estimators=800, learning_rate=0.02, max_depth=3, subsample=0.9, colsample_bytree=0.7, reg_lambda=2.0, n_jobs=4, random_state=s, eval_metric="auc", tree_method="hist", verbosity=0)

l5s, x3s, gts = [], [], []
for (a, b) in folds:
    Xtr = make_X(0, a).fillna(-9.0); Xva = make_X(a, b).fillna(-9.0)
    prs = []
    for s in (42, 7, 123):
        m = lgbf(s); m.fit(Xtr, Y[:a]); prs.append(m.predict_proba(Xva)[:, 1])
    prs2 = []
    for s in (42, 7):
        m = xgbf(s); m.fit(Xtr, Y[:a]); prs2.append(m.predict_proba(Xva)[:, 1])
    l5 = np.mean(prs, axis=0); x3 = np.mean(prs2, axis=0)
    l5s.append(l5); x3s.append(x3); gts.append((LR[a:b] >= PROM).astype(float))

def ev(name, score):
    aps = [ap(Y[a:b], sc) for (a, b), sc in zip(folds, score)]
    # gate-in-top500 fraction
    g = []
    for sc, gt in zip(score, gts):
        top500 = np.argsort(sc)[::-1][:500]
        g.append(gt[top500].sum() / max(1, gt.sum()))
    print(f"{name:16s} mean={np.mean(aps):.4f} last3={[round(x,4) for x in aps[-3:]]} gateInTop500={[round(x,2) for x in g]}")

base_blend = [(st.rankdata(a) + st.rankdata(b)) / 2 for a, b in zip(l5s, x3s)]
ev("blend", base_blend)
bn = [ (sc - sc.min()) / (sc.max() - sc.min()) for sc in base_blend ]
ev("hyb .5/.5", [0.5 * s + 0.5 * g for s, g in zip(bn, gts)])
ev("hyb .3/.7", [0.3 * s + 0.7 * g for s, g in zip(bn, gts)])
ev("hyb .7/.3", [0.7 * s + 0.3 * g for s, g in zip(bn, gts)])
ev("mult 1+g", [s * (1 + g) for s, g in zip(bn, gts)])
ev("gate only", gts)

hn = [(st.rankdata(s).astype(float) / len(s)) for s in base_blend]
hard = [10 * g + 0.001 * h for g, h in zip(gts, hn)]
ev("HARD gate+model", hard)
hard0 = [10 * g for g in gts]
ev("HARD gate uniform", hard0)
# model rank only within gate, 0 outside
wg = []
for s, g in zip(bn, gts):
    rr = st.rankdata(np.where(g > 0, s, -1))
    wg.append(np.where(g > 0, rr, 0))
ev("rank-within-gate", wg)
