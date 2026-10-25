import pandas as pd, numpy as np
from sklearn.metrics import average_precision_score as ap
import lightgbm as lgb
import scipy.stats as st
exec(open("code/cv2.py").read().split('print("features"')[0])

def outcome_times(p, prom):
    n = len(p); up = np.full(n, n); dn = np.full(n, n)
    for t in range(n):
        pt = p[t]; seg = p[t+1:]
        upv = seg >= pt * np.exp(prom)
        if upv.any(): up[t] = t + 1 + int(np.argmax(upv))
        dnv = seg < pt
        if dnv.any(): dn[t] = t + 1 + int(np.argmax(dnv))
    return up, dn
UP, DN = outcome_times(P, PROM)
RR = (UP < DN)

folds = []
start0 = int(N * 0.40); K = 10; seglen = (N - start0) // K
for i in range(K):
    a = start0 + i * seglen; b = N if i == K - 1 else start0 + (i + 1) * seglen
    folds.append((a, b))

FEATS = ["lr", "r1", "r1_abs", "vol24", "vol48", "taker1", "taker24", "hsd05", "hour", "dow"]
def lgbf(s):
    return lgb.LGBMClassifier(n_estimators=600, learning_rate=0.01, num_leaves=3, min_child_samples=80,
                              subsample=0.9, subsample_freq=1, colsample_bytree=0.8, reg_lambda=10.0,
                              random_state=s, verbose=-1, n_jobs=4)

base = []
for lam in (0.05, 0.1, 0.2):
    aps = []
    for (a, b) in folds:
        Xtr = make_X(0, a); Xva = make_X(a, b)
        gtr = LR[:a] >= PROM
        m = lgbf(42); m.fit(Xtr[gtr][FEATS].fillna(-9.0), RR[:a][gtr])
        q = m.predict_proba(Xva[FEATS].fillna(-9.0))[:, 1]
        gva = LR[a:b] >= PROM
        rl = st.rankdata(np.where(gva, LR[a:b], -1)).astype(float)
        rq = st.rankdata(np.where(gva, q, -1)).astype(float)
        score = gva * (rl + lam * rq)
        aps.append(ap(Y[a:b], score))
    print(f"lr-rank + {lam}*model-rank (within gate)  mean={np.mean(aps):.4f} last3={[round(x,3) for x in aps[-3:]]}")
aps = [ap(Y[a:b], (LR[a:b] >= PROM).astype(float) * (1 + LR[a:b])) for a, b in folds]
print(f"lr only (ref)                     mean={np.mean(aps):.4f}")
