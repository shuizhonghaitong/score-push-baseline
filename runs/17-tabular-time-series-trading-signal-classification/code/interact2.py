import pandas as pd, numpy as np
from sklearn.metrics import average_precision_score as ap
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

def rep(name, fn):
    aps = []
    for (a, b) in folds:
        g_ = (LR[a:b] >= PROM).astype(float)
        aps.append(ap(Y[a:b], g_ * fn(LR[a:b], R.values[a:b])))
    print(f"{name:38s} mean={np.mean(aps):.4f} last3={[round(x,3) for x in aps[-3:]]}")

rep("lr only", lambda lr, r1: 1 + lr)
for thr in (0.015, 0.02, 0.03, 0.05):
    for w in (1.0, 2.0, 3.0):
        rep(f"lr + {w}*f*hard({thr})", lambda lr, r1, w=w, thr=thr: 1 + lr + w * np.maximum(-r1, 0) * (lr < thr))
for w in (2.0, 3.0, 5.0):
    for L in (0.015, 0.02, 0.03):
        rep(f"lr + {w}*f*exp(-lr/{L})", lambda lr, r1, w=w, L=L: 1 + lr + w * np.maximum(-r1, 0) * np.exp(-lr / L))

# adaptive causal memory: P(RR) among past gate rows with similar lr & r1
def mem2(W, dw, dr, alpha):
    idx = np.where(LR >= PROM)[0]
    idx = idx[idx < N]
    lrs = LR[idx]; r1s = R.values[idx]; rrs = RR[idx]
    out = np.full(N + 0, np.nan)
    for k, t in enumerate(idx):
        m = (idx >= t - W) & (idx < t)
        if m.sum() == 0:
            out[t] = np.nan; continue
        wgt = np.exp(-np.abs(lrs[m] - lrs[k]) / dw) * np.exp(-np.abs(r1s[m] - r1s[k]) / dr)
        base = rrs[:k].mean() if k else 0.2
        out[t] = (rrs[m] * wgt).sum() / wgt.sum() * (1 - alpha) + base * alpha
    return out
MEM = mem2(336, 0.01, 0.002, 0.3)
def fn_mem(lr, r1):
    return 1 + np.nan_to_num(MEM, nan=0.2)
aps = []
for (a, b) in folds:
    g_ = (LR[a:b] >= PROM).astype(float)
    # gate score: mem value; non-gate 0
    sc = np.where(g_ > 0, 1 + np.nan_to_num(MEM[a:b], nan=0.2), 0.0)
    aps.append(ap(Y[a:b], sc))
print(f"{'mem2 (causal kernel)':38s} mean={np.mean(aps):.4f} last3={[round(x,3) for x in aps[-3:]]}")
# hybrid: lr + mem
aps = []
for (a, b) in folds:
    g_ = (LR[a:b] >= PROM).astype(float)
    sc = g_ * (1 + LR[a:b] + 0.5 * np.nan_to_num(MEM[a:b], nan=0.2))
    aps.append(ap(Y[a:b], sc))
print(f"{'lr + 0.5*mem2':38s} mean={np.mean(aps):.4f} last3={[round(x,3) for x in aps[-3:]]}")
