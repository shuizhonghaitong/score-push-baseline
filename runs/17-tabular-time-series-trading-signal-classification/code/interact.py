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

# 1) does the lr x r1 interaction generalize across time?
g = (LR[:N] >= PROM)
dg = pd.DataFrame({"rr": RR[:N][g], "lr": LR[:N][g], "r1": R.values[:N][g], "t": np.where(g[:N])[0]})
early = dg[dg["t"] < int(N*0.6)]
late = dg[dg["t"] >= int(N*0.6)]
for name, d in [("early", early), ("late", late)]:
    big = d["r1"] <= -0.01
    shal = d["lr"] < 0.02
    print(f"{name}: RR big-drop&shallow={d['rr'][big & shal].mean():.3f} (n={int((big & shal).sum())}) | big-drop&deep={d['rr'][big & ~shal].mean():.3f} (n={int((big & ~shal).sum())}) | small-drop={d['rr'][~big].mean():.3f} (n={int((~big).sum())})")

# 2) CV: composite within-gate scores
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
    print(f"{name:34s} mean={np.mean(aps):.4f} last3={[round(x,3) for x in aps[-3:]]}")

rep("lr only", lambda lr, r1: 1 + lr)
for w in (0.5, 1.0, 2.0, 4.0, 8.0):
    rep(f"lr + {w}*max(-r1,0)", lambda lr, r1, w=w: 1 + lr + w * np.maximum(-r1, 0))
for w in (0.5, 1.0, 2.0):
    rep(f"lr + {w}*max(-r1,0)*shallow", lambda lr, r1, w=w: 1 + lr + w * np.maximum(-r1, 0) * (lr < 0.02))
