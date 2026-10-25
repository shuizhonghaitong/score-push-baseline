import pandas as pd, numpy as np
exec(open("/mnt/disk3/liuyutong_home/workspace/skill/score-push/runs/17-tabular-time-series-trading-signal-classification/code/cv2.py").read().split('print("features"')[0])

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

WALK = np.zeros(len(P))
for t in range(len(P)):
    pt = P[t]; i = t
    while i > 0 and P[i-1] >= pt: i -= 1
    WALK[t] = t - i
g = (LR >= PROM)
dg = pd.DataFrame({"rr": RR[g], "lr": LR[g], "walk": WALK[g], "r1": R.values[g]})
print("gate rows:", len(dg))
qs = pd.qcut(dg["walk"], 5, labels=["w1","w2","w3","w4","w5"])
print("RR rate by walk-length quintile:")
print(dg.groupby(qs, observed=True)["rr"].agg(["count","mean","min","max"]).to_string())
qs2 = pd.qcut(dg["r1"], 5, labels=["r1..r5"])
print("\nRR rate by r1 quintile:")
print(dg.groupby(qs2, observed=True)["rr"].agg(["count","mean"]).to_string())
# interaction: deep lr + short walk?
dg["deep"] = dg["lr"] >= 0.03
dg["short"] = dg["walk"] <= 24
ct = pd.crosstab(dg["deep"], dg["short"], values=dg["rr"], aggfunc=["mean","count"])
print("\nRR by deep x short-walk:")
print(ct.to_string())
