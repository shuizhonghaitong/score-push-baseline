import pandas as pd, numpy as np
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
SURV1 = (P[1:] >= P[:-1])  # next hour >= p_t

g = (LR[:N] >= PROM)
surv1 = np.full(N, np.nan); surv1[:N-1] = SURV1[:N-1]
dg = pd.DataFrame({
    "rr": RR[:N][g], "lr": LR[:N][g],
    "r1": R.values[:N][g],
    "surv1": surv1[g],
})
dg = dg.dropna(subset=["surv1"])
print("gate rows:", len(dg))
print("P(survive 1h | gate):", dg["surv1"].mean())
print("P(RR | gate & surv1):", dg["rr"][dg["surv1"]].mean())
print("P(RR | gate & !surv1):", dg["rr"][dg["surv1"] == 0].mean())

print("\n-- P(RR) by lr bucket x r1 bucket (within gate) --")
lrq = pd.qcut(dg["lr"], 4, labels=["L1","L2","L3","L4"])
r1b = pd.cut(dg["r1"], [-9, -0.01, -0.005, -0.002, 0, 1], labels=["d>=1%",".5-1%",".2-.5%",".0-.2%","up"])
ct = pd.crosstab(lrq, r1b, values=dg["rr"], aggfunc=["mean","count"])
print(ct.to_string())

print("\n-- P(surv1) by r1 bucket --")
print(dg.groupby(r1b, observed=True)["surv1"].agg(["count","mean"]).to_string())
