import pandas as pd, numpy as np
from sklearn.metrics import average_precision_score as ap

RUN = "/mnt/disk3/liuyutong_home/workspace/skill/score-push/runs/17-tabular-time-series-trading-signal-classification"
tr = pd.read_csv(f"{RUN}/data/archive_btc_train.csv", parse_dates=["datetime"])
y = tr["Signal"].values; p = tr["prices"].values; n = len(y)
PROM = 0.00995

def left_rise(p):
    out = np.full(len(p), np.nan)
    for t in range(len(p)):
        pt = p[t]; i = t
        while i > 0 and p[i-1] >= pt: i -= 1
        out[t] = np.log(p[i:t+1].max() / pt)
    return out

def right_rise(p):
    out = np.full(len(p), np.nan)
    for t in range(len(p)):
        pt = p[t]; j = t + 1
        while j < len(p) and p[j] >= pt: j += 1
        seg = p[t+1:j+1]
        if len(seg): out[t] = np.log(seg.max() / pt)
    return out

lr = left_rise(p); rr = right_rise(p)
gate = lr >= PROM
print(f"gate rows: {gate.sum()}, positives in gate: {int((gate & (y==1)).sum())}, P(right|gate)={y[gate].mean():.4f}")
print("AP of uniform gate:", ap(y, gate.astype(float)))

# P(right) by lr bucket among gate rows
rrg = (rr >= PROM)
for lo, hi in [(PROM, 0.015), (0.015, 0.02), (0.02, 0.03), (0.03, 0.05), (0.05, 1)]:
    m = (lr >= lo) & (lr < hi)
    if m.sum():
        print(f"lr in [{lo:.3f},{hi:.3f}): n={m.sum()} P(right)={rrg[m].mean():.3f} pos={int((m&(y==1)).sum())}")

# P(right) by drop size just before (r1), hour of day
df = tr.copy()
df["lr"] = lr; df["rr"] = rr; df["gate"] = gate; df["y"] = y
g = df[df["gate"]]
df["r1"] = np.log(df["prices"] / df["prices"].shift(1))
g = g.copy(); g["r1"] = np.log(df["prices"].shift(0) / df["prices"].shift(1)).values[gate]
print("\nP(right) by last-step drop bucket (gate rows):")
for lo, hi in [(-9, -0.005), (-0.005, -0.002), (-0.002, -0.001), (-0.001, 0), (0, 9)]:
    m = (g["r1"] >= lo) & (g["r1"] < hi)
    if m.sum(): print(f"  r1 in [{lo},{hi}): n={m.sum()} P(right)={g['y'][m].mean():.3f}")
print("P(right) by hour of day (gate rows):")
g["hour"] = df["datetime"].dt.hour.values[gate]
hh = g.groupby("hour")["y"].agg(["count", "mean"])
print(hh[hh["count"] >= 20].to_string())
print("P(right) by weekday (gate rows):")
g["dow"] = df["datetime"].dt.dayofweek.values[gate]
print(g.groupby("dow")["y"].agg(["count","mean"]).to_string())

# rolling vol of past returns vs P(right)
r = np.log(p / np.r_[p[0], p[:-1]])
vol24 = pd.Series(r).rolling(24).std().values
g2 = g.copy(); g2["vol24"] = vol24[gate]
qs = pd.qcut(g2["vol24"], 4, labels=["q1","q2","q3","q4"])
print("\nP(right) by 24h-vol quartile (gate rows):")
print(g2.groupby(qs, observed=True)["y"].agg(["count","mean"]).to_string())

# taker buy ratio (from additional features)
ta = pd.read_csv(f"{RUN}/data/train_additional_features.csv")
tk = (ta["taker_buy_base_asset_volume"] / ta["volume"]).values
g3 = g.copy(); g3["tk"] = tk[gate]
qs3 = pd.qcut(g3["tk"], 4, labels=["q1","q2","q3","q4"])
print("\nP(right) by taker-buy-ratio quartile (gate rows):")
print(g3.groupby(qs3, observed=True)["y"].agg(["count","mean"]).to_string())
