"""Label reverse-engineering + causal baselines."""
import pandas as pd, numpy as np, json
from scipy.signal import find_peaks
from sklearn.metrics import average_precision_score as ap

RUN = "/mnt/disk3/liuyutong_home/workspace/skill/score-push/runs/17-tabular-time-series-trading-signal-classification"
tr = pd.read_csv(f"{RUN}/data/archive_btc_train.csv", parse_dates=["datetime"])
ta = pd.read_csv(f"{RUN}/data/train_additional_features.csv", parse_dates=["datetime"])
df = tr.merge(ta, on="Id", suffixes=("", "_a"))
assert (df["datetime"] == df["datetime_a"]).all()
df = df.drop(columns=["datetime_a"])
y = df["Signal"].values; p = df["prices"].values; n = len(df); pos = int(y.sum())
print(f"rows={n} pos={pos} base={pos/n:.4f}")

logp = np.log(p)
print("\n-- find_peaks on -log(price) --")
for prom in np.arange(0.0085, 0.0126, 0.0002):
    peaks, _ = find_peaks(-logp, prominence=prom)
    pred = np.zeros(n); pred[peaks] = 1
    tp = int(((pred == 1) & (y == 1)).sum()); fp = int(pred.sum()) - tp
    fn = pos - tp
    print(f"prom={prom:.4f} peaks={len(peaks)} tp={tp} fp={fp} fn={fn} ap={ap(y, pred):.4f}")

# best-match search including distance param
print("\n-- best exact match search --")
best = (0, None)
for prom in np.arange(0.009, 0.0115, 0.0001):
    peaks, _ = find_peaks(-logp, prominence=prom)
    pred = np.zeros(n); pred[peaks] = 1
    tp = int(((pred == 1) & (y == 1)).sum())
    if tp > best[0]:
        best = (tp, (prom, len(peaks)))
print("best exact:", best)

# causal left condition
ps = pd.Series(p)
print("\n-- causal left-side condition --")
for drop in (0.008, 0.009, 0.01, 0.011):
    for W in (24, 48, 96, 168, 336):
        roll_max = ps.rolling(W, min_periods=1).max()
        cond = ((roll_max - ps) / roll_max >= drop) & (ps.shift(1) >= ps)
        c = cond.values
        tp = int(((c == 1) & (y == 1)).sum())
        print(f"drop={drop} W={W}: flagged={int(c.sum())} tp={tp} prec={tp/c.sum():.3f} rec={tp/pos:.3f} ap={ap(y, c.astype(float)):.4f}")

# also: left cond without 'last step down'
print("\n-- left cond (drop only, no step) --")
for drop in (0.009, 0.01):
    for W in (48, 96, 168):
        roll_max = ps.rolling(W, min_periods=1).max()
        c = ((roll_max - ps) / roll_max >= drop).values
        tp = int(((c == 1) & (y == 1)).sum())
        print(f"drop={drop} W={W}: flagged={int(c.sum())} tp={tp} prec={tp/c.sum():.3f} rec={tp/pos:.3f} ap={ap(y, c.astype(float)):.4f}")

# right-side analysis: among left-cond rows, which are positives?
W, drop = 96, 0.01
roll_max = ps.rolling(W, min_periods=1).max()
left = (((roll_max - ps) / roll_max >= drop) & (ps.shift(1) >= ps)).values
sub = df[left].reset_index()
print(f"\nleft-cond rows: {left.sum()}")
# future 1% up before future 1% down (relative to p_t), horizon 168h
def right_ok(idx, horizon=168):
    fut = p[idx+1:idx+1+horizon]
    if len(fut) == 0: return np.nan
    up = (fut / p[idx] - 1) >= 0.01
    dn = (fut / p[idx] - 1) <= -0.01
    iu = np.argmax(up) if up.any() else np.inf
    idn = np.argmax(dn) if dn.any() else np.inf
    if np.isinf(iu) and np.isinf(idn): return np.nan
    return int(iu < idn)
ro = np.array([right_ok(i) for i in sub["Id"]])
tab = pd.crosstab(y[left], ro, dropna=False)
print(tab)
