import pandas as pd, numpy as np
from scipy.signal import find_peaks
from sklearn.metrics import average_precision_score as ap

RUN = "/mnt/disk3/liuyutong_home/workspace/skill/score-push/runs/17-tabular-time-series-trading-signal-classification"
tr = pd.read_csv(f"{RUN}/data/archive_btc_train.csv", parse_dates=["datetime"])
y = tr["Signal"].values; p = tr["prices"].values; n = len(y)

def left_rise(p, prom=None):
    """ln rise from left reference to trough: log(p_leftmax/p_t), where p_leftmax =
    max price between the last point left of t with price < p_t (exclusive) and t."""
    out = np.full(len(p), np.nan)
    for t in range(len(p)):
        pt = p[t]
        i = t
        while i > 0 and p[i-1] >= pt:
            i -= 1
        seg = p[i:t+1]
        out[t] = np.log(seg.max() / pt)
    return out

def right_rise(p, prom=None):
    """ln rise on right: max price between t (exclusive) and first point with price < p_t, / p_t.
    If price never goes below p_t, use max over rest of series (truncated)."""
    out = np.full(len(p), np.nan)
    for t in range(len(p)):
        pt = p[t]
        j = t + 1
        while j < len(p) and p[j] >= pt:
            j += 1
        seg = p[t+1:j+1]
        if len(seg) == 0:
            continue
        out[t] = np.log(seg.max() / pt)
    return out

lr = left_rise(p)
rr = right_rise(p)

# is label == (lr >= prom) & (rr >= prom)? search prom
print("-- exact rule search: (left>=prom) & (right>=prom) --")
for prom in np.arange(0.0098, 0.0103, 0.00005):
    m = (lr >= prom) & (rr >= prom)
    tp = int(((m) & (y == 1)).sum()); fp = int(m.sum() - tp); fn = int(y.sum() - tp)
    if tp >= 221:
        print(f"prom={prom:.5f} tp={tp} fp={fp} fn={fn} ap={ap(y, m.astype(float)):.4f}")

# fine grid
best = (-1, None)
for prom in np.arange(0.0095, 0.0110, 0.00001):
    m = (lr >= prom) & (rr >= prom)
    d = int(((m == 1) & (y == 1)).sum() + ((m == 0) & (y == 0)).sum())
    if d > best[0]:
        best = (d, prom)
print("best match (agreement):", best, "of", n)

prom = best[1]
m = ((lr >= prom) & (rr >= prom))
fp_idx = np.where((m == 1) & (y == 0))[0]
fn_idx = np.where((m == 0) & (y == 1))[0]
print("false positives:", [(int(i), tr['datetime'].iloc[i], round(p[i],2)) for i in fp_idx])
print("false negatives:", [(int(i), tr['datetime'].iloc[i], round(p[i],2), round(lr[i],5), round(rr[i],5)) for i in fn_idx])
for i in fn_idx:
    t = int(i)
    print("fn context:", tr['datetime'].iloc[t-3:t+4].astype(str).tolist(), p[t-3:t+4].round(2).tolist())

# strict right-side crosstab among left rows
left = (lr >= 0.01)
sub = tr[left].reset_index(drop=True)
print(f"\nleft rows (lr>=0.01): {left.sum()}")
# right condition strict: rise >= exp(0.01)-1 before price < p_t, using our rr
rr_strict = (rr >= 0.01) & left
ct = pd.crosstab(y[left], rr_strict)
print(ct)
print("P(right | left) =", float(rr_strict.mean()))

# how far to the right does the rise happen for positives vs negs?
def right_lag(p, t, thr=0.01):
    pt = p[t]
    for j in range(t+1, len(p)):
        if p[j] < pt:
            return -1  # failed: price dipped below p_t before rise
        if p[j]/pt - 1 >= thr:
            return j - t
    return -1
lags = [right_lag(p, i) for i in np.where(left)[0]]
lag_pos = [l for l, yy in zip(lags, y[left]) if yy == 1]
lag_neg_ok = [l for l, yy in zip(lags, y[left]) if yy == 0 and l >= 0]
lag_neg_fail = [l for l, yy in zip(lags, y[left]) if yy == 0 and l < 0]
print("pos lags: median", np.median(lag_pos), "max", max(lag_pos), "n", len(lag_pos))
print("neg: failed(dip first)", len(lag_neg_fail), " rose>=1% before dip:", len(lag_neg_ok))
