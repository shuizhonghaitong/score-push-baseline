"""Expanding-window CV for causal models. All features use data <= t only."""
import pandas as pd, numpy as np, json
from sklearn.metrics import average_precision_score as ap
import lightgbm as lgb, xgboost as xgb

RUN = "/mnt/disk3/liuyutong_home/workspace/skill/score-push/runs/17-tabular-time-series-trading-signal-classification"
PROM = 0.00995

tr = pd.read_csv(f"{RUN}/data/archive_btc_train.csv", parse_dates=["datetime"])
ta = pd.read_csv(f"{RUN}/data/train_additional_features.csv", parse_dates=["datetime"])
te = pd.read_csv(f"{RUN}/data/archive_btc_test.csv", parse_dates=["datetime"])
tea = pd.read_csv(f"{RUN}/data/test_additional_features.csv", parse_dates=["datetime"])
trf = tr.merge(ta, on="Id", suffixes=("", "_a"))
tf = te.merge(tea, on="Id", suffixes=("", "_a"))
assert (trf["datetime"] == trf["datetime_a"]).all() and (tf["datetime"] == tf["datetime_a"]).all()
trf = trf.drop(columns=["datetime_a"]); tf = tf.drop(columns=["datetime_a"])

allf = pd.concat([trf, tf], ignore_index=True)
allf = allf.sort_values("datetime").reset_index(drop=True)
P = allf["prices"].values
Y = trf["Signal"].values
N, NT = len(trf), len(tf)
DT = allf["datetime"].values

def left_rise_series(p):
    out = np.full(len(p), -1e9)
    for t in range(len(p)):
        pt = p[t]; i = t
        while i > 0 and p[i-1] >= pt: i -= 1
        out[t] = np.log(p[i:t+1].max() / pt)
    return out

LR = left_rise_series(P)

def make_X(p_all, lr, start, end, tk, vol, nof):
    """causal features for rows [start, end) using series p_all (full history allowed left of start)"""
    seg_p = pd.Series(p_all)
    r = np.log(seg_p / seg_p.shift(1))
    idx = slice(start, end)
    X = pd.DataFrame(index=range(start, end))
    X["lr"] = lr[idx]
    X["lr_margin"] = lr[idx] - PROM
    for lag in (1, 2, 3, 4, 6, 8, 12, 24):
        X[f"r{lag}"] = r.shift(lag - 1).values[idx]
    for w in (6, 12, 24, 48, 168):
        X[f"vol{w}"] = r.rolling(w).std().values[idx]
        X[f"ret{w}"] = (seg_p / seg_p.shift(w) - 1).values[idx]
    # crossings below current price in last w hours (choppiness)
    for w in (6, 24):
        X[f"below{w}"] = (seg_p.shift(1) < seg_p).rolling(w).sum().values[idx]
    # time since last hour where price dropped >=0.5%
    bigdrop = (r <= -0.005).astype(float)
    X["hours_since_drop05"] = bigdrop[::-1].cummax()  # placeholder, fixed below
    # taker ratio / volume
    v = pd.Series(vol); tkv = pd.Series(tk); nofv = pd.Series(nof)
    X["taker1"] = (tkv / v).values[idx]
    X["taker24"] = (tkv.rolling(24).sum() / v.rolling(24).sum()).values[idx]
    X["lv24"] = (np.log1p(v) - np.log1p(v).rolling(168).mean()).values[idx]
    X["ltn24"] = (np.log1p(nofv) - np.log1p(nofv).rolling(168).mean()).values[idx]
    # calendar
    d = pd.DatetimeIndex(DT[start:end])
    X["hour"] = d.hour; X["dow"] = d.dayofweek
    X["h_sin"] = np.sin(2 * np.pi * d.hour / 24); X["h_cos"] = np.cos(2 * np.pi * d.hour / 24)
    return X

def hours_since_true(series_bool):
    """hours since last True (causal)"""
    idx = np.arange(len(series_bool))
    out = np.full(len(series_bool), np.nan)
    last = np.nan
    for i, b in enumerate(series_bool):
        if b: last = i
        out[i] = (i - last) if not np.isnan(last) else np.nan
    return out

tk = allf["taker_buy_base_asset_volume"].values
vol = allf["volume"].values
nof = allf["number_of_trades"].values
bigdrop = (np.log(P / np.r_[P[0], P[:-1]]) <= -0.005)
HSD = hours_since_true(bigdrop)

folds = []
start0 = int(N * 0.60)
seglen = (N - start0) // 4
for i in range(4):
    a = start0 + i * seglen
    b = N if i == 3 else start0 + (i + 1) * seglen
    folds.append((a, b))
print("folds:", folds, "N:", N)

def add_hsd(X):
    X = X.copy()
    X["hsd05"] = HSD[X.index[0]:X.index[0] + len(X)]
    return X

def cv_lgb(params, two_stage=False):
    aps = []
    for (a, b) in folds:
        Xtr = add_hsd(make_X(P, LR, 0, a, tk, vol, nof))
        Xva = add_hsd(make_X(P, LR, a, b, tk, vol, nof))
        if not two_stage:
            m = lgb.LGBMClassifier(**params, verbose=-1, n_jobs=4)
            m.fit(Xtr, Y[:a])
            pr = m.predict_proba(Xva)[:, 1]
            aps.append(ap(Y[a:b], pr))
            continue
        gtr = LR[:a] >= PROM
        gva = LR[a:b] >= PROM
        mr = lgb.LGBMClassifier(**params, verbose=-1, n_jobs=4)
        ytrg = right_cond(P, 0, a)[gtr]
        mr.fit(Xtr[gtr], ytrg)
        q = mr.predict_proba(Xva)[:, 1]
        pr = np.where(gva, q, 0.0)
        aps.append(ap(Y[a:b], pr))
    return aps

def right_cond(p, start, end):
    """right side condition (label's future part) for rows [start,end) -- used only as TRAIN target"""
    out = np.zeros(end - start)
    for k, t in enumerate(range(start, end)):
        pt = p[t]; j = t + 1
        while j < len(p) and p[j] >= pt: j += 1
        seg = p[t+1:j+1]
        if len(seg):
            out[k] = int(np.log(seg.max() / pt) >= PROM)
    return out

PARAMS = dict(n_estimators=400, learning_rate=0.03, num_leaves=31, min_child_samples=8,
              subsample=0.9, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
              random_state=42)

print("\n-- single-stage LightGBM (full label) --")
a1 = cv_lgb(PARAMS)
print([round(x, 4) for x in a1], "mean:", round(float(np.mean(a1)), 4))

print("\n-- two-stage (gate x LGB right-side) --")
a2 = cv_lgb(PARAMS, two_stage=True)
print([round(x, 4) for x in a2], "mean:", round(float(np.nanmean(a2)), 4))

# uniform gate baseline on CV folds
ag = [ap(Y[a:b], (LR[a:b] >= PROM).astype(float)) for a, b in folds]
print("\n-- uniform exact gate --")
print([round(x, 4) for x in ag], "mean:", round(float(np.mean(ag)), 4))

# xgboost single stage
print("\n-- single-stage XGBoost --")
aps = []
for (a, b) in folds:
    Xtr = add_hsd(make_X(P, LR, 0, a, tk, vol, nof)); Xva = add_hsd(make_X(P, LR, a, b, tk, vol, nof))
    m = xgb.XGBClassifier(n_estimators=400, learning_rate=0.03, max_depth=4, subsample=0.9,
                          colsample_bytree=0.8, reg_lambda=1.0, n_jobs=4, random_state=42,
                          eval_metric="auc", tree_method="hist", verbosity=0)
    m.fit(Xtr, Y[:a])
    aps.append(ap(Y[a:b], m.predict_proba(Xva)[:, 1]))
print([round(x, 4) for x in aps], "mean:", round(float(np.mean(aps)), 4))
