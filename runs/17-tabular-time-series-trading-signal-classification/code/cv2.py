"""CV sweep v2: richer causal features + tuning. All features causal (<= t)."""
import pandas as pd, numpy as np, itertools
from sklearn.metrics import average_precision_score as ap
import lightgbm as lgb, xgboost as xgb

RUN = "/mnt/disk3/liuyutong_home/workspace/skill/score-push/runs/17-tabular-time-series-trading-signal-classification"
PROM = 0.00995

tr = pd.read_csv(f"{RUN}/data/archive_btc_train.csv", parse_dates=["datetime"])
ta = pd.read_csv(f"{RUN}/data/train_additional_features.csv", parse_dates=["datetime"])
te = pd.read_csv(f"{RUN}/data/archive_btc_test.csv", parse_dates=["datetime"])
tea = pd.read_csv(f"{RUN}/data/test_additional_features.csv", parse_dates=["datetime"])
trf = tr.merge(ta, on="Id", suffixes=("", "_a")).drop(columns=["datetime_a"])
tf = te.merge(tea, on="Id", suffixes=("", "_a")).drop(columns=["datetime_a"])
allf = pd.concat([trf, tf], ignore_index=True).sort_values("datetime").reset_index(drop=True)
P = allf["prices"].values
Y = trf["Signal"].values
N, NT = len(trf), len(tf)
DT = pd.DatetimeIndex(allf["datetime"])

def left_rise_series(p):
    out = np.full(len(p), -1e9)
    for t in range(len(p)):
        pt = p[t]; i = t
        while i > 0 and p[i-1] >= pt: i -= 1
        out[t] = np.log(p[i:t+1].max() / pt)
    return out
LR = left_rise_series(P)

S = pd.Series(P)
R = np.log(S / S.shift(1))

def hours_since_true(series_bool):
    out = np.full(len(series_bool), np.nan); last = -1
    for i, b in enumerate(series_bool):
        if b: last = i
        out[i] = (i - last) if last >= 0 else np.nan
    return out
HSD05 = hours_since_true(R <= -0.005)
HSD1 = hours_since_true(R <= -0.01)

def make_X(a, b):
    X = pd.DataFrame(index=range(a, b))
    X["lr"] = LR[a:b]; X["lr_margin"] = LR[a:b] - PROM
    for lag in (1, 2, 3, 4, 6, 8, 12, 24):
        X[f"r{lag}"] = R.shift(lag - 1).values[a:b]
    X["r1_abs"] = np.abs(X["r1"]); X["r1_sq"] = X["r1"] ** 2
    for w in (6, 12, 24, 48, 168):
        X[f"vol{w}"] = R.rolling(w).std().values[a:b]
        X[f"ret{w}"] = (S / S.shift(w) - 1).values[a:b]
        X[f"dd{w}"] = (np.log(S / S.rolling(w, min_periods=1).max())).values[a:b]
        X[f"minr{w}"] = R.rolling(w).min().values[a:b]
        X[f"cumr{w}"] = R.rolling(w).sum().values[a:b]
    for w in (6, 24):
        X[f"below{w}"] = (S.shift(1) < S).rolling(w).sum().values[a:b]
    X["hsd05"] = HSD05[a:b]; X["hsd1"] = HSD1[a:b]
    # hours since price was within 0.3% of the 48h max (freshness of the high)
    near = (np.log(S.rolling(48, min_periods=1).max() / S) <= 0.003)
    X["hsd_near48max"] = hours_since_true(near)[a:b]
    tkv = allf["taker_buy_base_asset_volume"].values; v = allf["volume"].values
    nof = allf["number_of_trades"].values
    tv = pd.Series(tkv); vv = pd.Series(v); nv = pd.Series(nof)
    X["taker1"] = (tv / vv).values[a:b]
    X["taker24"] = (tv.rolling(24).sum() / vv.rolling(24).sum()).values[a:b]
    X["lv24"] = (np.log1p(vv) - np.log1p(vv).rolling(168).mean()).values[a:b]
    X["ltn24"] = (np.log1p(nv) - np.log1p(nv).rolling(168).mean()).values[a:b]
    X["hour"] = DT[a:b].hour; X["dow"] = DT[a:b].dayofweek
    X["h_sin"] = np.sin(2*np.pi*DT[a:b].hour/24); X["h_cos"] = np.cos(2*np.pi*DT[a:b].hour/24)
    return X

folds6 = []
start0 = int(N * 0.50)
seglen = (N - start0) // 6
for i in range(6):
    a = start0 + i * seglen
    b = N if i == 5 else start0 + (i + 1) * seglen
    folds6.append((a, b))

def run(name, model, Xtr, Xva, ytr, yva):
    model.fit(Xtr, ytr)
    pr = model.predict_proba(Xva)[:, 1]
    return pr

def eval_cfg(name, factory):
    aps = []
    for (a, b) in folds6:
        Xtr = make_X(0, a); Xva = make_X(a, b)
        pr = run(name, factory(), Xtr, Xva, Y[:a], Y[a:b])
        aps.append(ap(Y[a:b], pr))
    print(f"{name:38s} {[round(x,4) for x in aps]}  mean={np.mean(aps):.4f}")
    return aps

print("features:", make_X(0, 10).shape[1])
eval_cfg("lgb_leaves31", lambda: lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=31, min_child_samples=8, subsample=0.9, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0, random_state=42, verbose=-1, n_jobs=4))
eval_cfg("lgb_leaves15_c20", lambda: lgb.LGBMClassifier(n_estimators=800, learning_rate=0.02, num_leaves=15, min_child_samples=20, subsample=0.9, subsample_freq=1, colsample_bytree=0.7, reg_lambda=2.0, random_state=42, verbose=-1, n_jobs=4))
eval_cfg("lgb_leaves7_c40", lambda: lgb.LGBMClassifier(n_estimators=1200, learning_rate=0.02, num_leaves=7, min_child_samples=40, subsample=0.9, subsample_freq=1, colsample_bytree=0.7, reg_lambda=3.0, random_state=42, verbose=-1, n_jobs=4))
eval_cfg("xgb_d4", lambda: xgb.XGBClassifier(n_estimators=400, learning_rate=0.03, max_depth=4, subsample=0.9, colsample_bytree=0.8, reg_lambda=1.0, n_jobs=4, random_state=42, eval_metric="auc", tree_method="hist", verbosity=0))
eval_cfg("xgb_d3", lambda: xgb.XGBClassifier(n_estimators=800, learning_rate=0.02, max_depth=3, subsample=0.9, colsample_bytree=0.7, reg_lambda=2.0, n_jobs=4, random_state=42, eval_metric="auc", tree_method="hist", verbosity=0))
