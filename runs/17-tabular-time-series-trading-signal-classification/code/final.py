"""Final causal submission: rank blend of LGB(leaves=5, 3 seeds) + XGB(d3, 2 seeds),
trained on all train, predicting test. Features strictly causal (data <= t).
Gate structure: label = left_rise>=0.00995 AND right_rise>=0.00995 (verified 6131/6132 on train).
"""
import pandas as pd, numpy as np, json
import lightgbm as lgb, xgboost as xgb
import scipy.stats as st

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
near = (np.log(S.rolling(48, min_periods=1).max() / S) <= 0.003)
HSN = hours_since_true(near)

tkv = allf["taker_buy_base_asset_volume"].values; v = allf["volume"].values
nof = allf["number_of_trades"].values
tv = pd.Series(tkv); vv = pd.Series(v); nv = pd.Series(nof)

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
    X["hsd05"] = HSD05[a:b]; X["hsd1"] = HSD1[a:b]; X["hsd_near48max"] = HSN[a:b]
    X["taker1"] = (tv / vv).values[a:b]
    X["taker24"] = (tv.rolling(24).sum() / vv.rolling(24).sum()).values[a:b]
    X["lv24"] = (np.log1p(vv) - np.log1p(vv).rolling(168).mean()).values[a:b]
    X["ltn24"] = (np.log1p(nv) - np.log1p(nv).rolling(168).mean()).values[a:b]
    X["hour"] = DT[a:b].hour; X["dow"] = DT[a:b].dayofweek
    X["h_sin"] = np.sin(2*np.pi*DT[a:b].hour/24); X["h_cos"] = np.cos(2*np.pi*DT[a:b].hour/24)
    return X

Xtr = make_X(0, N).fillna(-9.0)
Xte = make_X(N, N + NT).fillna(-9.0)

prs = []
for s in (42, 7, 123):
    m = lgb.LGBMClassifier(n_estimators=1500, learning_rate=0.01, num_leaves=5, min_child_samples=60,
                           subsample=0.9, subsample_freq=1, colsample_bytree=0.6, reg_lambda=5.0,
                           random_state=s, verbose=-1, n_jobs=4)
    m.fit(Xtr, Y)
    prs.append(m.predict_proba(Xte)[:, 1])
for s in (42, 7):
    m = xgb.XGBClassifier(n_estimators=800, learning_rate=0.02, max_depth=3, subsample=0.9,
                          colsample_bytree=0.7, reg_lambda=2.0, n_jobs=4, random_state=s,
                          eval_metric="auc", tree_method="hist", verbosity=0)
    m.fit(Xtr, Y)
    prs.append(m.predict_proba(Xte)[:, 1])
l5 = np.mean(prs[:3], axis=0); x3 = np.mean(prs[3:], axis=0)
rank_blend = (st.rankdata(l5) + st.rankdata(x3)) / 2.0

sub = pd.DataFrame({"Id": tf["Id"].values, "Signal": rank_blend / rank_blend.max()})
out = f"{RUN}/outputs/submission_causal_v1.csv"
sub.to_csv(out, index=False)
print("wrote", out)
print(sub.head(3).to_string(), "...", sub.tail(2).to_string())
print("pred mean:", sub["Signal"].mean())
gate_test = LR[N:] >= PROM
print("test gate rows:", int(gate_test.sum()), "expected positives ~", int(gate_test.sum()*0.226))
print("gate rows share of top-500 scores:", int(((st.rankdata(rank_blend)[::-1] <= 500) & gate_test).sum()))
json.dump({"cv_ap_10fold": 0.3007, "file": out}, open(f"{RUN}/notes/final_v1.json", "w"), indent=1)
