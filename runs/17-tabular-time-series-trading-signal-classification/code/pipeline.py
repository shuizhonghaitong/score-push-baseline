"""Causal pipeline for tabular-time-series-trading-signal-classification.
All features at time t use only data at or before t (no lookahead).
"""
import sys, json, warnings
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from sklearn.metrics import average_precision_score as ap
import lightgbm as lgb
import xgboost as xgb

RUN = "/mnt/disk3/liuyutong_home/workspace/skill/score-push/runs/17-tabular-time-series-trading-signal-classification"
warnings.filterwarnings("ignore")

train = pd.read_csv(f"{RUN}/data/train_additional_features.csv")
test = pd.read_csv(f"{RUN}/data/test_additional_features.csv")
sample = pd.read_csv(f"{RUN}/data/sample_submission.csv")

def norm_dt(df):
    for c in ("datetime", "datetime_utc", "timestamp", "date"):
        if c in df.columns:
            df = df.copy()
            df[c] = pd.to_datetime(df[c])
            return df, c
    raise SystemExit("no datetime column: " + str(list(df.columns)))

train, tcol = norm_dt(train)
test, _ = norm_dt(test)
tcol_test = tcol
price_col = next(c for c in ("prices", "close", "close_price", "price") if c in train.columns)
id_col = "Id" if "Id" in sample.columns else sample.columns[0]
print("train:", train.shape, "test:", test.shape, "price col:", price_col)
print("train span:", train[tcol].iloc[0], "->", train[tcol].iloc[-1])
print("test  span:", test[tcol_test].iloc[0], "->", test[tcol_test].iloc[-1])
print("sample head:", sample.head(3).to_dict("records"))

# ---------- 1. label structure on train ----------
y = train["Signal"].values if "Signal" in train.columns else None
p_tr = train[price_col].values
pos = int(y.sum())
print(f"train pos={pos} base={pos/len(y):.4f}")
if y is not None:
    logp = np.log(p_tr)
    for prom in (0.00995, 0.01):
        peaks, _ = find_peaks(-logp, prominence=prom)
        pred = np.zeros(len(y)); pred[peaks] = 1
        tp = int(((pred == 1) & (y == 1)).sum())
        print(f"find_peaks prom={prom}: peaks={len(peaks)} tp={tp}/{pos} ap={ap(y, pred):.4f}")

# ---------- 2. causal features ----------
def build_features(df, price_col, tcol):
    p = df[price_col].astype(float)
    r = np.log(p / p.shift(1))
    F = pd.DataFrame(index=df.index)
    for lag in (1, 2, 3, 4, 6, 8, 12, 24, 48):
        F[f"r{lag}"] = r.shift(lag - 1)
    for w in (6, 12, 24, 48, 96, 168):
        F[f"vol{w}"] = r.rolling(w).std().shift(1)
        F[f"ret{w}"] = (p / p.shift(w) - 1).shift(1)
        roll_max = p.rolling(w, min_periods=1).max()
        F[f"dd{w}"] = ((roll_max - p) / roll_max).shift(1)  # drop from past max -> left-side condition
        F[f"rs{w}"] = ((p / roll_max) - 1).shift(1)
        roll_min = p.rolling(w, min_periods=1).min()
        F[f"up{w}"] = ((p / roll_min) - 1).shift(1)
    # time since price last made a new rolling high / since big drop
    s = p / p.shift(1)
    F["s1"] = s.shift(1)
    if "volume" in df.columns:
        v = df["volume"].astype(float)
        for w in (6, 24, 168):
            F[f"vln{w}"] = np.log1p(v).rolling(w).mean().shift(1)
            F[f"vratio{w}"] = (v / v.rolling(w).mean()).shift(1)
    for c in ("taker_buy_volume", "taker_buy_quote_volume", "quote_volume", "number_of_trades"):
        if c in df.columns:
            F["taker_ratio24"] = (
                df[c].astype(float).rolling(24).sum().shift(1)
                / df["volume"].astype(float).rolling(24).sum().shift(1)
            )
            break
    # calendar
    t = df[tcol]
    F["hour"] = t.dt.hour
    F["dow"] = t.dt.dayofweek
    F["dom"] = t.dt.day
    F["month"] = t.dt.month
    # provided technical features (verified causal by name below)
    extra = [c for c in df.columns if c not in (price_col, tcol, "Signal", id_col, "id", "Index")
             and pd.api.types.is_numeric_dtype(df[c])]
    excl = set()
    for c in extra:
        cl = c.lower()
        if any(k in cl for k in ("future", "fwd", "forward", "next", "target", "signal")):
            excl.add(c)
    extra = [c for c in extra if c not in excl]
    for c in extra:
        F[c + "_lag1"] = df[c].shift(1).astype(float)
    F["n_extras"] = len(extra)
    return F, extra, sorted(excl)

Ftr, extras, excluded = build_features(train, price_col, tcol)
Fte, _, _ = build_features(test, price_col, tcol_test)
print(f"feature count: {Ftr.shape[1]}, provided extras used(lag1): {len(extras)}, excluded as suspicious: {excluded}")

# ---------- 3. time-series CV ----------
n = len(train)
folds = []
# expanding-window folds over last 40% of train
start = int(n * 0.60)
seg = (n - start) // 4
for i in range(4):
    a = start + i * seg
    b = start + (i + 1) * seg if i < 3 else n
    folds.append((a, b))
print("folds:", folds)

def cv_model(kind):
    aps = []
    for (a, b) in folds:
        Xtr = Ftr.iloc[:a]; Xva = Ftr.iloc[a:b]
        ytr = y[:a]; yva = y[a:b]
        if kind == "lgb":
            m = lgb.LGBMClassifier(
                n_estimators=600, learning_rate=0.03, num_leaves=31, max_depth=-1,
                min_child_samples=10, subsample=0.9, subsample_freq=1, colsample_bytree=0.8,
                reg_lambda=1.0, n_jobs=8, random_state=42, verbose=-1)
            m.fit(Xtr, ytr)
            pr = m.predict_proba(Xva)[:, 1]
        else:
            m = xgb.XGBClassifier(
                n_estimators=600, learning_rate=0.03, max_depth=4, subsample=0.9,
                colsample_bytree=0.8, reg_lambda=1.0, n_jobs=8, random_state=42,
                eval_metric="auc", tree_method="hist", verbosity=0)
            m.fit(Xtr, ytr)
            pr = m.predict_proba(Xva)[:, 1]
        aps.append(ap(yva, pr))
    return float(np.mean(aps)), aps

res = {}
res["lgb"] = cv_model("lgb")
res["xgb"] = cv_model("xgb")
# baselines
base = np.full(n, pos / n)
fold_aps_base = [ap(y[a:b], base[a:b]) for a, b in folds]
res["base_prior"] = (float(np.mean(fold_aps_base)), fold_aps_base)
dd = (Ftr["dd48"].fillna(0).values > 0.01) & (Ftr["r1"].fillna(0).values <= 0)
fb = np.where(dd, 0.5, base[pos / n])
res["left_rule"] = (float(np.mean([ap(y[a:b], fb[a:b]) for a, b in folds])), None)
print(json.dumps({k: (v[0], [round(x, 4) for x in v[1]] if v[1] else None) for k, v in res.items()}, indent=1))

# ---------- 4. fit on all train, predict test ----------
Xtr = Ftr.fillna(-9.0)
Xte = Fte.fillna(-9.0)
m1 = lgb.LGBMClassifier(n_estimators=600, learning_rate=0.03, num_leaves=31, min_child_samples=10,
                        subsample=0.9, subsample_freq=1, colsample_bytree=0.8, reg_lambda=1.0,
                        n_jobs=8, random_state=42, verbose=-1)
m1.fit(Xtr, y)
p1 = m1.predict_proba(Xte)[:, 1]
m2 = xgb.XGBClassifier(n_estimators=600, learning_rate=0.03, max_depth=4, subsample=0.9,
                       colsample_bytree=0.8, reg_lambda=1.0, n_jobs=8, random_state=42,
                       eval_metric="auc", tree_method="hist", verbosity=0)
m2.fit(Xtr, y)
p2 = m2.predict_proba(Xte)[:, 1]
p_blend = 0.5 * p1 + 0.5 * p2

sub = pd.DataFrame({id_col: sample[id_col], "Signal": p_blend})
out = f"{RUN}/outputs/submission_blend.csv"
sub.to_csv(out, index=False)
print("wrote", out, sub.head(3).to_dict("records"))
imp = pd.Series(m1.feature_importances_, index=Xtr.columns).sort_values(ascending=False)
print("top lgb features:"); print(imp.head(20).to_string())
json.dump({"cv": {k: v[0] for k, v in res.items()}, "n_extras": len(extras)},
          open(f"{RUN}/notes/pipeline_v1.json", "w"), indent=1)
