import pandas as pd, numpy as np
import lightgbm as lgb, xgboost as xgb
import scipy.stats as st
exec(open("/mnt/disk3/liuyutong_home/workspace/skill/score-push/runs/17-tabular-time-series-trading-signal-classification/code/cv2.py").read().split('print("features"')[0])

Xtr = make_X(0, N).fillna(-9.0)
Xte = make_X(N, N + NT).fillna(-9.0)
prs = []
for s in (42, 7, 123):
    m = lgb.LGBMClassifier(n_estimators=1500, learning_rate=0.01, num_leaves=5, min_child_samples=60, subsample=0.9, subsample_freq=1, colsample_bytree=0.6, reg_lambda=5.0, random_state=s, verbose=-1, n_jobs=4)
    m.fit(Xtr, Y); prs.append(m.predict_proba(Xte)[:, 1])
for s in (42, 7):
    m = xgb.XGBClassifier(n_estimators=800, learning_rate=0.02, max_depth=3, subsample=0.9, colsample_bytree=0.7, reg_lambda=2.0, n_jobs=4, random_state=s, eval_metric="auc", tree_method="hist", verbosity=0)
    m.fit(Xtr, Y); prs.append(m.predict_proba(Xte)[:, 1])
sc = (st.rankdata(np.mean(prs[:3], axis=0)) + st.rankdata(np.mean(prs[3:], axis=0))) / 2

lt = LR[N:]
gate = lt >= PROM
te_df = tf.copy(); te_df["lr"] = lt; te_df["score"] = sc
print("test lr percentiles:", np.percentile(lt, [10, 25, 50, 75, 90, 99]).round(4))
print("train lr percentiles:", np.percentile(LR[:N], [10, 25, 50, 75, 90, 99]).round(4))
print("gate:", int(gate.sum()), "near-miss [0.008,0.00995):", int(((lt >= 0.008) & (lt < PROM)).sum()))
top = te_df.sort_values("score", ascending=False)
print("\ntop-60 rows score/lr/r1:")
top = top.copy()
top["r1"] = (np.log(te_df["prices"] / te_df["prices"].shift(1))).values
print(top.head(60)[["Id", "datetime", "prices", "lr", "r1", "score"]].to_string())
print("\nnon-gate rows with score > median gate score:",
      int(((~gate) & (sc > np.median(sc[gate]))).sum()))
print("gate rows below some non-gate row:", int(((np.maximum.accumulate(sc)[np.argsort(np.argsort(-sc))] > 0) ).sum()))
# proper: for each gate row, count non-gate above it
order = np.argsort(-sc)
isg = gate[order]
running_nongate = 0
bad = 0
for i, g in enumerate(isg):
    if not g: running_nongate += 1
    elif running_nongate > 0: bad += 1
print("gate rows with a non-gate row ranked above:", bad)
