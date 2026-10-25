"""Local check of a submission's realized PnL against (fake or real) labels.
Usage: python code/verify_score.py submission.csv labels_dir
labels_dir: dir with train.csv/test.csv (mint,mt,max_x,final_x)
"""
import sys
import numpy as np
import pandas as pd

sub = pd.read_csv(sys.argv[1])
lab = {}
for f in ("train.csv", "test.csv"):
    try:
        d = pd.read_csv(f"{sys.argv[2]}/{f}")
    except FileNotFoundError:
        continue
    if "max_x" not in d.columns:
        continue
    for m, mx, fx in zip(d["mint"], d["max_x"], d["final_x"]):
        lab[m] = (mx, fx)
COST = 0.3
mints = sub["mint"].values
tp = sub["tp"].astype(float).values
mx = np.array([lab[m][0] if m in lab else np.nan for m in mints])
fx = np.array([lab[m][1] if m in lab else np.nan for m in mints])
have = np.isin(mints, list(lab))
tp = np.where(have, tp, 0.0)
hit = (mx >= tp) & (tp > 0)
pnl = np.where(tp <= 0, 0.0, np.where(hit, tp - 1.0 - COST,
          np.maximum(np.where(have, fx, 0.0) - 1.0 - COST, -1.0)))
n_tr = int((tp > 0).sum())
print(f"tokens: {len(mints)}, with labels: {int(have.sum())}, trades: {n_tr}")
print(f"realized PnL: {pnl.sum():+.3f}  (hits: {int(hit.sum())}, hit pnl: {pnl[hit].sum():+.2f}, "
      f"miss pnl: {pnl[(tp > 0) & ~hit].sum():+.2f})")
tpv = sub.loc[sub.tp > 0, "tp"]
print("tp distribution:", tpv.round(1).value_counts().sort_index().to_dict())
