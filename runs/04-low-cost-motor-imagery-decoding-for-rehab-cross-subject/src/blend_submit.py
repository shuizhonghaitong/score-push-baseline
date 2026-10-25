"""Build a 0.45/0.55-style ML+DL blend from local prob files and submit it.
usage: blend_submit.py <ml_probs.csv> <dl_probs.csv> <dl_col> <out_csv> "<desc>"
  weights: uses OOF-selected 0.45*ML + 0.55*DL (see experiments.csv row 25/26);
  overrides via env WML (e.g. WML=0.5)
"""
import os, subprocess, sys
import numpy as np
import pandas as pd

ml_p_path, dl_p_path, dl_col, out_path, desc = sys.argv[1:6]
w_ml = float(os.environ.get("WML", "0.45"))
ml = pd.read_csv(ml_p_path)
dl = pd.read_csv(dl_p_path)
assert len(ml) == len(dl) == 360
Pml = ml["P"].to_numpy(); Pdl = dl[dl_col].to_numpy()
Pfin = w_ml * Pml + (1 - w_ml) * Pdl
predb = np.zeros(len(Pfin), int)
tsub = ml["file"].to_numpy() if "file" in ml.columns else dl["file"].to_numpy()
for f in sorted(set(tsub)):
    m = np.flatnonzero(tsub == f)
    k2 = len(m) // 2
    predb[m[np.argpartition(-Pfin[m], k2)[:k2]]] = 1
out = pd.DataFrame({"ID": range(len(Pfin)), "TARGET": ["move" if p else "rest" for p in predb]})
out.to_csv(out_path, index=False)
print(out.TARGET.value_counts().to_dict(), "w_ml=", w_ml, "dl_col=", dl_col)
run_guard = os.path.join(os.path.dirname(__file__), "kaggle_submit.py")
r = subprocess.run([sys.executable, run_guard, out_path, desc])
sys.exit(r.returncode)
