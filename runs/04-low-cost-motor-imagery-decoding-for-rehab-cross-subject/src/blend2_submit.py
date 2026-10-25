"""2-seed DL blend: P = w_ml*ML + (1-w_ml)*mean(DL_col_seedA, DL_col_seedB); submit.
usage: blend2_submit.py <ml_probs.csv> <dlA_probs.csv> <dlB_probs.csv> <dl_col> <out_csv> "<desc>"
"""
import os, subprocess, sys
import numpy as np
import pandas as pd

ml_p_path, dlA_path, dlB_path, dl_col, out_path, desc = sys.argv[1:7]
w_ml = float(os.environ.get("WML", "0.45"))
ml = pd.read_csv(ml_p_path)
dlA = pd.read_csv(dlA_path); dlB = pd.read_csv(dlB_path)
assert len(ml) == len(dlA) == len(dlB) == 360
Pdl = 0.5 * dlA[dl_col].to_numpy() + 0.5 * dlB[dl_col].to_numpy()
Pfin = w_ml * ml["P"].to_numpy() + (1 - w_ml) * Pdl
predb = np.zeros(len(Pfin), int)
tsub = ml["file"].to_numpy()
for f in sorted(set(tsub)):
    m = np.flatnonzero(tsub == f)
    k2 = len(m) // 2
    predb[m[np.argpartition(-Pfin[m], k2)[:k2]]] = 1
out = pd.DataFrame({"ID": range(len(Pfin)), "TARGET": ["move" if p else "rest" for p in predb]})
out.to_csv(out_path, index=False)
print(out.TARGET.value_counts().to_dict(), "w_ml=", w_ml, "dl_col=", dl_col)
r = subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), "kaggle_submit.py"), out_path, desc])
sys.exit(r.returncode)
