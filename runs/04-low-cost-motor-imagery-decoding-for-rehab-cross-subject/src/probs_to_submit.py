"""Build per-file balanced submission from a probs CSV (cols: file,P_fin) and submit.
usage: probs_to_submit.py <probs_csv> <out_csv> "<desc>"
"""
import subprocess, sys
import numpy as np
import pandas as pd

probs = pd.read_csv(sys.argv[1])
out_path, desc = sys.argv[2], sys.argv[3]
P = probs["P_fin"].to_numpy()
tsub = probs["file"].to_numpy()
predb = np.zeros(len(P), int)
for f in sorted(set(tsub)):
    m = np.flatnonzero(tsub == f)
    k2 = len(m) // 2
    predb[m[np.argpartition(-P[m], k2)[:k2]]] = 1
out = pd.DataFrame({"ID": range(len(P)), "TARGET": ["move" if p else "rest" for p in predb]})
out.to_csv(out_path, index=False)
print(out.TARGET.value_counts().to_dict())
r = subprocess.run([sys.executable, "src/kaggle_submit.py", out_path, desc])
sys.exit(r.returncode)
