"""One-shot CV: load npz once, run 3 folds, write results file."""
import sys, time
from pathlib import Path
import numpy as np

t0 = time.time()
sys.path.insert(0, "scripts")
import fullres_linear as fl
print(f"[cv] import {time.time()-t0:.1f}s", flush=True)

scens = ["ssp126", "ssp370", "ssp585"]
missing = [s for s in scens if not (fl.CACHE / f"full_{s}.npz").exists()]
if missing:
    print(f"[cv] missing npz: {missing} — aborting", flush=True)
    sys.exit(2)
data = {s: fl.load_full(s) for s in scens}
print(f"[cv] loaded {len(data)} scenarios in {time.time()-t0:.1f}s", flush=True)

allper = {}
for ho in scens:
    tr = [s for s in scens if s != ho]
    t1 = time.time()
    pred = fl.emulate(tr, data[ho]["lbc"], None, ridge=1e-3)
    S, per = fl.fullres_nnse(data[ho]["idx"], pred)
    allper[ho] = per
    print(f"[cv] fold holdout={ho} S={S:+.5f} ({time.time()-t1:.0f}s)", flush=True)
    for k in fl.IDX_NAMES:
        print(f"    {k:8s} {per[k]:+.4f}", flush=True)

w126, w370, w585 = 0.4, 0.4, 0.2
est = {k: w126*allper["ssp126"][k] + w370*allper["ssp370"][k] + w585*allper["ssp585"][k]
       for k in fl.IDX_NAMES}
avg = {k: float(np.mean([allper[ho][k] for ho in scens])) for k in fl.IDX_NAMES}
print("\n== summary: 3-fold mean / weighted(0.4,0.4,0.2) ==", flush=True)
for k in fl.IDX_NAMES:
    print(f"    {k:8s} {avg[k]:+.4f}  {est[k]:+.4f}", flush=True)
print(f"    MEAN    {float(np.mean(list(avg.values()))):+.4f}  {float(np.mean(list(est.values()))):+.4f}", flush=True)
out = Path("models/cache/cv_fullres_result.txt")
with open(out, "w") as f:
    f.write("3-fold mean / weighted(0.4,0.4,0.2)\n")
    for k in fl.IDX_NAMES:
        f.write(f"{k} {avg[k]:+.4f} {est[k]:+.4f}\n")
    f.write(f"MEAN {float(np.mean(list(avg.values()))):+.4f} {float(np.mean(list(est.values()))):+.4f}\n")
print(f"[cv] wrote {out} total {time.time()-t0:.0f}s", flush=True)
