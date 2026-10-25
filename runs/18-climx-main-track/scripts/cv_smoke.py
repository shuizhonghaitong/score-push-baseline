"""Smoke CV on only ssp126 + ssp370 (before ssp585 arrives).
2 folds: train on one, predict the other. Conservative (1-SSP training)."""
import sys, time
from pathlib import Path
import numpy as np

t0 = time.time()
sys.path.insert(0, "scripts")
import fullres_linear as fl

scens = ["ssp126", "ssp370"]
data = {s: fl.load_full(s) for s in scens}
print(f"[smoke] loaded {len(data)} in {time.time()-t0:.1f}s", flush=True)

allper = {}
for ho in scens:
    tr = [s for s in scens if s != ho]
    t1 = time.time()
    pred = fl.emulate(tr, data[ho]["lbc"], None, ridge=1e-3)
    S, per = fl.fullres_nnse(data[ho]["idx"], pred)
    allper[ho] = per
    print(f"[smoke] fold holdout={ho} (train {tr}) S={S:+.5f} ({time.time()-t1:.0f}s)", flush=True)
    for k in fl.IDX_NAMES:
        print(f"    {k:8s} {per[k]:+.4f}", flush=True)

est = {k: 0.5 * allper["ssp126"][k] + 0.5 * allper["ssp370"][k] for k in fl.IDX_NAMES}
print(f"[smoke] weighted(0.5,0.5) MEAN = {float(np.mean(list(est.values()))):+.4f}", flush=True)
out = Path("models/cache/cv_smoke_126_370.txt")
out.write_text(
    "2-scen smoke (1-SSP training folds)\n"
    + "".join(f"{k} {allper['ssp126'][k]:+.4f} {allper['ssp370'][k]:+.4f} {est[k]:+.4f}\n" for k in fl.IDX_NAMES)
    + f"MEAN {float(np.mean([allper['ssp126'][k] for k in fl.IDX_NAMES])):+.4f} "
      f"{float(np.mean([allper['ssp370'][k] for k in fl.IDX_NAMES])):+.4f} "
      f"{float(np.mean(list(est.values()))):+.4f}\n"
)
print(f"[smoke] wrote {out} total {time.time()-t0:.0f}s", flush=True)
