"""Per-index optimal delta scaling: reg GT_delta on pred_delta (in-sample, 2-SSP).
slope>1: model under-predicts amplitude; slope<1: over-predicts."""
import sys, time
import numpy as np
sys.path.insert(0, "scripts")
import fullres_linear as fl

t0 = time.time()
scens = ["ssp126", "ssp370"]
data = {s: fl.load_full(s) for s in scens}
ci = fl._land_cells()
for ho in scens:
    pred = fl.emulate(scens, data[ho]["lbc"], None)
    print(f"== in-sample {ho}", flush=True)
    for k in fl.IDX_NAMES:
        g = data[ho]["idx"][k].reshape(86, -1)[:, ci].astype(np.float64)
        p = pred[k].reshape(86, -1)[:, ci].astype(np.float64)
        gd = g[-1] - g[0]
        pd = p[-1] - p[0]
        var = g.var(0)
        m = var > 0.25  # cells with meaningful variability
        a = (pd[m] - pd[m].mean())
        b = (gd[m] - gd[m].mean())
        slope = float((a * b).sum() / max((a * a).sum(), 1e-12))
        corr = float((a * b).sum() / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
        print(f"  {k:8s} slope {slope:+.3f} corr {corr:+.3f}  GTamp {np.abs(gd).mean():.3f} PredAmp {np.abs(pd).mean():.3f}", flush=True)
print(f"done {time.time()-t0:.0f}s", flush=True)
