"""Diagnostics: 2-SSP in-sample NSE + 1-SSP fold shape-vs-amplitude decomposition."""
import sys, time
import numpy as np
sys.path.insert(0, "scripts")
import fullres_linear as fl

t0 = time.time()
scens = ["ssp126", "ssp370"]
data = {s: fl.load_full(s) for s in scens}
ci = fl._land_cells()
C = ci.size

def nnse_series(gt, pred):
    """Per-cell NSE over the time axis (same formula as scorer, no regions)."""
    g = gt.astype(np.float64); p = pred.astype(np.float64)
    var = g.var(axis=0).reshape(-1)[ci]
    mse = ((g - p) ** 2).mean(axis=0).reshape(-1)[ci]
    valid = var > 1e-6
    r2 = np.where(valid, 1 - mse / np.maximum(var, 1e-6), np.nan)
    return r2, valid

def report(tag, per_idx_nnse, per_idx_corr, per_idx_opt):
    print(f"== {tag} ==", flush=True)
    for k in fl.IDX_NAMES:
        print(f"  {k:8s} NSE {per_idx_nnse[k]:+.4f}  corr {per_idx_corr[k]:+.3f}  bestLinNSE {per_idx_opt[k]:+.4f}", flush=True)

# ---- 2-SSP in-sample (train on both, evaluate each) ----
res = {}
for ho in scens:
    pred = fl.emulate(scens, data[ho]["lbc"], None, ridge=1e-3)
    S, _ = fl.fullres_nnse(data[ho]["idx"], pred)
    res[ho] = (pred, S)
print(f"[diag] 2-SSP in-sample: " + "  ".join(f"{k}: {v[1]:+.4f}" for k, v in res.items()), flush=True)

# ---- 1-SSP folds: decomposition ----
for ho in scens:
    tr = [s for s in scens if s != ho]
    pred = fl.emulate(tr, data[ho]["lbc"], None, ridge=1e-3)
    nnse, corr, opt = {}, {}, {}
    for k in fl.IDX_NAMES:
        g = data[ho]["idx"][k]
        p = pred[k]
        r2, valid = nnse_series(g, p)
        nnse[k] = float(np.nanmean(r2))
        gc = g[None, ci].T.reshape(NY := g.shape[0], C) if False else g.reshape(NY2 := 86, -1)[:, ci].astype(np.float64)
        pc = p.reshape(86, -1)[:, ci].astype(np.float64)
        cg = gc - gc.mean(axis=0); cp = pc - pc.mean(axis=0)
        denom = (np.linalg.norm(cg, axis=0) * np.linalg.norm(cp, axis=0) + 1e-12)
        c = (cg * cp).sum(axis=0) / denom
        corr[k] = float(np.nanmean(c[valid]))
        # best 2-param linear map per cell: a*pred + b -> gt
        pg = pc - pc.mean(axis=0)
        vpp = (pg ** 2).sum(axis=0) + 1e-12
        a = (pg * cg).sum(axis=0) / vpp
        mse = ((cg - a * pg) ** 2).mean(axis=0)
        var = gc.var(axis=0)
        opt[k] = float(np.nanmean(np.where(valid, 1 - mse / np.maximum(var, 1e-6), np.nan)))
    Smean = float(np.mean(list(nnse.values())))
    report(f"1-SSP fold ho={ho} (train {tr})", nnse, corr, opt)
    print(f"  MEAN NSE {Smean:+.4f}", flush=True)

print(f"[diag] done {time.time()-t0:.0f}s", flush=True)
