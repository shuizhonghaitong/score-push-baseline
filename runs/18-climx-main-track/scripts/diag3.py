"""2-SSP deep diagnostics: field-fit in-sample R2, composite corr/NSE, test sanity."""
import sys, time
import numpy as np
sys.path.insert(0, "scripts")
import fullres_linear as fl

t0 = time.time()
scens = ["ssp126", "ssp370"]
data = {s: fl.load_full(s) for s in scens}
ci = fl._land_cells(); C = ci.size
N = 86

# --- replicate field fit for 2-SSP, per-scenario in-sample R2 per field ---
lbc_train = np.stack([data[s]["lbc"] for s in scens], axis=1)
anchor = lbc_train.mean(axis=1)
Lc = lbc_train.reshape(-1, 4) - anchor.repeat(2, axis=0)
Vb, kb = fl._lbc_basis(Lc)
ev = np.linalg.svd(Lc, compute_uv=False)**2
print(f"[d3] k={kb} eig share cum: {ev.cumsum()/ev.sum()}", flush=True)
Z = Lc @ Vb
zscale = np.sqrt((Z**2).sum(0)/Z.shape[0]) + 1e-30
Zs = Z / zscale
Xd = np.hstack([np.ones((Z.shape[0],1)), Zs])
print(f"[d3] Xd cond={np.linalg.cond(Xd):.1f}", flush=True)

for f in fl.FIELDS6:
    A = np.stack([data[s]["mean"][f].reshape(N,-1)[:, ci] for s in scens], axis=1).reshape(-1, C)
    mA = A.mean(0, keepdims=True)
    An = A - mA
    th = fl._pinv_solve(Xd, An, 1e-3)
    fit = Xd @ th.T
    r2 = 1 - ((An - fit)**2).sum(0) / (An**2).sum(0)
    r2_s = [float(np.mean(r2[:N])), float(np.mean(r2[N:]))]
    print(f"[d3] field {f:8s} in-sample R2: 126 {r2_s[0]:+.3f}  370 {r2_s[1]:+.3f}", flush=True)

# --- composite in-sample via emulate ---
for ho in scens:
    pred = fl.emulate(scens, data[ho]["lbc"], None, ridge=1e-3)
    S, per = fl.fullres_nnse(data[ho]["idx"], pred)
    print(f"\n[d3] composite in-sample ho={ho} S={S:+.4f} (region-weighted)", flush=True)
    for k in fl.IDX_NAMES:
        print(f"    {k:8s} rw {per[k]:+.4f}", flush=True)
    for k in fl.IDX_NAMES:
        g = data[ho]["idx"][k].reshape(N,-1)[:, ci].astype(np.float64)
        p = pred[k].reshape(N,-1)[:, ci].astype(np.float64)
        cg, cp = g - g.mean(0), p - p.mean(0)
        c = (cg*cp).sum(0)/(np.linalg.norm(cg,axis=0)*np.linalg.norm(cp,axis=0)+1e-12)
        var = g.var(0); mse = ((g-p)**2).mean(0)
        nnse = 1 - mse/np.maximum(var,1e-6)
        # best 2-param per-cell: a*p + b
        pg = p - p.mean(0)
        a = (pg*cg).sum(0)/((pg**2).sum(0)+1e-12)
        mse2 = ((cg - a*pg)**2).mean(0)
        bl = 1 - mse2/np.maximum(var,1e-6)
        print(f"    {k:8s} NSE {float(np.mean(nnse)):+.4f}  corr {float(np.mean(c)):+.3f}  bestLin {float(np.mean(bl)):+.4f}  ampR {float(np.mean(np.std(p,0)/(np.std(g,0)+1e-12))):.2f}", flush=True)

# --- test sanity: 2015->2100 tasmax warming ---
lbc_t = fl.test_lbc()
pred = fl.emulate(scens, lbc_t, None, ridge=1e-3)
print(f"\n[d3] test-prediction index amplitudes (2015->2100 delta, mean over land):")
for k in fl.IDX_NAMES:
    p = pred[k].reshape(N,-1)[:, ci]
    print(f"    {k:8s} {p[-1].mean()-p[0].mean():+.3f}", flush=True)
print(f"[d3] done {time.time()-t0:.0f}s", flush=True)
