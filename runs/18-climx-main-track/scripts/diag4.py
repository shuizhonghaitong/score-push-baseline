"""CSDI/WSDI explosion probe + region-weighted per-index in-sample."""
import sys, time
import numpy as np
sys.path.insert(0, "scripts")
import fullres_linear as fl

scens = ["ssp126", "ssp370"]
data = {s: fl.load_full(s) for s in scens}
ci = fl._land_cells(); C = ci.size
N = 86

for ho in scens:
    pred = fl.emulate(scens, data[ho]["lbc"], None, ridge=1e-3)
    S, per = fl.fullres_nnse(data[ho]["idx"], pred)
    print(f"\n[d4] in-sample ho={ho} S={S:+.4f} (region-weighted per index):", flush=True)
    for k in fl.IDX_NAMES:
        print(f"    {k:8s} {per[k]:+.4f}", flush=True)

    if ho == "ssp370":
        for k in ["CSDI", "WSDI", "GSL", "R95p", "R10mm"]:
            g = data[ho]["idx"][k].reshape(N,-1)[:, ci].astype(np.float64)
            p = pred[k].reshape(N,-1)[:, ci].astype(np.float64)
            var = g.var(0); mse = ((g-p)**2).mean(0)
            nse = 1 - mse/np.maximum(var, 1e-6)
            valid = var > 1e-6
            q = np.quantile(nse[valid], [0, 0.01, 0.05, 0.5, 0.95, 1.0])
            worst = np.argsort(nse)[:8]
            print(f"[d4] {k}: NSE quantiles min={q[0]:.1f} p1={q[1]:.2f} p5={q[2]:.2f} med={q[3]:.3f} p95={q[4]:.3f} max={q[5]:.3f}  #valid={valid.sum()}", flush=True)
            for c in worst:
                print(f"    cell {c}: var={var[c]:.4f} stdG={np.std(g[:,c]):.3f} stdP={np.std(p[:,c]):.3f} "
                      f"ampP/G={np.std(p[:,c])/(np.std(g[:,c])+1e-9):.3f} corr={np.corrcoef(p[:,c], g[:,c])[0,1]:+.3f} "
                      f"G2015={g[0,c]:.2f} G2100={g[-1,c]:.2f} P2015={p[0,c]:.2f} P2100={p[-1,c]:.2f}", flush=True)
