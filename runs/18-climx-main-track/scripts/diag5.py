"""Single-cell probe: 2-SSP in-sample for 126, TXx + CSDI. Verify projection consistency."""
import sys, time
import numpy as np
sys.path.insert(0, "scripts")
import fullres_linear as fl

scens = ["ssp126", "ssp370"]
data = {s: fl.load_full(s) for s in scens}
ci = fl._land_cells(); C = ci.size
N = 86

pred = fl.emulate(scens, data["ssp126"]["lbc"], None, ridge=1e-3)

for name in ["TXx", "CSDI"]:
    g = data["ssp126"]["idx"][name].reshape(N,-1)[:, ci].astype(np.float64)
    p = pred[name].reshape(N,-1)[:, ci].astype(np.float64)
    var = g.var(0); mse = ((g-p)**2).mean(0)
    nse = 1 - mse/np.maximum(var, 1e-6)
    order = np.argsort(nse)
    print(f"\n[d5] {name} in-sample-126: worst cells", flush=True)
    for c in order[:3]:
        print(f"  cell {c}: nse={nse[c]:+.2f} varG={var[c]:.4f}", flush=True)
        print("    G[0::10]:", np.array2string(g[0::10, c], precision=2, max_line_width=200), flush=True)
        print("    P[0::10]:", np.array2string(p[0::10, c], precision=2, max_line_width=200), flush=True)
    print(f"  best cells", flush=True)
    for c in order[-3:]:
        print(f"  cell {c}: nse={nse[c]:+.2f} varG={var[c]:.4f}", flush=True)

# direct check: replicate per-cell OLS for one bad cell, TXx
name = "TXx"
P = fl.IDX_PREDICTORS[name]
lbc_train = np.stack([data[s]["lbc"] for s in scens], axis=1)
anchor = lbc_train.mean(axis=1)
Lc = lbc_train.reshape(-1,4) - anchor.repeat(2, 0)
Vb, kb = fl._lbc_basis(Lc)
Z = Lc @ Vb
zs = Z / (np.sqrt((Z**2).sum(0)/Z.shape[0]) + 1e-30)
Xd = np.hstack([np.ones((Z.shape[0],1)), zs])
n = Z.shape[0]
Y = np.stack([data[s]["idx"][name].reshape(N,-1)[:, ci] for s in scens], axis=1).reshape(n, C)
g126 = Y[:172:2, :]  # 126 rows? verify ordering
print("\n[d5] row-order check: Y row0 should equal idx126 year0 at some cell; Y[0,ci[0]]", Y[0, 0], " idx126 t0 cell0", data['ssp126']['idx'][name][0].reshape(-1)[ci[0]], flush=True)
