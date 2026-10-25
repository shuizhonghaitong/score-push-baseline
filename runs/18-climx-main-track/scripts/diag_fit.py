import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "scripts")
import numpy as np
import train_v4 as t4
import train_v5 as t5
from climx_indices import percentile_doy_window

Cn = 216
model = t4.V4(hist_start=1961, sigma=2.0)
model.fit(["ssp126", "ssp585"])
import netCDF4 as nc
from climx_indices import base_period_slice
ds2 = nc.Dataset(str(t4.lite_file("hist_late")))
model.base = base_period_slice({v: np.asarray(ds2.variables[v][0], dtype=np.float64) for v in t4.VARS})
ds2.close()

n = 170
I_pool = {}
idx = 0
for s in ["ssp126", "ssp585"]:
    gt = t4.gt_indices_npz(__import__("pathlib").Path("models/cache"), s)
    for k in ["TXx", "SU", "WSDI"]:
        I_pool.setdefault(k, np.empty((n, 12, 18)))
        I_pool[k][idx:idx+85] = gt[k][:85]
    idx += 85
F_pool = np.concatenate([model.F_test[s][:85] for s in ["ssp126", "ssp585"]])
mu = {k: I_pool[k].mean(axis=0) for k in I_pool}
beta = {}
aint = {}
for k in I_pool:
    ak, bk = t5.fit_line(I_pool[k].reshape(n, Cn), F_pool)
    beta[k] = bk; aint[k] = ak

th_wsdi = percentile_doy_window(model.base["tasmax"], 0.90)
S = model.S["tasmax"]
Fbar = model.Fbar

def idx_at(sl, d, Fp, eps=0.0, w=None):
    field = S + sl * (Fp - Fbar) + d
    if eps > 0 and w is not None:
        field = field + eps * w[:, None, None]
    r = t5.var_indices_year("tasmax", field, {"tasmax": th_wsdi})
    return r["TXx"], r["SU"], r["WSDI"]

for cell in [(7,10), (4,3), (9,15)]:
    i, j = cell
    c2 = i*18+j
    print(f"\ncell {cell}:")
    print(f"  target lines at F=0/0.3/0.5:")
    for Fp in [0.0, 0.3, 0.5]:
        print(f"    F={Fp}: TXx={aint['TXx'][c2]+beta['TXx'][c2]*Fp:7.2f}  WSDI={aint['WSDI'][c2]+beta['WSDI'][c2]*Fp:7.1f}  SU={aint['SU'][c2]+beta['SU'][c2]*Fp:5.1f}")
    # brute-force best (sl, d) per F point (independent) to see if a solution even exists
    print("  per-F best possible (sl,d) match (brute 2D, independent per F):")
    for Fp in [0.0, 0.3, 0.5]:
        best = None
        for sl in np.arange(0, 10, 0.25):
            for d in np.arange(-4, 5, 0.25):
                txx, su, wsd = idx_at(sl, d, Fp)
                e = (txx[i,j] - (aint['TXx'][c2]+beta['TXx'][c2]*Fp))**2/4.0 \
                    + (su[i,j] - (aint['SU'][c2]+beta['SU'][c2]*Fp))**2/625.0 \
                    + (wsd[i,j] - (aint['WSDI'][c2]+beta['WSDI'][c2]*Fp))**2/1600.0
                if best is None or e < best[0]:
                    best = (e, sl, d, txx[i,j], su[i,j], wsd[i,j])
        print(f"    F={Fp}: best e={best[0]:.3f} sl={best[1]:.2f} d={best[2]:.2f} -> TXx={best[3]:.2f} WSDI={best[5]:.1f} SU={best[4]:.0f}")
