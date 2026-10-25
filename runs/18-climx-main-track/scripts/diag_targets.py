"""V6 design test: per-cell target curves via local median in F (pool of 2 train SSPs),
evaluated as index predictions on the holdout SSP. Direct R^2 of T(y) vs GT."""
import sys, numpy as np
sys.path.insert(0, 'scripts')
import train_v4 as t4
from climx_common import SUB_COLUMNS, lite_file
from scipy.ndimage import gaussian_filter
from pathlib import Path
import netCDF4 as nc

gt = {s: t4.gt_indices_npz(Path("models/cache"), s) for s in t4.SSPS}
co2_ref = float(np.asarray(nc.Dataset(str(lite_file("ssp126"))).variables["CO2_LBC"][0, 0]))
F = {}
for s in t4.SSPS:
    c = np.asarray(nc.Dataset(str(lite_file(s))).variables["CO2_LBC"][0], dtype=np.float64)
    F[s] = np.log(c / co2_ref)[:85]

for ho in t4.SSPS:
    tr = [s for s in t4.SSPS if s != ho]
    Ftr = np.concatenate([F[s] for s in tr])                       # (170,)
    Fh = F[ho]
    print(f"== holdout {ho} (F range {Fh.min():.3f}..{Fh.max():.3f}) ==")
    print(f"{'var':8s} | " + " | ".join(f"h{h}/sm{sm}" for h in (0.08, 0.15, 0.25) for sm in (0, 2)))
    for k in SUB_COLUMNS:
        line = [f"{k:8s}"]
        Itr3 = np.stack([gt[s][k][:85].reshape(85, -1) for s in tr], axis=1)  # (85, 2, 216)
        Itr = Itr3.transpose(1, 0, 2).reshape(-1, 216)  # (170, 216) aligned to Ftr
        for h in (0.08, 0.15, 0.25):
            for sm in (0, 2):
                T = np.zeros((85, 216))
                for t in range(85):
                    w = (np.abs(Ftr - Fh[t]) < h)
                    T[t] = np.nanmedian(np.where(w[:, None], Itr, np.nan), axis=0)
                if sm > 0:
                    T = gaussian_filter(T, (sm, 0.01), mode="nearest")
                Iho = gt[ho][k][:85].reshape(85, -1)
                var = Iho.var(axis=0)
                valid = var > 1e-6
                r2 = 1 - ((Iho - T)**2).mean(axis=0) / var
                r2v = r2[valid]
                line.append(f"med{np.nanmedian(r2v):+.3f} me{r2v.mean():+.3f} n{valid.sum()}")
        print(" | ".join(line))
