import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "scripts")
import numpy as np
import train_v4 as t4
import train_v5 as t5
from climx_indices import percentile_doy_window

Cn = 216
model = t4.V4(hist_start=1961, sigma=2.0)
model.fit(["ssp126", "ssp370", "ssp585"])
import netCDF4 as nc
from climx_indices import base_period_slice
ds2 = nc.Dataset(str(t4.lite_file("hist_late")))
model.base = base_period_slice({v: np.asarray(ds2.variables[v][0], dtype=np.float64) for v in t4.VARS})
ds2.close()

th_wsdi = percentile_doy_window(model.base["tasmax"], 0.90)
S = model.S["tasmax"]  # (365,12,18)

# pick a mid-lat cell with active WSDI
cell = (7, 10)
i, j = cell
# GT: WSDI and tasmax annual mean for the 3 ssps
rows = []
for s in t4.SSPS:
    ds = t4.lite_file(s)
    import netCDF4 as nc
    d = nc.Dataset(str(ds))
    tx = np.asarray(d.variables["tasmax"][0], dtype=np.float64)
    F = np.log(np.asarray(d.variables["CO2_LBC"][0]) / model.F_test[s][0] * model.F_test[s][0]) if False else model.F_test[s][:85]
    d.close()
    for y in range(85):
        yr = tx[y*365:(y+1)*365, i, j]
        L = yr.mean()
        mask = (yr > th_wsdi[:, i, j]).astype(bool)
        w = t5.windowed_run_sum(mask.reshape(365,1), 6)[0]
        txx = yr.max()
        rows.append((L, w, txx, F[y]))
rows = np.array(rows)
print(f"cell {cell}: n={len(rows)}")
print("L range:", rows[:,0].min().round(2), rows[:,0].max().round(2))
print("WSDI range:", rows[:,1].min().round(1), rows[:,1].max().round(1))
print("TXx range:", rows[:,2].min().round(2), rows[:,2].max().round(2))

# WSDI of smooth field as function of level shift
print("\nSmooth field WSDI/TXx/SU vs level shift (K) at F=0.3 (slope 0):")
for shift in np.arange(-2.0, 4.01, 0.5):
    field = S + shift  # (365,12,18)
    w = t5.windowed_run_sum((field > th_wsdi).astype(bool), 6)[i, j]
    txx = field[:, i, j].max()
    su = (field[:, i, j] > 298.15).sum()
    print(f"  shift {shift:+.1f}: WSDI={w:6.1f}  TXx={txx:7.2f}  SU={su:3d}")

# noisy expectations binned by L
print("\nNoisy GT binned by level L (mean WSDI, mean TXx, n):")
import collections
bins = np.linspace(rows[:,0].min(), rows[:,0].max()+1e-9, 8)
for a, b_ in zip(bins[:-1], bins[1:]):
    m = (rows[:,0] >= a) & (rows[:,0] < b_)
    if m.sum() == 0: continue
    print(f"  L [{a:7.2f},{b_:7.2f}) n={m.sum():3d}: WSDI={rows[m,1].mean():7.1f}  TXx={rows[m,2].mean():7.2f}")
