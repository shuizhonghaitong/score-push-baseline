import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "scripts")
import numpy as np, netCDF4 as nc
import train_v4 as t4
from climx_indices import compute_indices
from rnNSE_lite import score_lite
from climx_common import lite_file, SUB_COLUMNS

scen = "ssp370"
model = t4.V4(hist_start=1961, sigma=1.5)
model.fit(["ssp126", "ssp585"])
ds2 = nc.Dataset(str(lite_file("hist_late")))
from climx_indices import base_period_slice
base = base_period_slice({v: np.asarray(ds2.variables[v][0], dtype=np.float64) for v in t4.VARS})
ds2.close()
model.base = base
gt = t4.gt_indices_npz(__import__("pathlib").Path("models/cache"), scen)

# 1) s=0 climatology only
sh0 = dict(tasmax=0.0, tasmin=0.0, tas=0.0, pr=0.0)
d = model.predict_daily(scen, sh0)
idx = compute_indices(d, base)
S, per = score_lite(gt, idx)
print(f"climatology-only: S={S:+.5f}")
for k in SUB_COLUMNS: print(f"   {k:8s} {per[k]:+.4f}")

# 2) s=1 full
sh1 = dict(tasmax=1.0, tasmin=1.0, tas=1.0, pr=1.0)
d = model.predict_daily(scen, sh1)
idx = compute_indices(d, base)
S, per = score_lite(gt, idx)
print(f"full s=1: S={S:+.5f}")
for k in SUB_COLUMNS: print(f"   {k:8s} {per[k]:+.4f}")

# 3) trajectory check: predicted vs GT index mean over cells (time series)
def ts(idx):
    return idx.reshape(86, -1).mean(axis=1)
gt_ts = {k: ts(gt[k]) for k in SUB_COLUMNS}
pred_ts = {k: ts(idx[k]) for k in SUB_COLUMNS}
print("\nTrajectory (year: gt -> pred) for selected:")
for k in ["TXx", "TNn", "SU", "GSL", "R10mm", "CDD"]:
    a, b = gt_ts[k], pred_ts[k]
    print(f"  {k:8s} 2015: {a[0]:8.2f} -> {b[0]:8.2f} | 2050: {a[35]:8.2f} -> {b[35]:8.2f} | 2100: {a[85]:8.2f} -> {b[85]:8.2f}")
    # per-cell R2 of the index time series
    va = a.var(); mse = ((a-b)**2).mean()
    print(f"           global R2={1-mse/va:+.3f}")

# 4) ORACLE: fit m on the test scenario itself (cheating) to see structure ceiling
def oracle_m(scen):
    ds = nc.Dataset(str(lite_file(scen)))
    out = {}
    co2_ref = np.asarray(ds.variables["CO2_LBC"][0][0])
    F = np.log(np.asarray(ds.variables["CO2_LBC"][0]) / co2_ref)
    for v in t4.VARS:
        a = np.asarray(ds.variables[v][0], dtype=np.float64)
        A = np.stack([getattr(a[y*365:(y+1)*365], t4.AGG[v])(axis=0) for y in range(85)])
        Fc = F[:85] - F[:85].mean()
        m = (A * Fc[:, None, None]).sum(axis=0) / (Fc**2).sum()
        out[v] = m
    ds.close()
    return out
om = oracle_m(scen)
model.m = om  # cheating
# but S/Fbar must also be test-scenario consistent; keep train S/Fbar for now
d = model.predict_daily(scen, dict(tasmax=1.0, tasmin=1.0, tas=1.0, pr=1.0))
idx = compute_indices(d, base)
S, per = score_lite(gt, idx)
print(f"\nORACLE m (cheating) s=1: S={S:+.5f}")
for k in SUB_COLUMNS: print(f"   {k:8s} {per[k]:+.4f}")
