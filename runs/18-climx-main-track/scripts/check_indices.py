"""Cross-check numpy indices vs official xclim indices on lite data (12x18)."""
import sys, time, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "scripts")
import numpy as np
import netCDF4 as nc
from climx_common import lite_file, SUB_COLUMNS
from climx_indices import compute_indices, base_period_slice
from climx_indices_official import compute_indices_official

scen = sys.argv[1] if len(sys.argv) > 1 else "ssp370"

ds = nc.Dataset(str(lite_file(scen)))
daily = {v: np.asarray(ds.variables[v][0], dtype=np.float64) for v in ["tas", "tasmax", "tasmin", "pr"]}
ds.close()
ds2 = nc.Dataset(str(lite_file("hist_late")))
base_raw = {v: np.asarray(ds2.variables[v][0], dtype=np.float64) for v in ["tas", "tasmax", "tasmin", "pr"]}
ds2.close()
base = base_period_slice(base_raw)

t0 = time.time()
ref = compute_indices_official(daily, base)
t1 = time.time()
mine = compute_indices(daily, base)
t2 = time.time()
print(f"official {t1-t0:.1f}s  numpy {t2-t1:.1f}s")
for k in SUB_COLUMNS:
    a, b = ref[k], mine[k]
    d = np.abs(a - b)
    scale = max(1.0, np.abs(a).max())
    print(f"{k:8s} shape {a.shape} maxabs={d.max():.6g} rms={np.sqrt((d**2).mean()):.3g} ref range [{a.min():.4g},{a.max():.4g}]")
