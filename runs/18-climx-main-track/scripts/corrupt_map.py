"""Map which (variable, year-slice) regions of the ssp585 file fail to read."""
import sys, time
import numpy as np
import netCDF4 as nc

f = sys.argv[1] if len(sys.argv) > 1 else "data/NorESM2-MM_r1i1p1f1_with_forcing_train_ssp585.nc"
ds = nc.Dataset(f)
step = 2000
out = {}
for v in ["huss", "pr", "psl", "sfcWind", "tas", "tasmax", "tasmin"]:
    var = ds[v]
    n = var.shape[1]
    bad = []
    for a in range(0, n, step):
        b = min(a + step, n)
        try:
            x = np.asarray(var[0, a:b, 40:60, 100:120], dtype=np.float32)
            assert np.isfinite(x).all()
        except Exception:
            bad.append((a, b))
    out[v] = bad
    print(f"{v}: bad slices {bad}", flush=True)
ds.close()
anybad = any(v for v in out.values())
print("MAP_DONE", "CORRUPTED" if anybad else "CLEAN", flush=True)
import sys
sys.exit(1 if anybad else 0)
