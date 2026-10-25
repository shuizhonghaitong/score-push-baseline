"""Stream-read the ssp585 full file variable by variable; locate first HDF failure."""
import sys, time
import numpy as np
import netCDF4 as nc

f = "data/NorESM2-MM_r1i1p1f1_with_forcing_train_ssp585.nc"
t0 = time.time()
ds = nc.Dataset(f, mmap=False)
print("vars:", list(ds.variables.keys()), flush=True)
step = 1000
bad = []
for v in ds.variables.keys():
    var = ds[v]
    if var.ndim < 2 or var.shape[0] < 100:
        continue
    n = var.shape[1] if var.ndim >= 3 else var.shape[0]
    for a in range(0, n, step):
        b = min(a + step, n)
        try:
            x = np.asarray(var[0, a:b, :, :] if var.ndim == 4 else var[a:b], dtype=np.float32)
            assert np.isfinite(x).all()
        except Exception as e:
            bad.append((v, a, b, repr(e)[:120]))
            print(f"BAD {v} [{a}:{b}]: {e!r}", flush=True)
            break
    else:
        print(f"OK {v} (n={n}) {time.time()-t0:.0f}s", flush=True)
ds.close()
print("DONE", flush=True)
