"""Global-land annual-mean aerosol series + extra LBC from full files.
Usage: extract_aerosol.py ssp126 [ssp370 ...] test
Writes models/cache/aero_annual_{scen}.npz with keys aero (86,5), cf2cl2 (86,).
"""
import sys, time
import numpy as np
import netCDF4 as nc
from pathlib import Path

sys.path.insert(0, "scripts")
from climx_common import full_file, ar6_mask_full

AERO = ["BC_AX", "BC_N", "SO2", "SO4_PR", "OM_NI"]
land = (ar6_mask_full() >= 0).reshape(192, 288)

for scen in sys.argv[1:]:
    f = full_file(scen if scen.startswith("ssp") else "test_forcings")
    t0 = time.time()
    ds = nc.Dataset(str(f))
    out = {}
    for v in AERO:
        x = ds[v][0]  # (1032,192,288) monthly
        x = x.reshape(86, 12, 192, 288)[:, :, land].astype(np.float32)  # (86,12,Nland)
        out[v] = x.mean(axis=(1, 2)).astype(np.float32)  # (86,)
    if "CF2CL2_LBC" in ds.variables:
        out["cf2cl2"] = np.asarray(ds["CF2CL2_LBC"][0, :86], dtype=np.float32)
    ds.close()
    o = Path("models/cache") / f"aero_annual_{scen}.npz"
    np.savez_compressed(o, **out)
    print(f"[aero] {scen} -> {o} {time.time()-t0:.0f}s", flush=True)
print("[aero] ALL_DONE", flush=True)
