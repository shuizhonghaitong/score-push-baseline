import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "scripts")
import numpy as np
import pandas as pd
import netCDF4 as nc
import xarray as xr
from climx_common import lite_file
from climx_indices import base_period_slice

ds2 = nc.Dataset(str(lite_file("hist_late")))
base = base_period_slice({v: np.asarray(ds2.variables[v][0], dtype=np.float64) for v in ["tas","tasmax","tasmin","pr"]})
ds2.close()

a = base["tasmax"]  # (10950, 12, 18)
T = a.shape[0]
t = xr.cftime_range("1961-01-01", periods=T, freq="1D", calendar="noleap")
da = xr.DataArray(a, dims=["time","lat","lon"], coords={"time": t, "lat": np.arange(12), "lon": np.arange(18)})
rr = da.rolling(min_periods=1, center=True, time=5).construct("window")

# replicate xclim grouping: (year, dayofyear)
crd = xr.Coordinates.from_pandas_multiindex(
    pd.MultiIndex.from_arrays((rr.time.dt.year.values, rr.time.dt.dayofyear.values), names=("year","dayofyear")), "time")
rr2 = rr.drop_vars("time").assign_coords(crd)
rrr = rr2.unstack("time").stack(stack_dim=("year","window"))  # (dayofyear, lat, lon, stack)

def xclim_samples(doy0, i, j):
    # doy0: 0-based
    arr = np.asarray(rrr.sel(dayofyear=doy0+1, lat=i, lon=j).values)
    return arr  # (n_samples,) possibly with nans

def my_samples(doy0, i, j):
    half = 2
    out = []
    for y in range(30):
        c = y*365 + doy0
        lo, hi = max(c-half,0), min(c+2, T-1)
        out.append(a[lo:hi+1, i, j])
    return np.concatenate(out)

# check boundaries first
for tidx in [0, 1, 2, T-3, T-2, T-1]:
    xs = np.asarray(rr.sel(time=t[tidx], lat=5, lon=7).values)
    print(f"t={tidx}: xr window size={xs.size} values={xs}")

i, j = 5, 7
for doy0 in [0, 1, 2, 29, 363, 364]:
    xs = xclim_samples(doy0, i, j)
    ys = my_samples(doy0, i, j)
    xs_c = xs[~np.isnan(xs)]
    print(f"doy {doy0}: xr n={xs.size} (valid {xs_c.size}), mine n={ys.size}")
    if xs_c.size == ys.size:
        print("   sets equal:", np.allclose(np.sort(xs_c), np.sort(ys)))
    else:
        print("   XR:", np.sort(xs_c))
        print("   MY:", np.sort(ys))
