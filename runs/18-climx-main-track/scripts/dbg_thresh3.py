import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "scripts")
import inspect
import numpy as np
import netCDF4 as nc
import xarray as xr
from climx_common import lite_file
from climx_indices import base_period_slice, _hyndman8

print(inspect.getsource(__import__("xclim.core.utils", fromlist=["_nan_quantile"])._nan_quantile))

ds2 = nc.Dataset(str(lite_file("hist_late")))
base = base_period_slice({v: np.asarray(ds2.variables[v][0], dtype=np.float64) for v in ["tasmax"]})
ds2.close()
a = base["tasmax"]; T = a.shape[0]
t = xr.cftime_range("1961-01-01", periods=T, freq="1D", calendar="noleap")
da = xr.DataArray(a, dims=["time","lat","lon"], coords={"time": t, "lat": np.arange(12), "lon": np.arange(18)})
rr = da.rolling(min_periods=1, center=True, time=5).construct("window")
import pandas as pd
crd = xr.Coordinates.from_pandas_multiindex(
    pd.MultiIndex.from_arrays((rr.time.dt.year.values, rr.time.dt.dayofyear.values), names=("year","dayofyear")), "time")
rr2 = rr.drop_vars("time").assign_coords(crd)
rrr = rr2.unstack("time").stack(stack_dim=("year","window"))

from xclim.core.utils import calc_perc
i, j = 5, 7
for doy0, per in [(0, 90.0), (29, 90.0), (363, 90.0)]:
    raw = np.asarray(rrr.sel(dayofyear=doy0+1, lat=i, lon=j).values)  # (150,) with nans
    q_xclim = calc_perc(raw[None], percentiles=[per], alpha=1/3, beta=1/3)
    valid = raw[~np.isnan(raw)]
    q_mine = _hyndman8(np.sort(valid), per/100.0)
    print(f"doy {doy0} p{per}: xclim={q_xclim:.6f} mine={q_mine:.6f} n={valid.size}")
    # manual HF method 8
    n = valid.size
    s = np.sort(valid)
    gamma = 1 - 1/3 - 1/3
    p_ = per/100.0
    h = gamma + n * (1/3 + p_ - 1) / (n - 1)
    print("   HF8 manual h:", h, "interp:", np.interp(h, np.arange(1, n+1), s))
