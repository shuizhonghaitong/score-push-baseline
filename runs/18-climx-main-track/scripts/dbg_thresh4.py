import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "scripts")
import numpy as np, pandas as pd
import netCDF4 as nc
import xarray as xr
from climx_common import lite_file
from climx_indices import base_period_slice, percentile_doy_window

ds2 = nc.Dataset(str(lite_file("hist_late")))
base = base_period_slice({v: np.asarray(ds2.variables[v][0], dtype=np.float64) for v in ["tas","tasmax","tasmin","pr"]})
ds2.close()

a = base["tasmax"]; T = a.shape[0]
t = xr.cftime_range("1961-01-01", periods=T, freq="1D", calendar="noleap")
da = xr.DataArray(a, dims=["time","lat","lon"], coords={"time": t, "lat": np.arange(12), "lon": np.arange(18)})
rr = da.rolling(min_periods=1, center=True, time=5).construct("window")
crd = xr.Coordinates.from_pandas_multiindex(
    pd.MultiIndex.from_arrays((rr.time.dt.year.values, rr.time.dt.dayofyear.values), names=("year","dayofyear")), "time")
rr2 = rr.drop_vars("time").assign_coords(crd)
rrr = rr2.unstack("time").stack(stack_dim=("year","window"))  # (lat, lon, dayofyear, stack_dim)

from xclim.core.utils import calc_perc
th_x = np.asarray(calc_perc(np.asarray(rrr.values), percentiles=[90.0], alpha=1/3, beta=1/3))  # (lat,lon,doy,1)
th_x = th_x[..., 0]  # (lat, lon, 365) -> transpose
th_x = np.transpose(th_x, (2, 0, 1))  # (365, lat, lon)
mine = percentile_doy_window(a, 0.90)  # (365, 12, 18)
d = np.abs(mine - th_x)
doy_k, i, j = [int(x) for x in np.unravel_index(int(d.argmax()), d.shape)]  # (doy, i, j)
print("worst: doy", doy_k, "cell", (i, j), "xclim", th_x[doy_k,i,j], "mine", mine[doy_k,i,j])
raw = np.asarray(rrr.values[i, j, doy_k, :])
valid = raw[~np.isnan(raw)]
half=2
out=[]
for y in range(30):
    c = y*365 + doy_k
    lo2, hi2 = max(c-half,0), min(c+2, T-1)
    out.append(a[lo2:hi2+1, i, j])
mine_s = np.concatenate(out)
print("n xclim:", valid.size, "n mine:", mine_s.size)
print("sets equal:", np.allclose(np.sort(valid), np.sort(mine_s)))
if not np.allclose(np.sort(valid), np.sort(mine_s)):
    sv, sm = np.sort(valid), np.sort(mine_s)
    n = min(len(sv), len(sm))
    diffs = np.abs(sv[:n]-sm[:n])
    w = diffs.argmax()
    print("first big diff at sorted idx", w, sv[w-2:w+3], sm[w-2:w+3])
