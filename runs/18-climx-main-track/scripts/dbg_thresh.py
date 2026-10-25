import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "scripts")
import numpy as np
import netCDF4 as nc
import xarray as xr
from climx_common import lite_file
from climx_indices import base_period_slice, percentile_doy_window, _hyndman8

ds2 = nc.Dataset(str(lite_file("hist_late")))
base_raw = {v: np.asarray(ds2.variables[v][0], dtype=np.float64) for v in ["tas", "tasmax", "tasmin", "pr"]}
ds2.close()
base = base_period_slice(base_raw)

# official threshold via xclim
import xclim
import xclim.core.calendar as cc
KG = 86400.0
def official_doy_perc(var, per, mm=False):
    a = base[var] * (KG if mm else 1.0)
    t = xr.cftime_range("1961-01-01", periods=a.shape[0], freq="1D", calendar="noleap")
    da = xr.DataArray(a, dims=["time", "lat", "lon"], coords={"time": t, "lat": np.zeros(12), "lon": np.zeros(18)})
    p = cc.percentile_doy(da, window=5, per=per)
    p = p.squeeze("percentiles", drop=True)
    return np.asarray(p.values, dtype=np.float64)[..., 0] if p.dims == ("lat", "lon", "dayofyear", "percentiles") else np.moveaxis(np.asarray(p.values, dtype=np.float64), -1, 0)

for var, per, mm in [("tasmax", 90.0, False), ("tasmin", 10.0, False), ("pr", 95.0, True)]:
    mine = percentile_doy_window(base[var]*(KG if mm else 1.0), per/100.0)
    ref = official_doy_perc(var, per, mm)
    d = np.abs(mine - ref)
    print(f"{var} p{per}: maxabs={d.max():.4g} rms={np.sqrt((d**2).mean()):.4g}")
    # per-doy profile diff
    dd = d.max(axis=(1,2))
    print("   worst doys:", np.argsort(dd)[-6:], " vals:", dd[np.argsort(dd)[-6:]].round(3))
    # check sample sets at one cell, one doy
    if var == "tasmax":
        y, c = 0, 0
        T = base[var].shape[0]; half=2
        # my samples for doy 5 (index 4)
        c0 = 4
        mysamp = base[var][max(c0-2,0):c0+3, c, c].copy()
        # xclim: rolling construct
        t = xr.cftime_range("1961-01-01", periods=T, freq="1D", calendar="noleap")
        da = xr.DataArray(base[var], dims=["time","lat","lon"], coords={"time":t})
        rr = da.rolling(min_periods=1, center=True, time=5).construct("window")
        # select time index 4, cell (c,c)
        xsamp = np.asarray(rr.sel(time=t[4], lat=da.lat[c], lon=da.lon[c]).values, dtype=float)
        print("  my sample (doy5):", np.sort(mysamp), " n=", len(mysamp))
        print("  xr sample (doy5):", np.sort(xsamp))
        # quantiles
        from xclim.core.utils import calc_perc
        q_ref = calc_perc(xsamp[None], percentiles=[per])  # (1, n=1?)
        print("  quantile mine:", _hyndman8(np.sort(mysamp), per/100.0), " xclim:", calc_perc(mysamp[None,:], [per], 1/3, 1/3))
