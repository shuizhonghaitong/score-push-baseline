"""Cross-check numpy index implementation vs official xclim-based code."""
import sys
sys.path.insert(0, "scripts")
import numpy as np
import xarray as xr
import netCDF4 as nc
from pathlib import Path

from climx_common import lite_file, TARGET_VARS
from climx_indices import compute_indices, base_period_slice, year_slices

def to_xr(arr, var, start="2015-01-01"):
    time = xr.cftime_range(start, periods=arr.shape[0], freq="1D", calendar="noleap")
    da = xr.DataArray(arr, dims=["time", "lat", "lon"],
                      coords={"time": time, "lat": [0.0], "lon": [0.0]})
    if var in ("tas", "tasmax", "tasmin"):
        da.attrs["units"] = "K"
    elif var == "pr":
        da.attrs["units"] = "kg m-2 s-1"
    else:
        da.attrs["units"] = "1"
    return da

def main():
    # use a small slice to keep xclim fast: 10 lat x 10 lon of ssp126
    ds126 = nc.Dataset(str(lite_file("ssp126")))
    dslate = nc.Dataset(str(lite_file("hist_late")))
    sl = (slice(None), slice(2,12), slice(4,14))
    daily_np = {v: np.asarray(ds126.variables[v][0][:, 2:12, 4:14], dtype=np.float64) for v in ["tas", "tasmax", "tasmin", "pr"]}
    base_np = {v: np.asarray(dslate.variables[v][0][:, 2:12, 4:14], dtype=np.float64) for v in ["tas", "tasmax", "tasmin", "pr"]}
    ds126.close(); dslate.close()
    base_np = base_period_slice(base_np)

    # numpy implementation
    out_np = compute_indices(daily_np, base_np)

    # xclim implementation (official repo code)
    sys.path.insert(0, "ref/ClimX")
    from src.indices_xclim import SELECTED_INDICES
    time = xr.cftime_range("2015-01-01", periods=daily_np["tas"].shape[0], freq="1D", calendar="noleap")
    tb = xr.cftime_range("1961-01-01", periods=base_np["tas"].shape[0], freq="1D", calendar="noleap")
    lat = np.zeros(daily_np["tas"].shape[1]); lon = np.zeros(daily_np["tas"].shape[2])
    def ds_of(d):
        data_vars = {}
        for v in d:
            t = time if d is daily_np else tb
            da = xr.DataArray(d[v], dims=["time", "lat", "lon"], coords={"time": t, "lat": lat, "lon": lon})
            da.attrs["units"] = "K" if v in ("tas", "tasmax", "tasmin") else ("kg m-2 s-1" if v == "pr" else "1")
            data_vars[v] = da
        return xr.Dataset(data_vars)
    ds = ds_of(daily_np)
    ds_base = ds_of(base_np)

    maxrel = {}
    for name, fn in SELECTED_INDICES.items():
        try:
            x = fn(ds, historical_data_path=ds_base)  # pass dataset directly? code checks isinstance Path -> opens; Dataset passes through _open_baseline_dataset? No: _open_baseline_dataset expects str/Path. Use _percentile cache path... pass via kwarg historical_data_path only works for path. Workaround: monkeypatch not needed - call with historical_data_path=None for the 12 non-percentile.
        except Exception as e:
            pass
    # Simpler: call functions with historical dataset injected through the code's own helper
    from src import indices_xclim as ix
    def call(name, fn, **kw):
        kw.setdefault("freq", "YS")
        return fn(ds, **kw)
    for name in ["TXx", "TNn", "SU", "TR", "FD", "ID", "GSL", "Rx5day", "CDD", "CWD", "R10mm", "SDII"]:
        xc = call(name, SELECTED_INDICES[name])
        xn = out_np[name]
        xc = np.asarray(xc.values, dtype=np.float64)
        d = np.abs(xc - xn)
        scale = np.maximum(np.abs(xc), 1e-6).max()
        maxrel[name] = d.max() / max(scale, 1e-300)
        print(f"{name}: maxabsdiff={d.max():.6g} rel={d.max()/max(scale,1e-300):.3e}")
    # percentile-based ones: use internal _percentile_threshold with the base dataset
    import xclim.core.calendar as cal
    def doy_th(var, q):
        da = ds_base[var].sel(time=slice("1961-01-01", "1990-12-31"))
        p = cal.percentile_doy(da, window=5, per=q).squeeze("percentiles", drop=True)
        return p
    th90tx = doy_th("tasmax", 90)
    th10tn = doy_th("tasmin", 10)
    th95pr = doy_th("pr", 95)
    def align(th):
        return th.sel(dayofyear=ds["tas"].time.dt.dayofyear)
    # WSDI
    from xclim.indices import run_length as rl
    mask = ds["tasmax"] > align(th90tx)
    wsd_xc = np.zeros((86, 10, 10))
    csd_xc = np.zeros_like(wsd_xc)
    for y, (s, e) in enumerate(year_slices(mask.time.size)):
        sub = mask.sel(time=slice(str(2015+y), str(2015+y)+"-12-31"))
        wsd_xc[y] = np.asarray(rl.windowed_run_count(sub, window=6, dim="time").values)
        sub2 = ds["tasmin"] < align(th10tn)
        sub2 = sub2.sel(time=slice(str(2015+y), str(2015+y)+"-12-31"))
        csd_xc[y] = np.asarray(rl.windowed_run_count(sub2, window=6, dim="time").values)
    for nm, xc in [("WSDI", wsd_xc), ("CSDI", csd_xc)]:
        xn = out_np[nm]
        d = np.abs(xc - xn); scale = max(np.abs(xc).max(), 1e-9)
        print(f"{nm}: maxabsdiff={d.max():.6g} rel={d.max()/scale:.3e}")
    # R95p
    pr_mm = ds["pr"] * 86400
    thr = th95pr.sel(dayofyear=pr_mm.time.dt.dayofyear)
    r95_xc = (pr_mm.where(pr_mm.groupby("time.dayofyear") > thr)).resample(time="YS").sum(dim="time").values
    xn = out_np["R95p"]
    d = np.abs(np.asarray(r95_xc, dtype=np.float64) - xn); scale = max(np.abs(xn).max(), 1e-9)
    print(f"R95p: maxabsdiff={d.max():.6g} rel={d.max()/scale:.3e}")

if __name__ == "__main__":
    main()
