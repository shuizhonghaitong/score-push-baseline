"""Compute the 15 ClimX annual indices using the OFFICIAL repo code (xclim).

This is the source of truth for the index definitions. Used for local
validation and to build the final submission.

daily : dict var -> (T, lat, lon) float, T days 2015-01-01..2100-12-30 (noleap, 2100->364)
base  : dict var -> (10950, lat, lon) float, 1961-01-01..1990-12-31 (noleap)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import xarray as xr

REF = Path(__file__).resolve().parents[1] / "ref" / "ClimX"
sys.path.insert(0, str(REF))
from src.indices_xclim import SELECTED_INDICES  # noqa: E402

SUB_NAMES = ["TXx", "TNn", "SU", "TR", "FD", "ID", "WSDI", "CSDI", "GSL",
             "Rx5day", "CDD", "CWD", "R95p", "R10mm", "SDII"]


def _time(n, start):
    return xr.cftime_range(start, periods=n, freq="1D", calendar="noleap")


def _dataset(arrs, start, lat, lon):
    t = _time(arrs["tas"].shape[0], start)
    dv = {}
    for v, a in arrs.items():
        da = xr.DataArray(a, dims=["time", "lat", "lon"], coords={"time": t, "lat": lat, "lon": lon})
        if v in ("tas", "tasmax", "tasmin"):
            da.attrs["units"] = "K"
        elif v == "pr":
            da.attrs["units"] = "kg m-2 s-1"
        else:
            da.attrs["units"] = "1"
        dv[v] = da
    return xr.Dataset(dv)


def compute_indices_official(daily, base, start="2015-01-01", base_start="1961-01-01",
                             names=None):
    names = names or SUB_NAMES
    lat = np.zeros(daily["tas"].shape[1], dtype=float)
    lon = np.zeros(daily["tas"].shape[2], dtype=float)
    ds = _dataset(daily, start, lat, lon)
    dsb = _dataset(base, base_start, lat, lon)
    # The repo percentile helpers accept a historical_data_path (str/Path) or a
    # Dataset. Pass the base dataset directly; _open_baseline_dataset only opens
    # Paths, so we monkey-inject via the cache by path is not needed: the helpers
    # check isinstance(historical_data_path, xr.Dataset) first? They do:
    #   _open_baseline_dataset handles str/Path. But _percentile_threshold receives
    #   historical_data_path and calls _open_baseline_dataset. So a Dataset must be
    #   handled. Patch to accept Dataset.
    import src.indices_xclim as ix
    _orig_open = ix._open_baseline_dataset

    def _open_ds(source):
        if isinstance(source, xr.Dataset):
            return source
        return _orig_open(source)

    ix._open_baseline_dataset = _open_ds
    try:
        out = {}
        for name in names:
            fn = SELECTED_INDICES[name]
            res = fn(ds, historical_data_path=dsb, freq="YS")
            out[name] = np.asarray(res.values, dtype=np.float64)
        return out
    finally:
        ix._open_baseline_dataset = _orig_open
