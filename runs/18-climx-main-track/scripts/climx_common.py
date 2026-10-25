"""Shared helpers for the ClimX run: paths, time slicing, grid maps."""
from __future__ import annotations

import netCDF4 as nc
import numpy as np
from pathlib import Path

RUN = Path("/mnt/disk3/liuyutong_home/workspace/skill/score-push/runs/18-climx-main-track")
DATA = RUN / "data"

TARGET_VARS = ["tas", "tasmax", "tasmin", "pr", "huss", "psl", "sfcWind"]
LITE_PREFIX = "NorESM2-MM_r1i1p1f1_with_forcing_subsampled_16_"
FULL_PREFIX = "NorESM2-MM_r1i1p1f1_with_forcing_"

LITE_FILES = {
    "hist_early": f"{LITE_PREFIX}hist_train_1850_1931.nc",
    "hist_late": f"{LITE_PREFIX}hist_train_1932_2014.nc",
    "ssp126": f"{LITE_PREFIX}train_ssp126.nc",
    "ssp370": f"{LITE_PREFIX}train_ssp370.nc",
    "ssp585": f"{LITE_PREFIX}train_ssp585.nc",
    "forcings_all": f"{LITE_PREFIX}forcings_all.nc",
    "test_forcings": f"{LITE_PREFIX}test_forcings_ssp245.nc",
}
FULL_FILES = {
    "hist_early": f"{FULL_PREFIX}hist_train_1850_1931.nc",
    "hist_late": f"{FULL_PREFIX}hist_train_1932_2014.nc",
    "ssp126": f"{FULL_PREFIX}train_ssp126.nc",
    "ssp370": f"{FULL_PREFIX}train_ssp370.nc",
    "ssp585": f"{FULL_PREFIX}train_ssp585.nc",
    "forcings_all": f"{FULL_PREFIX}forcings_all.nc",
    "test_forcings": f"{FULL_PREFIX}test_forcings_ssp245.nc",
}

SSPS = ["ssp126", "ssp370", "ssp585"]

def lite_file(key: str) -> Path:
    return DATA / LITE_FILES[key]

def full_file(key: str) -> Path:
    return DATA / FULL_FILES[key]

def projection_year_slices(n_days: int = 31389) -> list[int]:
    """Start index of each of the 86 years (2015..2100, last has 364 days)."""
    ys = [0]
    for y in range(86):
        ys.append(ys[-1] + (364 if y + 1 == 86 else 365))
    assert ys[-1] == n_days, (ys[-1], n_days)
    return ys

def read_scenario(var: str, key: str, days: int | None = None) -> np.ndarray:
    """Return (time, lat, lon) float32 for one target var of one lite file."""
    ds = nc.Dataset(str(lite_file(key)))
    a = np.asarray(ds.variables[var][0], dtype=np.float32)
    ds.close()
    if days is not None:
        a = a[:days]
    return a

def read_hist(var: str, days: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (values 1850-1931, values 1932-2014, lat, lon) concatenated-ready."""
    a = read_scenario(var, "hist_early")
    b = read_scenario(var, "hist_late")
    return a, b

def lite_latlon() -> tuple[np.ndarray, np.ndarray]:
    ds = nc.Dataset(str(lite_file("test_forcings")))
    lat = np.asarray(ds.variables["lat"][:], dtype=np.float64)
    lon = np.asarray(ds.variables["lon"][:], dtype=np.float64)
    ds.close()
    return lat, lon

# ---- full-grid constants (from rnNSE.py / submission spec) ----
NLAT_FULL, NLON_FULL = 192, 288

# rnNSE grid latitudes: linspace -90..90, 192 points (see rnNSE.py _LATS)
LATS_FULL = np.linspace(-90.0, 90.0, NLAT_FULL)
LONS_FULL = 1.25 * np.arange(NLON_FULL)  # 1.25 deg cells, centers at 1.25*k

def ar6_mask_full() -> np.ndarray:
    """Decode the RLE region mask embedded in rnNSE.py -> (192, 288) int, -1 ocean."""
    import re
    src = (RUN / "ref/ClimX/rnNSE.py").read_text()
    m = re.search(r"_MASK_RLE = \[(.*?)\]", src, re.S)
    body = m.group(1)
    pairs = re.findall(r"\(\s*(-?\d+)\s*,\s*(\d+)\s*\)", body)
    flat = np.concatenate([np.full(int(n), int(r), dtype=np.int16) for r, n in pairs])
    assert flat.size == NLAT_FULL * NLON_FULL, flat.size
    return flat.reshape(NLAT_FULL, NLON_FULL)

_LI_LJ = None

def _li_lj() -> tuple[np.ndarray, np.ndarray]:
    global _LI_LJ
    if _LI_LJ is None:
        lat_l, lon_l = lite_latlon()
        d = np.abs(LATS_FULL[:, None] - lat_l[None, :])
        li = d.argmin(axis=1)
        d2 = np.abs((LONS_FULL % 360)[:, None] - lon_l[None, :])
        d2 = np.minimum(d2, 360 - d2)
        lj = d2.argmin(axis=1)
        _LI_LJ = (li, lj)
    return _LI_LJ


def nearest_upsample_lite_to_full(a: np.ndarray) -> np.ndarray:
    """Nearest-neighbor map from the 12x18 lite grid to the 192x288 full grid.
    a has trailing dims (lat12, lon18) plus optional leading dims (e.g. (86,12,18));
    returns a with the last two dims replaced by (192, 288)."""
    li, lj = _li_lj()
    return a[..., li[:, None], lj[None, :]]

# Submission: 86 years x 192 lat x 288 lon, year-major order
N_SUB_ROWS = 86 * NLAT_FULL * NLON_FULL
SUB_COLUMNS = ["TXx", "TNn", "SU", "TR", "FD", "ID", "WSDI", "CSDI", "GSL",
               "Rx5day", "CDD", "CWD", "R95p", "R10mm", "SDII"]

def make_submission_parquet(index_vals: dict[str, np.ndarray], out_path: Path):
    """index_vals: name -> (86, 192, 288) float. Writes parquet with row_id + 15 cols."""
    import pyarrow as pa
    import pyarrow.parquet as pq
    cols = {}
    for name in SUB_COLUMNS:
        v = np.ascontiguousarray(index_vals[name], dtype=np.float32).reshape(-1)
        assert v.size == N_SUB_ROWS
        assert np.isfinite(v).all(), f"non-finite values in {name}"
        cols[name] = pa.array(v, type=pa.float32())
    table = pa.table({"row_id": pa.array(np.arange(N_SUB_ROWS, dtype=np.int64)), **cols})
    pq.write_table(table, str(out_path), compression="snappy")
    return out_path
