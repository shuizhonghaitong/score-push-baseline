"""Extract full-res (192x288) annual fields + 15 official indices per scenario.

Usage:
  extract_full.py --scen ssp126 [--make-base]
Writes:
  models/cache/full_{scen}.npz   : idx (15,86,192,288) f32, means (6,86,192,288) f32,
                                   lbc (86,4) f32  [tas,tasmax,tasmin,wind,pr_mean,pr_tot]
  models/cache/base_thresh_full.npz : th_wsdi/th_csdi/th_r95p (365,192,288) f64  [if --make-base]
"""
from __future__ import annotations
import argparse, os, sys, time
from pathlib import Path
import numpy as np
import netCDF4 as nc

sys.path.insert(0, str(Path(__file__).parent))
from climx_common import full_file, lite_file, DATA, RUN
from climx_indices import (
    compute_indices, year_slices, percentile_doy_window, base_period_slice)
from climx_indices_fast import compute_indices_fast

def compute_indices_dispatch(daily: dict, base: dict) -> dict:
    if daily["tasmax"].shape[0] == 31389:
        return compute_indices_fast(daily, base)
    return compute_indices(daily, base)

N_DAYS = 31389
FIELDS = ["tas", "tasmax", "tasmin", "pr", "sfcWind"]
LBC = ["CO2_LBC", "CH4_LBC", "N2O_LBC", "CFC11eq_LBC"]
OUT_DIR = RUN / "models" / "cache"


def read_year(ds, var, s, e):
    return np.asarray(ds.variables[var][0, s:e, :, :], dtype=np.float32)


def extract_scenario(scen: str, base: dict | None, path: str | None = None):
    f = Path(path) if path else full_file(scen)
    out_idx = {}
    out_mean = {}
    ds = nc.Dataset(str(f))
    t0 = time.time()
    n_days = ds.dimensions["time"].size
    ys = year_slices(n_days)
    n_years = len(ys)
    lat, lon = 192, 288
    idx_arr = {k: np.zeros((n_years, lat, lon), dtype=np.float64)
               for k in ["TXx","TNn","SU","TR","FD","ID","GSL","WSDI","CSDI","Rx5day","CDD","CWD","R95p","R10mm","SDII"]}
    mean_tas = np.zeros((n_years, lat, lon), dtype=np.float32)
    mean_tasmax = np.zeros_like(mean_tas)
    mean_tasmin = np.zeros_like(mean_tas)
    mean_wind = np.zeros_like(mean_tas)
    pr_mean = np.zeros_like(mean_tas)
    pr_tot = np.zeros_like(mean_tas)
    have_base = base is not None
    for y, (s, e) in enumerate(ys):
        tas = read_year(ds, "tas", s, e)
        tasmax = read_year(ds, "tasmax", s, e)
        tasmin = read_year(ds, "tasmin", s, e)
        pr = read_year(ds, "pr", s, e)
        wind = read_year(ds, "sfcWind", s, e)
        mean_tas[y] = tas.mean(axis=0)
        mean_tasmax[y] = tasmax.mean(axis=0)
        mean_tasmin[y] = tasmin.mean(axis=0)
        mean_wind[y] = wind.mean(axis=0)
        pr_mm = pr * 86400.0
        pr_mean[y] = pr_mm.mean(axis=0)
        pr_tot[y] = pr_mm.sum(axis=0)
        del tas, tasmax, tasmin, pr, wind
        if y % 10 == 0:
            print(f"  {scen} year {y}/{n_years} {time.time()-t0:.0f}s", flush=True)
    if have_base:
        t0 = time.time()
        daily = {v: np.asarray(ds.variables[v][0, :, :, :], dtype=np.float32) for v in ["tas", "tasmax", "tasmin", "pr"]}
        print(f"  {scen} loaded full series for indices {time.time()-t0:.0f}s", flush=True)
        idx = compute_indices_dispatch(daily, base)
        for k in idx_arr:
            idx_arr[k] = idx[k]
        print(f"  {scen} indices done {time.time()-t0:.0f}s", flush=True)
    lbc = np.zeros((n_years, len(LBC)), dtype=np.float32)
    for j, v in enumerate(LBC):
        if v in ds.variables:
            lbc[:, j] = np.asarray(ds.variables[v][0, :n_years], dtype=np.float32)
    ds.close()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT_DIR / f"full_{scen}.npz",
                        **{f"idx_{k}": v.astype(np.float32) for k, v in idx_arr.items()},
                        mean_tas=mean_tas, mean_tasmax=mean_tasmax, mean_tasmin=mean_tasmin,
                        mean_wind=mean_wind, pr_mean=pr_mean, pr_tot=pr_tot,
                        lbc=lbc, have_base=have_base,
                        base_mode=np.array("full" if have_base else "none"))
    print(f"{scen}: done {time.time()-t0:.0f}s have_base={have_base}", flush=True)
    return OUT_DIR / f"full_{scen}.npz"


def make_base():
    """1961-1990 full-res thresholds from hist_late (1932-2014)."""
    f = full_file("hist_late")
    if not f.exists():
        print(f"missing {f}", flush=True)
        return None
    ds = nc.Dataset(str(f))
    t0 = time.time()
    n_days = ds.dimensions["time"].size
    s = (1961 - 1932) * 365
    e = s + 30 * 365
    tasmax = np.asarray(ds.variables["tasmax"][0, s:e, :, :], dtype=np.float64)
    tasmin = np.asarray(ds.variables["tasmin"][0, s:e, :, :], dtype=np.float64)
    pr = np.asarray(ds.variables["pr"][0, s:e, :, :], dtype=np.float64) * 86400.0
    ds.close()
    th_wsdi = percentile_doy_window(tasmax, 0.90)
    print(f"  base th_wsdi {time.time()-t0:.0f}s", flush=True)
    th_csdi = percentile_doy_window(tasmin, 0.10)
    print(f"  base th_csdi {time.time()-t0:.0f}s", flush=True)
    th_r95p = percentile_doy_window(pr, 0.95)
    print(f"  base th_r95p {time.time()-t0:.0f}s", flush=True)
    # base-era (1961-1990) field means for model baselines
    means = {}
    for v in ["tas", "tasmax", "tasmin", "sfcWind", "pr"]:
        arr = np.asarray(ds2_var(full_file("hist_late"), v, s, e), dtype=np.float32)
        means[f"base_{v}"] = (arr.mean(axis=0) * (86400.0 if v == "pr" else 1.0)).astype(np.float32)
        del arr
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = OUT_DIR / f"base_thresh_full.tmp{os.getpid()}.npz"
    np.savez_compressed(tmp, th_wsdi=th_wsdi, th_csdi=th_csdi, th_r95p=th_r95p, **means)
    tmp.rename(OUT_DIR / "base_thresh_full.npz")
    print(f"base: done {time.time()-t0:.0f}s", flush=True)
    return OUT_DIR / "base_thresh_full.npz"


def ds2_var(f, var, s, e):
    ds = nc.Dataset(str(f))
    a = np.asarray(ds.variables[var][0, s:e, :, :], dtype=np.float32)
    ds.close()
    return a


def pass2_indices(scen: str, base_path: Path | None = None, path: str | None = None,
                  basemode: str = "full"):
    base_path = base_path or (OUT_DIR / "base_thresh_full.npz")
    """Fill WSDI/CSDI/R95p in an existing full_{scen}.npz once base is available.
    Recomputes ALL indices (simpler) and rewrites the file."""
    base_npz = np.load(base_path)
    base = {"tasmax": base_npz["th_wsdi"], "tasmin": base_npz["th_csdi"], "pr": base_npz["th_r95p"]}
    f = Path(path) if path else full_file(scen)
    t0 = time.time()
    n_days = None
    daily = None
    try:
        ds = nc.Dataset(str(f), mmap=True)
        n_days = ds.dimensions["time"].size
        daily = {v: np.asarray(ds.variables[v][0, :, :, :], dtype=np.float32) for v in ["tas", "tasmax", "tasmin", "pr"]}
    except Exception:
        ds.close()
        ds = nc.Dataset(str(f))
        n_days = ds.dimensions["time"].size
        daily = {}
        for v in ["tas", "tasmax", "tasmin", "pr"]:
            arr = np.empty((n_days, 192, 288), dtype=np.float32)
            for a in range(0, n_days, 5000):
                arr[a:a + 5000] = ds.variables[v][0, a:a + 5000]
            daily[v] = arr
    print(f"  {scen} pass2 loaded full series {time.time()-t0:.0f}s", flush=True)
    ds.close()
    t0 = time.time()
    idx = compute_indices_dispatch(daily, base)
    print(f"  {scen} pass2 indices computed {time.time()-t0:.0f}s", flush=True)
    idx_arr = {k: v for k, v in idx.items()}
    p = OUT_DIR / f"full_{scen}.npz"
    old = np.load(p)
    tmp = OUT_DIR / f"full_{scen}.tmp{os.getpid()}.npz"
    data = {k: old[k] for k in old.files if not k.startswith("idx_")}
    data.update({f"idx_{k}": v.astype(np.float32) for k, v in idx_arr.items()})
    data["have_base"] = np.array(True)
    data["base_mode"] = np.array(basemode)
    np.savez_compressed(tmp, **data)
    old.close()
    tmp.rename(p)
    print(f"{scen} pass2 done {time.time()-t0:.0f}s", flush=True)


def make_lite_base_map():
    """Approx base thresholds at full res from LITE hist (12x18) — temporary CV use."""
    from climx_common import nearest_upsample_lite_to_full, read_hist
    a, b = read_hist("tasmax")
    c, d = read_hist("tasmin")
    e, ff = read_hist("pr")
    s = (1961 - 1932) * 365
    e2 = s + 30 * 365
    tasmax = b[s:e2].astype(np.float64)
    tasmin = d[s:e2].astype(np.float64)
    pr = ff[s:e2].astype(np.float64) * 86400.0
    th_wsdi = nearest_upsample_lite_to_full(percentile_doy_window(tasmax, 0.90).astype(np.float32)).astype(np.float64)
    th_csdi = nearest_upsample_lite_to_full(percentile_doy_window(tasmin, 0.10).astype(np.float32)).astype(np.float64)
    th_r95p = nearest_upsample_lite_to_full(percentile_doy_window(pr, 0.95).astype(np.float32)).astype(np.float64)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT_DIR / "base_thresh_lite.npz",
                        th_wsdi=th_wsdi, th_csdi=th_csdi, th_r95p=th_r95p)
    print("lite-mapped base thresholds written", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--scen", default="")
    ap.add_argument("--make-base", action="store_true")
    ap.add_argument("--make-lite-base", action="store_true")
    ap.add_argument("--pass2", action="store_true",
                    help="recompute all indices for --scen using base thresholds")
    ap.add_argument("--basemode", default="full",
                    choices=["full", "lite"],
                    help="which base thresholds npz to use for --pass2")
    ap.add_argument("--file", default="", help="explicit input .nc path")
    a = ap.parse_args()
    if a.make_base:
        make_base()
    if a.make_lite_base:
        make_lite_base_map()
    if a.pass2:
        pass2_indices(a.scen, OUT_DIR / f"base_thresh_{a.basemode}.npz" if a.basemode else None,
                      a.file or None, a.basemode)
    if a.scen and not a.pass2:
        base_path = OUT_DIR / "base_thresh_full.npz"
        base = None
        if base_path.exists():
            bn = np.load(base_path)
            base = {"tasmax": bn["th_wsdi"], "tasmin": bn["th_csdi"], "pr": bn["th_r95p"]}
            bn.close()
        extract_scenario(a.scen, base, a.file or None)
