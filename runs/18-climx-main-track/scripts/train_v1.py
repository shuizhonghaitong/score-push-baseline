"""V1 forcing-response emulator.

Per (cell, var, doy): fit value = b + m * F with F = ln(CO2(t)/CO2_2015).
Train on hist (optional window) + N SSP scenarios; predict the held-out
scenario's daily fields (noise-free scenario-mean trajectory).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import netCDF4 as nc

from climx_common import TARGET_VARS, lite_file, NLAT_FULL, NLON_FULL
from climx_indices import base_period_slice
from climx_indices_official import compute_indices_official

HIST_EARLY = lite_file("hist_early")
HIST_LATE = lite_file("hist_late")

N_HIST_DAYS = 60224  # 1850-2014
N_PROJ_DAYS = 31389  # 2015-2100 (2100: 364 days)


def co2_yearly(key: str) -> np.ndarray:
    """(n_years,) CO2 (mol/mol) for a scenario file; hist -> 165 years, ssp -> 86."""
    ds = nc.Dataset(str(lite_file(key)))
    c = np.asarray(ds.variables["CO2_LBC"][0], dtype=np.float64)
    yrs = ds.variables["time_year"]
    y0 = nc.num2date(yrs[0], yrs.units, yrs.calendar).year
    ds.close()
    return c, y0


def load_forcings_co2() -> dict:
    """scenario -> (co2 array, first year)."""
    out = {}
    for key in ["hist_early", "hist_late", "ssp126", "ssp370", "ssp585", "test_forcings"]:
        c, y0 = co2_yearly(key)
        out[key] = (c, y0)
    # hist co2: concat early(1850-1931) + late(1932-2014)
    out["hist"] = (np.concatenate([out["hist_early"][0], out["hist_late"][0]]), 1850)
    out["ssp245"] = out["test_forcings"]
    return out



def _load_var(key: str, var: str) -> np.ndarray:
    ds = nc.Dataset(str(lite_file(key)))
    v = np.asarray(ds.variables[var][0], dtype=np.float32)
    ds.close()
    return v

def main(holdout: str | None = "ssp370", hist_start: int = 1961,
         out_prefix: str = "models/v1", save: bool = True):
    t0 = time.time()
    run = Path(__file__).resolve().parents[1]
    outdir = run / out_prefix
    outdir.mkdir(parents=True, exist_ok=True)

    co2 = load_forcings_co2()
    co2_ref = co2["ssp126"][0][0]  # 2015 value (same across scenarios)
    # F per (scenario, year): ln(CO2/co2_2015)
    F = {}
    for s, (c, y0) in co2.items():
        if s in ("hist_early", "hist_late"):
            continue
        F[s] = np.log(c / co2_ref)

    # sample list: (scenario, year_index)
    samples = []
    hist_y0 = co2["hist"][1]
    for y in range(hist_start, 2014):  # 2014 has 364 days
        samples.append(("hist", y - hist_y0))
    train_ssps = [s for s in ["ssp126", "ssp370", "ssp585"] if s != holdout]
    for s in train_ssps:
        for y in range(85):  # 2015..2099 (2100 has 364 days)
            samples.append((s, y))
    Fs = np.array([F[s][yi] for s, yi in samples])
    n_samples = len(samples)
    print(f"samples: {n_samples} (hist from {hist_start}), F range [{Fs.min():.3f}, {Fs.max():.3f}]")

    n_doy, n_lat, n_lon = 365, 12, 18
    params = {}
    for var in TARGET_VARS:
        # load scenario arrays once
        h1 = _load_var("hist_early", var)
        h2 = _load_var("hist_late", var)
        hist = np.concatenate([h1, h2], axis=0)
        del h1, h2
        ssps = {s: _load_var(s, var) for s in train_ssps}
        Y = np.empty((n_samples, n_doy, n_lat, n_lon), dtype=np.float32)
        si = 0
        for s, yi in samples:
            if s == "hist":
                v = hist
            else:
                v = ssps[s]
            Y[si] = v[yi * 365: yi * 365 + 365]
            si += 1
        del hist, ssps
        # OLS: y = b + m*F  (per (doy, lat, lon))
        Fc = Fs - Fs.mean()
        varF = (Fc ** 2).sum()
        Y64 = Y.astype(np.float64)
        m = (Y64 * Fc[:, None, None, None]).sum(axis=0) / varF
        b = Y64.mean(axis=0) - m * Fs.mean()
        params[var] = (b.astype(np.float32), m.astype(np.float32))
        print(f"{var}: fit {time.time()-t0:.0f}s | m range [{m.min():.4g}, {m.max():.4g}]")
        del Y, Y64

    # ---- predictions for all scenarios (held-out + others for diagnostics) ----
    pred = {}
    for s in ([holdout] if holdout else []) + ["ssp245"]:
        Fs_s = F[s]
        flat = {}
        for var in TARGET_VARS:
            b, m = params[var]
            d = b[None, :, :, :] + m[None, :, :, :] * Fs_s[:, None, None, None]  # (86, 365, 12, 18)
            arr = np.empty((31389, 12, 18), dtype=np.float32)
            n = 0
            for y in range(86):
                nn = 364 if y == 85 else 365
                arr[n:n+nn] = d[y][:nn]
                n += nn
            flat[var] = arr
        pred[s] = flat
    print(f"predictions done {time.time()-t0:.0f}s")

    if save:
        tag = f"holdout_{holdout}" if holdout else "full"
        d = {}
        for v in TARGET_VARS:
            d[v + "_b"] = params[v][0]
            d[v + "_m"] = params[v][1]
        np.savez_compressed(outdir / f"params_v1_{tag}.npz", **d,
                            F_hist=F["hist"], F_ssp126=F["ssp126"], F_ssp370=F["ssp370"],
                            F_ssp585=F["ssp585"], F_ssp245=F["ssp245"], co2_ref=co2_ref)
        for s in pred:
            np.savez_compressed(outdir / f"pred_v1_{s}_{tag}.npz", **pred[s])
        print(f"saved to {outdir}")
    # return for in-process use
    return params, pred, co2


if __name__ == "__main__":
    holdout = sys.argv[1] if len(sys.argv) > 1 else "ssp370"
    main(holdout=None if holdout == "none" else holdout)
