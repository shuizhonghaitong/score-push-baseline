"""V2 forcing-response emulator with stabilized slopes.

Variants:
  slope='ann'   : per (cell,var) slope from annual-mean regression (shared over doy)
  slope='doy'   : per (cell,var,doy) slope (as v1)
  smooth sigma  : Gaussian spatial smoothing of slopes (in lite cells)
  hist_start    : first hist year used
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import netCDF4 as nc

sys.path.insert(0, str(Path(__file__).resolve().parent))
from climx_common import TARGET_VARS, lite_file
from train_v1 import co2_yearly, load_forcings_co2, _load_var


def gaussian_kernel(sigma: float, size: int = 9) -> np.ndarray:
    ax = np.arange(size) - size // 2
    k = np.exp(-(ax ** 2) / (2 * sigma ** 2))
    k2 = np.outer(k, k)
    return k2 / k2.sum()


def smooth2d(a: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return a
    k = gaussian_kernel(sigma)
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(a, sigma=sigma, mode="nearest")


def fit(holdout: str | None = "ssp370", hist_start: int = 1961,
        slope: str = "ann", smooth_sigma: float = 0.0,
        pr_log: bool = False, out_prefix: str = "models/v2"):
    t0 = time.time()
    run = Path(__file__).resolve().parents[1]
    outdir = run / out_prefix
    outdir.mkdir(parents=True, exist_ok=True)

    co2 = load_forcings_co2()
    co2_ref = co2["ssp126"][0][0]
    F = {s: np.log(c / co2_ref) for s, (c, y0) in co2.items()
         if s not in ("hist_early", "hist_late")}

    samples = []
    hist_y0 = co2["hist"][1]
    for y in range(hist_start, 2014):
        samples.append(("hist", y - hist_y0))
    train_ssps = [s for s in ["ssp126", "ssp370", "ssp585"] if s != holdout]
    for s in train_ssps:
        for y in range(85):
            samples.append((s, y))
    Fs = np.array([F[s][yi] for s, yi in samples])
    n = len(samples)
    print(f"samples={n} slope={slope} smooth={smooth_sigma} pr_log={pr_log}")

    params = {}
    for var in TARGET_VARS:
        h = np.concatenate([_load_var("hist_early", var), _load_var("hist_late", var)], axis=0)
        ssps = {s: _load_var(s, var) for s in train_ssps}
        if slope == "ann":
            # annual means: (n, lat, lon)
            A = np.empty((n, 12, 18), dtype=np.float64)
            for si, (s, yi) in enumerate(samples):
                v = h if s == "hist" else ssps[s]
                A[si] = v[yi * 365: yi * 365 + 365].mean(axis=0)
            if pr_log:
                A = np.log1p(A * 86400)
            Fc = Fs - Fs.mean()
            varF = (Fc ** 2).sum()
            m = (A * Fc[:, None, None]).sum(axis=0) / varF
            b = A.mean(axis=0) - m * Fs.mean()
            # seasonal climatology per doy from all data
            Y = np.empty((n, 365, 12, 18), dtype=np.float32)
            for si, (s, yi) in enumerate(samples):
                v = h if s == "hist" else ssps[s]
                Y[si] = v[yi * 365: yi * 365 + 365]
            b_doy = Y.mean(axis=0).astype(np.float64)  # (365,12,18) raw climatology
            Fbar = Fs.mean()
            m_arr = m
            b_arr = b_doy  # prediction: b_doy + m*(F - Fbar)
            if pr_log:
                pass
            if smooth_sigma > 0:
                m_arr = smooth2d(m_arr, smooth_sigma)
            params[var] = dict(kind="ann", m=m_arr, b_ann=b, b_doy=b_arr, Fbar=Fbar, pr_log=pr_log)
        else:  # doy
            Y = np.empty((n, 365, 12, 18), dtype=np.float64)
            for si, (s, yi) in enumerate(samples):
                v = h if s == "hist" else ssps[s]
                Y[si] = v[yi * 365: yi * 365 + 365]
            Fc = Fs - Fs.mean()
            varF = (Fc ** 2).sum()
            m = (Y * Fc[:, None, None, None]).sum(axis=0) / varF
            b = Y.mean(axis=0) - m * Fs.mean()
            if smooth_sigma > 0:
                m = smooth2d(m, smooth_sigma)
            params[var] = dict(kind="doy", m=m, b=b, pr_log=pr_log)
        print(f"  {var}: {time.time()-t0:.0f}s m[{m.min():.4g},{m.max():.4g}]")
        del h, ssps
        if slope == "ann":
            del A, Y
        else:
            del Y

    # predictions
    pred = {}
    for s in ([holdout] if holdout else []) + ["ssp245"]:
        Fs_s = F[s]
        flat = {}
        for var in TARGET_VARS:
            P = params[var]
            if P["kind"] == "ann":
                off = P["b_doy"][None, :, :, :] + P["m"][None, :, :] * (Fs_s - P["Fbar"])[:, None, None, None]  # (86,365,12,18)
                if P["pr_log"]:
                    # b_doy was computed in log space; m in log space; convert
                    arr_log = off
                    arr = np.expm1(arr_log / 86400.0)
                else:
                    arr = off
            else:
                arr = P["b"] + P["m"][None, :, :, :] * Fs_s[:, None, None, None]
            out = np.empty((31389, 12, 18), dtype=np.float32)
            i = 0
            for y in range(86):
                nn = 364 if y == 85 else 365
                out[i:i+nn] = arr[y][:nn]
                i += nn
            flat[var] = out
        pred[s] = flat

    tag = f"ho_{holdout}" if holdout else "full"
    tag += f"_{slope}_sm{smooth_sigma}" + ("_logpr" if pr_log else "")
    d = {"F_hist": F["hist"], "F_ssp126": F["ssp126"], "F_ssp370": F["ssp370"],
         "F_ssp585": F["ssp585"], "F_ssp245": F["ssp245"], "co2_ref": co2_ref, "slope": slope}
    for var in TARGET_VARS:
        d[var + "_m"] = params[var]["m"]
        if params[var]["kind"] == "ann":
            d[var + "_b_doy"] = params[var]["b_doy"]
        else:
            d[var + "_b"] = params[var]["b"]
    np.savez_compressed(outdir / f"params_v2_{tag}.npz", **d)
    for s in pred:
        np.savez_compressed(outdir / f"pred_v2_{s}_{tag}.npz", **pred[s])
    print(f"saved {tag} ({time.time()-t0:.0f}s)")
    return tag


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", default="ssp370")
    ap.add_argument("--hist-start", type=int, default=1961)
    ap.add_argument("--slope", default="ann", choices=["ann", "doy"])
    ap.add_argument("--smooth", type=float, default=1.5)
    ap.add_argument("--pr-log", action="store_true")
    a = ap.parse_args()
    fit(holdout=None if a.holdout == "none" else a.holdout, hist_start=a.hist_start,
        slope=a.slope, smooth_sigma=a.smooth, pr_log=a.pr_log)
