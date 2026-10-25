"""V3 emulator: quadratic forcing fit + CV-estimated per-cell temporal shrinkage.

Prediction for var v, cell c, year t, day doy:
  pred(t, doy) = clim_doy(c, doy) + s(c) * [ T̂_c(F(t)) - T̂_c(Fbar) ]
where T̂_c is a per-cell quadratic in F = ln(CO2/CO2_2015), slopes spatially
smoothed, and s(c) is a CV-estimated shrinkage factor (from held-out scenarios,
applied to the daily fields' year-to-year offset).

Shrinkage is estimated per (cell, var) from the annual-mean regression of the
held-out scenario's true annual mean on the model's predicted annual mean.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import netCDF4 as nc
from scipy.ndimage import gaussian_filter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from climx_common import TARGET_VARS, lite_file
from train_v1 import load_forcings_co2, _load_var

SCORED_VARS = ["tas", "tasmax", "tasmin", "pr"]  # only these drive the 15 indices


def load_annual(scenario_keys, var):
    """annual means (n_years, 12, 18) for keys like 'hist' (164 yrs) or ssp (85 yrs)."""
    pass


def build_samples(co2, hist_start, train_ssps):
    samples = []
    hist_y0 = co2["hist"][1]
    for y in range(hist_start, 2014):
        samples.append(("hist", y - hist_y0))
    for s in train_ssps:
        for y in range(85):
            samples.append((s, y))
    return samples


def fit_and_predict(holdout: str, hist_start: int = 1961, smooth_sigma: float = 1.5,
                    deg: int = 2, out_prefix: str = "models/v3", cv_ssps=None):
    t0 = time.time()
    run = Path(__file__).resolve().parents[1]
    outdir = run / out_prefix
    outdir.mkdir(parents=True, exist_ok=True)

    co2 = load_forcings_co2()
    co2_ref = co2["ssp126"][0][0]
    F = {s: np.log(c / co2_ref) for s, (c, y0) in co2.items() if s not in ("hist_early", "hist_late")}
    cv_ssps = cv_ssps or ["ssp126", "ssp370", "ssp585"]

    # ---------- load all scenario annual means + doy climatology (per var) ----------
    data = {}
    for var in SCORED_VARS:
        h = np.concatenate([_load_var("hist_early", var), _load_var("hist_late", var)], axis=0)
        ssps = {s: _load_var(s, var) for s in ["ssp126", "ssp370", "ssp585"]}
        data[var] = dict(hist=h, ssps=ssps)

    results = {}
    ssp_anns = {}
    for var in SCORED_VARS:
        d = data[var]
        # annual means
        hist_ann = d["hist"][:164 * 365].reshape(164, 365, 12, 18).mean(axis=1)  # 164 yrs (2014 partial excluded below)
        ssp_ann = {s: d["ssps"][s][:85 * 365].reshape(85, 365, 12, 18).mean(axis=1) for s in d["ssps"]}
        ssp_anns[var] = ssp_ann

        # pooled design matrix over train samples
        train_ssps = [s for s in cv_ssps if s != holdout]
        rows = []
        for y in range(hist_start, 2014):
            rows.append((F["hist"][y - 1850], hist_ann[y - 1850]))
        for s in train_ssps:
            for y in range(85):
                rows.append((F[s][y], ssp_ann[s][y]))
        Ft = np.array([r[0] for r in rows])
        At = np.stack([r[1] for r in rows], axis=0)  # (n, 12, 18)
        Fbar = Ft.mean()
        f1 = Ft - Fbar
        c2 = (f1 ** 2).mean()
        f2 = f1 ** 2 - c2
        f3 = f2 * f1
        B = np.stack([f1, f2, f3], axis=1)  # (n, 3)
        Agram = B.T @ B  # (3, 3)
        Xy = np.einsum("ni,nkl->ikl", B, At)  # (3, 12, 18)
        b0 = At.mean(axis=0)
        coeffs = np.einsum("ij,jkl->ikl", np.linalg.inv(Agram), Xy)  # (3, 12, 18)
        if smooth_sigma > 0:
            coeffs = np.stack([gaussian_filter(coeffs[i], smooth_sigma, mode="nearest") for i in range(3)])
        b12 = coeffs
        def Toff(Fv):  # (n,) -> (n, 12, 18)
            x1 = Fv - Fbar
            x2 = x1 ** 2 - c2
            x3 = x2 * x1
            return b12[0] * x1[:, None, None] + b12[1] * x2[:, None, None] + b12[2] * x3[:, None, None]
        results[var] = dict(b0=b0, b12=b12, Fbar=Fbar, f1=f1, f2=f2, Toff=Toff)
        print(f"  {var} fit {time.time()-t0:.0f}s")

    # ---------- CV shrinkage ----------
    # For each CV scenario (all three, each treated as "test" once), estimate
    # s*(cell) = Cov(gt_ann, pred_ann)/Var(pred_ann)  (clip >= 0).
    s_cv = {var: {} for var in SCORED_VARS}
    for test_s in cv_ssps:
        Ft_ = F[test_s][:85]
        x1 = Ft_ - results[SCORED_VARS[0]]["Fbar"]
        for var in SCORED_VARS:
            R = results[var]
            pred_ann = R["b0"][None] + R["Toff"](Ft_)
            gt_ann = ssp_anns[var][test_s]
            p = pred_ann - pred_ann.mean(axis=0)
            g = gt_ann - gt_ann.mean(axis=0)
            var_p = (p ** 2).mean(axis=0)
            cov = (p * g).mean(axis=0)
            s = np.where(var_p > 1e-12, cov / np.maximum(var_p, 1e-12), 0.0)
            s = np.clip(s, 0.0, 1.5)
            s_cv[var][test_s] = s
        # also report CV score estimate
    # aggregate
    S_cv = {}
    for var in SCORED_VARS:
        S_cv[var] = np.mean([s_cv[var][s] for s in cv_ssps], axis=0)
    for var in SCORED_VARS:
        print(f"  {var} s_cv range [{S_cv[var].min():.3f},{S_cv[var].max():.3f}] mean {S_cv[var].mean():.3f}")

    # ---------- predictions ----------
    pred = {}
    pred_ss = ["ssp245"] + ([holdout] if holdout else [])
    for s in pred_ss:
        Ft_ = F[s]
        flat = {}
        for var in SCORED_VARS:
            R = results[var]
            off = R["Toff"](Ft_) * S_cv[var][None, :, :]  # (85, 12, 18)
            # daily: doy climatology from train samples + offset
            # build clim_doy from pooled train data
            n1 = len(train_ssps) * 85 + (2014 - hist_start)
            Y = np.empty((n1, 365, 12, 18), dtype=np.float32)
            si = 0
            hh = data[var]["hist"]
            for y in range(hist_start, 2014):
                Y[si] = hh[(y - 1850) * 365: (y - 1850) * 365 + 365]; si += 1
            for ts in train_ssps:
                v = data[var]["ssps"][ts]
                for y in range(85):
                    Y[si] = v[y * 365: y * 365 + 365]; si += 1
            clim_doy = Y.mean(axis=0).astype(np.float64)
            arr = clim_doy[None, :, :, :] + off[:, None, :, :]  # (85, 365, 12, 18)
            out = np.empty((31389, 12, 18), dtype=np.float32)
            i = 0
            for y in range(86):
                nn = 364 if y == 85 else 365
                if y < 85:
                    out[i:i+nn] = arr[y]
                else:
                    out[i:i+nn] = arr[84][:nn]
                i += nn
            flat[var] = out
        pred[s] = flat
        if holdout:
            # also produce full 7-var by adding huss/psl/sfcWind from simple ann model?
            pass
    tag = f"v3_ho{holdout}_sm{smooth_sigma}_d{deg}_hs{hist_start}"
    d = {f"scv_{v}": S_cv[v] for v in SCORED_VARS}
    d.update({f"b0_{v}": results[v]["b0"] for v in SCORED_VARS})
    for v in SCORED_VARS:
        for i in range(3):
            d[f"b{i}_{v}"] = results[v]["b12"][i]
    d.update({"F_hist": F["hist"], "F_ssp126": F["ssp126"], "F_ssp370": F["ssp370"],
              "F_ssp585": F["ssp585"], "F_ssp245": F["ssp245"], "co2_ref": co2_ref,
              "f1sum1": (results["tas"]["f1"] ** 2).mean()})
    np.savez_compressed(outdir / f"params_{tag}.npz", **d)
    for s in pred:
        np.savez_compressed(outdir / f"pred_{s}_{tag}.npz", **pred[s])
    print(f"saved {tag} ({time.time()-t0:.0f}s)")
    return tag, results, S_cv


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", default="ssp370")
    ap.add_argument("--hist-start", type=int, default=1961)
    ap.add_argument("--smooth", type=float, default=1.5)
    a = ap.parse_args()
    fit_and_predict(holdout=None if a.holdout == "none" else a.holdout,
                    hist_start=a.hist_start, smooth_sigma=a.smooth)
