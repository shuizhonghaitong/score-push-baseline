"""V8: template-year fields for tasmax/tasmin.

Per cell-year, pick the template model year (hist 1961-2013 + 85 y of each
train ssp) that best matches the secondary index targets once a per-cell level
shift d locks the main index (TXx/TNn) to the quantized T target exactly.
tas (GSL d-grid) and pr (two-stage template) stages are unchanged from V7.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import netCDF4 as nc

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_v6 import (
    V6, fit_targets, build_model, load_F, N, N_DAYS, _run_stats, indices_gsl,
)
from train_v7 import (
    V7, _quant, _med3, VALID_VAR, CAP, ABSFLOOR_TEMP, D_GSL,
    noise_scales, build_model_cached,
)
from climx_common import (
    SUB_COLUMNS, make_submission_parquet, nearest_upsample_lite_to_full, lite_file,
)
from climx_indices import compute_indices, KG_M2_S_TO_MM_DAY
from rnNSE_lite import score_lite
import train_v4 as t4

HE_LEN = 82 * 365  # hist_early covers 1850-1931
C = 12 * 18


def indices_temp2(field: np.ndarray, th2d: np.ndarray, is_max: bool) -> np.ndarray:
    """field (C,365), th2d (C,365) per-cell doy spell threshold.
    Returns (C,4): max case (TXx,SU,ID,WSDI), min case (TNn,TR,FD,CSDI)."""
    out = np.empty((field.shape[0], 4))
    if is_max:
        out[:, 0] = field.max(axis=1)
        out[:, 1] = (field > 298.15).sum(axis=1)
        out[:, 2] = (field < 273.15).sum(axis=1)
        out[:, 3] = _run_stats((field > th2d).astype(bool))[0]
    else:
        out[:, 0] = field.min(axis=1)
        out[:, 1] = (field > 293.15).sum(axis=1)
        out[:, 2] = (field < 273.15).sum(axis=1)
        out[:, 3] = _run_stats((field < th2d).astype(bool))[0]
    return out


def build_templates_cached(train_ssps, hist_start=1961):
    """tasmax/tasmin template years: hist 1961-2013 + first 85 y of each train ssp.
    Returns var -> (ny, 365, 12, 18) f32."""
    cache = Path("models/cache") / f"v8tpl_{'_'.join(train_ssps)}_{hist_start}.npz"
    if cache.exists():
        z = np.load(str(cache))
        print("  templates loaded from cache", flush=True)
        return {v: z[v] for v in ("tasmax", "tasmin")}
    out = {}
    for v in ("tasmax", "tasmin"):
        ds = nc.Dataset(str(lite_file("hist_early")))
        a1 = np.asarray(ds.variables[v][0], dtype=np.float32)
        ds.close()
        ds = nc.Dataset(str(lite_file("hist_late")))
        a2 = np.asarray(ds.variables[v][0], dtype=np.float32)
        ds.close()
        hist = np.concatenate([a1, a2], axis=0)
        years = [hist[(y - 1932) * 365 + HE_LEN:(y - 1932) * 365 + HE_LEN + 365]
                 for y in range(hist_start, 2014)]
        for s in train_ssps:
            ds = nc.Dataset(str(lite_file(s)))
            a = np.asarray(ds.variables[v][0], dtype=np.float32)
            ds.close()
            for yy in range(85):
                years.append(a[yy * 365:(yy + 1) * 365])
        out[v] = np.stack(years, axis=0).astype(np.float32)
        print(f"  templates {v}: {out[v].shape}", flush=True)
    np.savez(str(cache), tasmax=out["tasmax"], tasmin=out["tasmin"])
    print(f"  templates cached to {cache}", flush=True)
    return out


class V8(V7):
    def __init__(self, S, wob_max, wob_min, pr_years, base, th_wsdi, th_csdi,
                 th_r95p, noise=None, templates=None):
        super().__init__(S, wob_max, wob_min, pr_years, base, th_wsdi, th_csdi,
                         th_r95p, noise=noise)
        self.templates = templates

    def _temp_field_t(self, T, var, quant=True):
        is_max = var == "tasmax"
        keys = ["TXx", "SU", "ID", "WSDI"] if is_max else ["TNn", "TR", "FD", "CSDI"]
        th = self.th_wsdi if is_max else self.th_csdi  # (365,12,18)
        tpl = self.templates[var]                       # (ny,365,12,18) f32
        ny = tpl.shape[0]
        thc = th.reshape(365, C).T.copy()               # (216,365)
        thc_rep = np.repeat(thc, ny, axis=0)            # (ny*216,365)
        tpl_tab = tpl.transpose(0, 2, 3, 1).reshape(ny, C, 365).astype(np.float64)  # (ny,216,365)
        base_tab = tpl_tab.reshape(ny * C, 365)
        main_c = (base_tab.max(axis=1) if is_max else base_tab.min(axis=1)).reshape(ny, C)
        tab0 = indices_temp2(base_tab, thc_rep, is_max).reshape(ny, C, 4)
        poolmad = 1.4826 * np.median(np.abs(tab0 - tab0.mean(axis=0)[None]), axis=0)
        sig2 = np.maximum(poolmad[:, 1:] ** 2, 1e-9)    # (216,3)
        Tq = np.stack([(_quant(T[k], ABSFLOOR_TEMP[k]) if quant else T[k]).reshape(N, C)
                       for k in keys], axis=-1)         # (86,216,4)
        b = np.empty((N, C), dtype=np.int64)
        t0 = time.time()
        for t in range(N):
            d = (Tq[t, :, 0][None, :] - main_c).reshape(ny * C)
            tab = indices_temp2(base_tab + d[:, None], thc_rep, is_max).reshape(ny, C, 4)
            e2 = (tab[:, :, 1:] - Tq[t][None, :, 1:]) ** 2
            loss = np.minimum(e2 / sig2[None, :, :], CAP).sum(axis=2)  # (ny,216)
            b[t] = np.argmin(loss, axis=0)
            if t % 20 == 0:
                print(f"      {var} t={t}/{N} {time.time()-t0:.0f}s", flush=True)
        bc = b.copy()
        for c in range(C):
            bc[:, c] = _med3(b[:, c])
        out = np.empty((N_DAYS, 12, 18), np.float32)
        t0 = time.time()
        ar = np.arange(C)
        for t in range(N):
            nn = 364 if t == N - 1 else 365
            selc = bc[t]
            d = (Tq[t, :, 0] - main_c[selc, ar])
            sel = tpl_tab[selc, ar][:, :nn] + d[:, None]
            out[t * 365:t * 365 + nn] = sel.T.reshape(nn, 12, 18).astype(np.float32)
        print(f"      {var} materialize {time.time()-t0:.0f}s", flush=True)
        return out

    def predict_daily(self, T, quant=True):
        out = {}
        t0 = time.time()
        out["tasmax"] = self._temp_field_t(T, "tasmax", quant)
        print(f"  tasmax {time.time()-t0:.0f}s", flush=True)
        t0 = time.time()
        out["tasmin"] = self._temp_field_t(T, "tasmin", quant)
        print(f"  tasmin {time.time()-t0:.0f}s", flush=True)
        t0 = time.time()
        S_v = self.S["tas"]
        flat = np.empty((N_DAYS, 12, 18), dtype=np.float32)
        for i in range(12):
            for j in range(18):
                tab = indices_gsl(S_v[:, i, j][None] + D_GSL[:, None])
                Tg = _quant(T["GSL"][:, i, j], ABSFLOOR_TEMP["GSL"]) if quant else T["GSL"][:, i, j]
                sel_idx = np.argmin((tab[None] - Tg[:, None]) ** 2, axis=1)
                sel = D_GSL[_med3(sel_idx)]
                for t in range(N):
                    nn = 364 if t == N - 1 else 365
                    flat[t * 365:t * 365 + nn, i, j] = (
                        S_v[:nn, i, j] + sel[t]).astype(np.float32)
        out["tas"] = flat
        print(f"  tas {time.time()-t0:.0f}s", flush=True)
        t0 = time.time()
        out["pr"] = self._pr_field(T, quant=quant)
        print(f"  pr {time.time()-t0:.0f}s", flush=True)
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", default="ssp370")
    ap.add_argument("--h", type=float, default=0.15)
    ap.add_argument("--min-n", type=int, default=8)
    ap.add_argument("--smooth", type=float, default=2.0)
    ap.add_argument("--tau-age", type=float, default=0.0)
    ap.add_argument("--hist-start", type=int, default=1961)
    ap.add_argument("--no-quant", action="store_true")
    a = ap.parse_args()

    ho = None if a.holdout == "none" else a.holdout
    train_ssps = [s for s in t4.SSPS if s != ho] if ho else t4.SSPS
    print(f"holdout={ho} train={train_ssps}")
    F = load_F()
    m6 = build_model_cached(train_ssps, hist_start=a.hist_start)
    t0 = time.time()
    ns = noise_scales(train_ssps)
    t0 = time.time()
    tpl = build_templates_cached(train_ssps, hist_start=a.hist_start)
    m = V8(m6.S, m6.wob_max, m6.wob_min, m6.pr_years, m6.base,
           m6.th_wsdi, m6.th_csdi, m6.th_r95p, noise=ns, templates=tpl)
    scen = ho if ho else "test_forcings"
    F_test = F[scen]
    t0 = time.time()
    T = fit_targets(train_ssps, F, F_test, h=a.h, min_n=a.min_n, smooth=a.smooth,
                    tau_age=a.tau_age)
    print(f"  targets {time.time()-t0:.0f}s")
    t0 = time.time()
    daily = m.predict_daily(T, quant=not a.no_quant)
    print(f"  fields {time.time()-t0:.0f}s")
    idx = compute_indices(daily, m.base)
    if ho:
        gt = t4.gt_indices_npz(Path("models/cache"), ho)
        S, per = score_lite(gt, idx)
        print(f"\n== V8 CV holdout={ho}: S={S:+.5f} ==")
        for k in SUB_COLUMNS:
            print(f"  {k:8s} {per[k]:+.4f}")
    else:
        outd = Path("submissions")
        outd.mkdir(exist_ok=True)
        idx_full = {k: nearest_upsample_lite_to_full(idx[k].astype(np.float32))
                    for k in SUB_COLUMNS}
        p = make_submission_parquet(idx_full, outd / "submission_v8.parquet")
        print("wrote", p)


if __name__ == "__main__":
    main()
