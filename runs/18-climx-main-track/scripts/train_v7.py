"""V7: materialize the F-target curves T as daily fields with minimal extra
variance/bias (each field index ≈ T within grid resolution).

- tasmax/tasmin: per-year d locked to the main index (TXx/TNn) exactly via the
  true response g(eps) = max/min_doy(S + eps*wob); eps searched on a fine grid,
  then a local (dd, eps) refinement.  Index weights = 1/(varT+floor) and 0 for
  unscorable cells (varT <= 1e-6, mirroring score_lite's valid mask).
- tas: per-year d matching GSL (grid 0.1 K), median-filtered.
- pr: fixed per-cell template year (matched to the T mean, R95p band filter),
  then per-year two-stage (a, L0) refinement over a wide range.
- Per-cell target quantization kills config-flip jitter on flat cells.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import faulthandler
faulthandler.enable()
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_v6 import (
    V6, fit_targets, build_model, load_F,
    N, N_DAYS,
    indices_temp, indices_gsl, indices_pr,
)
from climx_common import SUB_COLUMNS, make_submission_parquet, nearest_upsample_lite_to_full
from climx_indices import compute_indices, KG_M2_S_TO_MM_DAY
from rnNSE_lite import score_lite
import train_v4 as t4

D_GSL = np.arange(-8.0, 16.01, 0.1)
EPS_G = np.arange(0.0, 8.0 + 1e-9, 0.05)
EPS_T = np.arange(0.0, 8.0 + 1e-9, 0.25)
QUANT_FRAC = 0.05
VALID_VAR = 1e-6
CAP = 2.0

ABSFLOOR_TEMP = {"TXx": 0.05, "TNn": 0.05, "SU": 1.0, "TR": 1.0, "FD": 1.0,
                 "ID": 1.0, "WSDI": 1.0, "CSDI": 1.0, "GSL": 1.0}
ABSFLOOR_PR = {"R95p": 20.0, "Rx5day": 2.0, "CDD": 1.0, "CWD": 1.0,
               "R10mm": 0.2, "SDII": 0.01}


def _med3(x: np.ndarray) -> np.ndarray:
    n = len(x)
    y = x.astype(float).copy()
    if n < 3:
        return y.astype(int)
    y[1:-1] = np.median(np.stack([x[:-2].astype(float), x[1:-1].astype(float), x[2:].astype(float)]), axis=0)
    y[0] = np.median(x[0:2].astype(float))
    y[-1] = np.median(x[-2:].astype(float))
    return y.astype(int)


def _quant(Tk: np.ndarray, absfloor: float) -> np.ndarray:
    """Per-cell quantization of the (N, ...) target curve (shape preserving)."""
    q = QUANT_FRAC * Tk.std(axis=0, keepdims=True) + absfloor
    return (Tk / q).round() * q


def _score_weights(T, absfloor) -> dict:
    """w_k(cell) = 1/(varT+floor) if varT > VALID_VAR else 0."""
    out = {}
    for k, af in absfloor.items():
        var = T[k].var(axis=0)
        floor = np.maximum(af, 0.05 * float(np.quantile(var[var > 0], 0.75)) if (var > 0).any() else af)
        w = np.where(var > VALID_VAR, 1.0 / (var + floor), 0.0)
        out[k] = w
    return out


def noise_scales(train_ssps, cache=Path("models/cache")):
    """Per-cell per-index noise variance: median over train SSPs of var(index -
    smooth(index)). Proxy for the GT variance component not captured by T."""
    from scipy.ndimage import gaussian_filter
    pcache = Path("models/cache") / f"nois_{'_'.join(train_ssps)}.npz"
    if pcache.exists():
        z = np.load(str(pcache))
        return {k: z[k] for k in SUB_COLUMNS}
    out = {}
    for k in SUB_COLUMNS:
        sigs = []
        for s in train_ssps:
            I = t4.gt_indices_npz(cache, s)[k]  # (86,12,18)
            sm = gaussian_filter(I, (3, 0.01, 0.01), mode="nearest")
            sigs.append((I - sm).var(axis=0))
        out[k] = np.median(np.stack(sigs), axis=0)
    np.savez(str(pcache), **out)
    return out


class V7(V6):
    # ------------------------------------------------------------------
    def _temp_field(self, T, S_v, wob, th_wsdi, th_csdi, keys, quant=True):
        """Two-amplitude wobble field: S + d + ep*max(w,0) + ec*min(w,0).
        d is locked so the main index (TXx/TNn) matches Tq exactly for every
        (ep, ec); (ep, ec) is searched to match the three secondary indices."""
        is_max = th_wsdi is not None
        E = EPS_T
        ne = len(E)
        ns = self.noise
        wT = {}
        for k in keys:
            var = T[k].var(axis=0)
            wT[k] = np.where(var > VALID_VAR, 1.0 / (var + ns[k]), 0.0)
        flat = np.empty((N_DAYS, 12, 18), dtype=np.float32)
        for i in range(12):
            for j in range(18):
                Sc = S_v[:, i, j]
                wc = wob[:, i, j]
                th_ = th_wsdi[:, i, j] if is_max else th_csdi[:, i, j]
                wp = np.maximum(wc, 0.0)
                wn = np.minimum(wc, 0.0)
                Tq = np.stack([(_quant(T[k][:, i, j], ABSFLOOR_TEMP[k]) if quant
                                else T[k][:, i, j]) for k in keys], axis=-1)  # (N,nk)
                Tm = Tq[:, 0]
                base_c = (Sc[None, None, :] + E[:, None, None] * wp[None, None, :]
                          + E[None, :, None] * wn[None, None, :])  # (ne,ne,365)
                M = base_c.max(axis=2) if is_max else base_c.min(axis=2)  # (ne,ne)
                tab0 = indices_temp(base_c.reshape(ne * ne, 365),
                                    th_ if is_max else None, None if is_max else th_)
                # family scale of each secondary index over the (ep,ec) grid
                sig2 = np.maximum(tab0[:, 1:].var(axis=0), 1e-12 * np.maximum(Tq[:, 1:].mean(axis=0), 1e-6) ** 2)
                b = np.empty(N, dtype=int)
                for t in range(N):
                    fields = base_c + (Tm[t] - M)[:, :, None]
                    tab = indices_temp(fields.reshape(ne * ne, 365),
                                       th_ if is_max else None, None if is_max else th_)
                    e2 = (tab[:, 1:] - Tq[t, 1:][None, :]) ** 2
                    loss = np.minimum(e2 / sig2[None, :], CAP).sum(axis=1)
                    b[t] = int(np.argmin(loss))
                b = _med3(b)
                epi = (b // ne).astype(int)
                eci = (b % ne).astype(int)
                for t in range(N):
                    nn = 364 if t == N - 1 else 365
                    d = Tm[t] - M[epi[t], eci[t]]
                    field = Sc + d + E[epi[t]] * wp + E[eci[t]] * wn
                    flat[t * 365:t * 365 + nn, i, j] = field[:nn].astype(np.float32)
        return flat

    # ------------------------------------------------------------------
    def __init__(self, S, wob_max, wob_min, pr_years, base, th_wsdi, th_csdi, th_r95p, noise=None):
        super().__init__(S, wob_max, wob_min, pr_years, base, th_wsdi, th_csdi, th_r95p)
        self.noise = noise

    def predict_daily(self, T, quant=True):
        out = {}
        t0 = time.time()
        out["tasmax"] = self._temp_field(T, self.S["tasmax"], self.wob_max,
                                         self.th_wsdi, None,
                                         ["TXx", "SU", "ID", "WSDI"], quant)
        print(f"  tasmax {time.time() - t0:.0f}s", flush=True)
        t0 = time.time()
        out["tasmin"] = self._temp_field(T, self.S["tasmin"], self.wob_min,
                                         None, self.th_csdi,
                                         ["TNn", "TR", "FD", "CSDI"], quant)
        print(f"  tasmin {time.time() - t0:.0f}s", flush=True)
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
        print(f"  tas {time.time() - t0:.0f}s", flush=True)
        t0 = time.time()
        out["pr"] = self._pr_field(T, quant=quant)
        print(f"  pr {time.time() - t0:.0f}s", flush=True)
        return out

    # ------------------------------------------------------------------
    def _pr_field(self, T, quant=True):
        keys = ["R95p", "CDD", "CWD", "Rx5day", "R10mm", "SDII"]
        ny = self.pr_years.shape[0]
        tab = np.empty((len(self.PR_A_GRID) * ny, 216, 6))
        t0 = time.time()
        for x, a in enumerate(self.PR_A_GRID):
            for y0 in range(ny):
                f = np.maximum(self.pr_years[y0].astype(np.float64) * a, 0.0)
                r = indices_pr(f.transpose(2, 0, 1).reshape(216, 365),
                               self.th_r95p.reshape(216, 365))
                tab[x * ny + y0] = r[:, [3, 1, 2, 0, 4, 5]]  # keys order
        print(f"    pr template table {time.time() - t0:.0f}s", flush=True)

        poolmad = 1.4826 * np.median(np.abs(tab - tab.mean(axis=0)[None]), axis=0)  # (216,6)
        poolmad = np.maximum(poolmad, 1e-3)
        sig2 = poolmad ** 2
        varT = np.stack([T[k].var(axis=0).reshape(-1) for k in keys], axis=-1).reshape(216, 6)
        active = (varT > VALID_VAR).any(axis=1)
        active = np.where(active, 1.0, 0.0).astype(bool)

        T95 = T["R95p"].mean(axis=0).reshape(216)
        r95 = tab[:, :, 0]
        mask = (r95 >= 0.5 * T95[None]) & (r95 <= 2.0 * T95[None])
        nkeep = mask.sum(axis=0)
        loose = (r95 >= 0.25 * T95[None]) & (r95 <= 4.0 * T95[None])
        mask = np.where(nkeep[None] >= 16, mask, loose)
        mask = np.where((T95 > 5.0)[None], mask, True)
        print(f"    pr band filter kept median {int(np.median(nkeep))}/{tab.shape[0]} templates", flush=True)

        ci = np.arange(12)[:, None].repeat(18, axis=1).reshape(216)
        cj = np.arange(18)[None, :].repeat(12, axis=0).reshape(216)
        flat = (np.arange(ny)[:, None] * 365 + np.arange(365)[None, :]) * 216
        flat = flat[:, :, None] + ci[None, None, :] * 18 + cj[None, None, :]
        py_flat = self.pr_years.reshape(-1)
        thc = self.th_r95p.reshape(216, 365)
        ar = np.arange(365)[None, :]
        cc = np.arange(216)[:, None]

        # stage 1: fixed per-cell template (match T mean)
        Tbar = np.stack([T[k].mean(axis=0).reshape(-1) for k in keys], axis=-1).reshape(216, 6)
        e2 = (tab - Tbar[None]) ** 2
        lossbar = np.minimum(e2 / sig2[None], CAP).sum(axis=2)
        lossbar = np.where(mask, lossbar, np.inf)
        b0 = lossbar.argmin(axis=0)
        ia0 = (b0 // ny).astype(int)
        iy0 = (b0 % ny).astype(int)
        ia0 = np.where(active, ia0, 3)  # a0 = 1.0 for inactive cells
        iy0 = np.where(active, iy0, 0)
        a0 = np.asarray(self.PR_A_GRID)[ia0]  # (216,)
        fbase = py_flat[flat[iy0[:, None], ar, cc]] * a0[:, None]  # (216, 365)

        # two-stage (a, L0) grids relative to fbase (which already carries a0)
        ac = np.arange(0.6, 1.6 + 1e-9, 0.1)      # 11
        Lc = np.arange(-4.0, 4.0 + 1e-9, 1.0) / KG_M2_S_TO_MM_DAY  # 9
        af = np.arange(-0.10, 0.10 + 1e-9, 0.02)  # 11
        Lf = np.arange(-1.0, 1.0 + 1e-9, 0.2) / KG_M2_S_TO_MM_DAY  # 11
        nAC, nAF = len(ac) * len(Lc), len(af) * len(Lf)
        thc_ac = np.repeat(thc, nAC, axis=0)
        thc_af = np.repeat(thc, nAF, axis=0)

        # constant per-cell target: T's annual part is uncorrelated noise for pr;
        # a constant at T-mean is the high-scoring prediction.
        tgtv = Tbar  # (216, 6)
        # coarse (a, L0)
        fields = (fbase[:, None, None, :] * ac[None, :, None, None]
                  + Lc[None, None, :, None])  # (216, 11, 9, 365)
        fields = np.maximum(fields.reshape(216 * nAC, 365), 0.0)
        r = indices_pr(fields, thc_ac).reshape(216, nAC, 6)
        e2 = (r - tgtv[:, None, :]) ** 2
        loss = np.minimum(e2 / sig2[:, None, :], CAP).sum(axis=2)
        b = loss.argmin(axis=1)
        ia_c = (b // len(Lc)).astype(int)
        iL_c = (b % len(Lc)).astype(int)
        a_f = ac[ia_c][:, None] + af[None, :]
        L_f = Lc[iL_c][:, None] + Lf[None, :]
        fields = (fbase[:, None, None, :] * a_f[:, :, None, None]
                  + L_f[:, None, :, None])
        fields = np.maximum(fields.reshape(216 * nAF, 365), 0.0)
        r = indices_pr(fields, thc_af).reshape(216, nAF, 6)
        e2 = (r - tgtv[:, None, :]) ** 2
        loss = np.minimum(e2 / sig2[:, None, :], CAP).sum(axis=2)
        b2 = loss.argmin(axis=1)
        ia2 = (b2 // len(Lf)).astype(int)
        iL2 = (b2 % len(Lf)).astype(int)
        a_sel = a_f[np.arange(216), ia2]
        L0_sel = L_f[np.arange(216), iL2]
        a_sel = np.where(active, a_sel, 1.0)
        L0_sel = np.where(active, L0_sel, 0.0)
        year_f = np.maximum(fbase * a_sel[:, None] + L0_sel[:, None], 0.0)  # (216, 365)
        print(f"    pr (a,L0) fit {time.time() - t0:.0f}s", flush=True)
        out = np.empty((N_DAYS, 12, 18), dtype=np.float32)
        t0 = time.time()
        for t in range(N):
            nn = 364 if t == N - 1 else 365
            out[t * 365:t * 365 + nn] = year_f[:, :nn].T.reshape(nn, 12, 18).astype(np.float32)
        print(f"    pr refinement {time.time() - t0:.0f}s", flush=True)
        return out


def build_model_cached(train_ssps, hist_start=1961):
    cache = Path("models/cache") / f"v7model_{'_'.join(train_ssps)}_{hist_start}.npz"
    if cache.exists():
        z = np.load(str(cache))
        S = {v: z["S_" + v] for v in t4.VARS}
        base = {v: z["base_" + v] for v in t4.VARS}
        m = V6(S, z["wob_max"], z["wob_min"], z["pr_years"], base,
               z["th_wsdi"], z["th_csdi"], z["th_r95p"])
        print("  model loaded from cache", flush=True)
        return m
    m = build_model(train_ssps, hist_start=hist_start)
    z = {"wob_max": m.wob_max, "wob_min": m.wob_min, "pr_years": m.pr_years,
         "th_wsdi": m.th_wsdi, "th_csdi": m.th_csdi, "th_r95p": m.th_r95p}
    for v in t4.VARS:
        z["S_" + v] = m.S[v]
        z["base_" + v] = m.base[v].astype(np.float32)
    np.savez(str(cache), **z)
    print(f"  model cached to {cache}", flush=True)
    return m


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
    print(f"  noise scales {time.time() - t0:.0f}s", flush=True)
    m = V7(m6.S, m6.wob_max, m6.wob_min, m6.pr_years, m6.base,
           m6.th_wsdi, m6.th_csdi, m6.th_r95p, noise=ns)
    scen = ho if ho else "test_forcings"
    F_test = F[scen]
    t0 = time.time()
    T = fit_targets(train_ssps, F, F_test, h=a.h, min_n=a.min_n, smooth=a.smooth, tau_age=a.tau_age)
    print(f"  targets {time.time() - t0:.0f}s")
    t0 = time.time()
    daily = m.predict_daily(T, quant=not a.no_quant)
    print(f"  fields {time.time() - t0:.0f}s")
    idx = compute_indices(daily, m.base)
    if ho:
        gt = t4.gt_indices_npz(Path("models/cache"), ho)
        S, per = score_lite(gt, idx)
        print(f"\n== V7 CV holdout={ho}: S={S:+.5f} ==")
        for k in SUB_COLUMNS:
            print(f"  {k:8s} {per[k]:+.4f}")
    else:
        outd = Path("submissions")
        outd.mkdir(exist_ok=True)
        idx_full = {k: nearest_upsample_lite_to_full(idx[k].astype(np.float32)) for k in SUB_COLUMNS}
        p = make_submission_parquet(idx_full, outd / "submission_v7.parquet")
        print("wrote", p)


if __name__ == "__main__":
    main()
