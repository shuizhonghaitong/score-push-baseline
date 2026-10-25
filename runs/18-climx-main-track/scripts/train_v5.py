"""V5 emulator: V4 trend model + per-cell daily-field calibration.

The official score is per-cell temporal R^2 of the 15 annual indices. The
noise-free daily field's indices are biased relative to the noisy GT indices
(Jensen gap), so we calibrate the *daily fields* per cell (allowed: ops on
daily fields, indices always via official code):

- tasmax: pred = S + u_max*m_max*(F-Fbar) + d_max   (matches the GT TXx line)
- tasmin: pred = S + u_min*m_min*(F-Fbar) + d_min   (matches the GT TNn line)
- tas:    pred = S + m_tas*(F-Fbar)                 (no closed form; GSL ok)
- pr:     pred = S_pr + event pattern (W days x A mm, uniformly spaced),
          (W, A) chosen per cell to match the 6 pr index targets.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import netCDF4 as nc
from scipy.ndimage import gaussian_filter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from climx_common import lite_file, SUB_COLUMNS
from climx_indices import (compute_indices, base_period_slice, percentile_doy_window,
                           KG_M2_S_TO_MM_DAY, longest_run_annual)
from rnNSE_lite import score_lite
import train_v4 as t4

VARS = t4.VARS
PR_IDX = ["Rx5day", "CDD", "CWD", "R95p", "R10mm", "SDII"]
# per-index loss scales (typical std of the targets)
PR_SCALE = {"Rx5day": 20.0, "CDD": 20.0, "CWD": 15.0, "R95p": 150.0,
            "R10mm": 5.0, "SDII": 0.8}
MAX_TGT = "TXx"
MIN_TGT = "TNn"


def fit_line(I_pool: np.ndarray, F_pool: np.ndarray):
    Fc = F_pool - F_pool.mean()
    varF = (Fc ** 2).sum()
    b = (I_pool * Fc[:, None]).sum(axis=0) / varF
    a = I_pool.mean(axis=0) - b * F_pool.mean()
    return a, b


def var_indices_year(v: str, field: np.ndarray, ths: dict) -> dict:
    """Annual indices for one synthetic 365-day year (365,12,18)."""
    out = {}
    if v == "tasmax":
        out["TXx"] = field.max(axis=0)
        out["SU"] = (field > 298.15).sum(axis=0).astype(np.float64)
        mask = field > ths["tasmax"]
        out["WSDI"] = windowed_run_sum(mask, 6)
    elif v == "tasmin":
        out["TNn"] = field.min(axis=0)
        out["TR"] = (field > 293.15).sum(axis=0).astype(np.float64)
        out["FD"] = (field < 273.15).sum(axis=0).astype(np.float64)
        mask = field < ths["tasmin"]
        out["CSDI"] = windowed_run_sum(mask, 6)
    elif v == "tas":
        out["GSL"] = growing_season_length(field)
    elif v == "pr":
        pr_mm = field * KG_M2_S_TO_MM_DAY
        csum = np.cumsum(pr_mm, axis=0)
        pr5 = np.full_like(pr_mm, np.nan)
        pr5[4] = csum[4]
        pr5[5:] = csum[5:] - csum[:-5]
        pr_lt1 = pr_mm < 1.0
        pr_ge1 = pr_mm >= 1.0
        pr_ge10 = pr_mm >= 10.0
        Cflat = int(np.prod(field.shape[1:]))
        out["Rx5day"] = np.nanmax(pr5, axis=0)
        out["CDD"] = longest_run(pr_lt1.reshape(365, Cflat)).reshape(field.shape[1:])
        out["CWD"] = longest_run(pr_ge1.reshape(365, Cflat)).reshape(field.shape[1:])
        out["R95p"] = np.where(pr_mm > ths["pr"], pr_mm, 0.0).sum(axis=0)
        out["R10mm"] = pr_ge10.sum(axis=0).astype(np.float64)
        ws = np.where(pr_ge1, pr_mm, 0.0).sum(axis=0)
        wc = pr_ge1.sum(axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            out["SDII"] = np.where(wc == 0, 0.0, ws / wc)
    return out


def windowed_run_sum(mask: np.ndarray, min_len: int) -> np.ndarray:
    """mask (365, 12, 18) -> total days in runs >= min_len per cell."""
    Cflat = int(np.prod(mask.shape[1:]))
    m = mask.reshape(365, Cflat).astype(bool)
    T = m.shape[0]
    pos = np.arange(T)[:, None]
    prev = np.empty_like(m)
    prev[0] = False
    prev[1:] = m[:-1]
    rs = m & ~prev
    starts = np.where(rs, pos, 0)
    last_start = np.maximum.accumulate(starts, axis=0)
    runlen = pos - last_start + 1
    nxt = np.concatenate([m[1:], np.zeros((1, m.shape[1]), dtype=bool)], axis=0)
    is_end = m & ~nxt
    total = np.where(is_end & (runlen >= min_len), runlen, 0).sum(axis=0)
    return total.reshape(mask.shape[1:])


def longest_run(m: np.ndarray) -> np.ndarray:
    T = m.shape[0]
    pos = np.arange(T)[:, None]
    prev = np.empty_like(m)
    prev[0] = False
    prev[1:] = m[:-1]
    rs = m & ~prev
    starts = np.where(rs, pos, 0)
    last_start = np.maximum.accumulate(starts, axis=0)
    runlen = np.where(m, pos - last_start + 1, 0)
    return runlen.max(axis=0).astype(np.float64)


def growing_season_length(tas_year: np.ndarray) -> np.ndarray:
    T = tas_year.shape[0]
    shape = tas_year.shape
    tas_year = tas_year.reshape(T, -1)
    pos = np.arange(T)[:, None]
    cond = tas_year >= 278.15
    m_start = cond & (pos < 181 + 6 - 1)
    beg = first_run_ge(m_start, 6)
    has_start = beg < T
    m_end = (~cond) & (pos >= beg[None, :]) & (pos >= 181)
    end = first_run_ge(m_end, 6)
    length = np.where(has_start, np.where(end < T, end - beg, T - beg), 0)
    return length.reshape(shape[1:])


def first_run_ge(m: np.ndarray, min_len: int) -> np.ndarray:
    T = m.shape[0]
    pos = np.arange(T)[:, None]
    mb = m.astype(bool)
    prev = np.empty_like(mb)
    prev[0] = False
    prev[1:] = mb[:-1]
    rs = mb & ~prev
    starts = np.where(rs, pos, 0)
    last_start = np.maximum.accumulate(starts, axis=0)
    runlen = pos - last_start + 1
    nxt = np.concatenate([mb[1:], np.zeros((1, mb.shape[1]), dtype=bool)], axis=0)
    is_end = mb & ~nxt
    full_len = np.where(is_end, runlen, 0)
    valid_end = is_end & (full_len >= min_len)
    cand = np.where(valid_end, last_start, T)
    return cand.min(axis=0)


VARTGT = {
    "tasmax": (["TXx", "SU", "WSDI"], {"TXx": 2.0, "SU": 25.0, "WSDI": 40.0}),
    "tasmin": (["TNn", "TR", "FD", "CSDI"], {"TNn": 2.0, "TR": 25.0, "FD": 25.0, "CSDI": 15.0}),
    "tas": (["GSL"], {"GSL": 30.0}),
    "pr": (["Rx5day", "CDD", "CWD", "R95p", "R10mm", "SDII"],
           {"Rx5day": 20.0, "CDD": 20.0, "CWD": 15.0, "R95p": 150.0,
            "R10mm": 5.0, "SDII": 0.8}),
}
F_POINTS = (0.05, 0.60)
_D = np.arange(365)
WOB = {
    "tasmax": np.sin(2 * np.pi * 7 * _D / 365) * 0.6 + np.sin(2 * np.pi * 3 * _D / 365 + 1.0) * 0.4,
    "tasmin": np.sin(2 * np.pi * 5 * _D / 365 + 2.0) * 0.6 + np.sin(2 * np.pi * 2 * _D / 365) * 0.4,
    "pr": None,  # pr uses events instead
    "tas": None,
}
EPS_GRID = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0]


def calibrate(model: t4.V4, train_ssps: list[str], sigma_c: float = 1.5,
              quick: bool = False) -> dict:
    t0 = time.time()
    C = (12, 18)
    Cn = int(np.prod(C))

    def smooth(x, sig=sigma_c):
        return gaussian_filter(x, sig, mode="nearest") if sig > 0 else x

    n = 85 * len(train_ssps)
    I_pool = {k: np.empty((n, 12, 18)) for k in SUB_COLUMNS}
    idx = 0
    for s in train_ssps:
        gt = t4.gt_indices_npz(Path("models/cache"), s)
        for k in SUB_COLUMNS:
            I_pool[k][idx:idx + 85] = gt[k][:85]
        idx += 85
    F_pool = np.concatenate([model.F_test[s][:85] for s in train_ssps])

    ths = {
        "tasmax": percentile_doy_window(model.base["tasmax"], 0.90),
        "tasmin": percentile_doy_window(model.base["tasmin"], 0.10),
        "pr": model.th_r95p,
    }

    calib = {"u": {v: np.ones(C) for v in VARS},
             "slope": {v: model.m[v].copy() for v in VARS},
             "d": {v: np.zeros(C) for v in VARS},
             "eps": {v: np.zeros(C) for v in VARS},
             "W": np.zeros(C, dtype=int), "A": np.zeros(C, dtype=np.float64),
             "a": np.ones(C, dtype=np.float64), "L0": np.zeros(C, dtype=np.float64)}

    for v in VARS:
        keys, scales = VARTGT[v]
        mu = {k: I_pool[k].mean(axis=0).reshape(-1) for k in keys}
        Fbar_t = F_pool.mean()
        beta = {}
        a_int = {}
        for k in keys:
            ak, bk = fit_line(I_pool[k].reshape(n, Cn), F_pool)
            beta[k] = bk
            a_int[k] = ak
        S_v = model.S[v]
        m_v = model.m[v]
        # base field per F
        baseF = {Fp: S_v for Fp in F_POINTS}  # d, u added below

        if v == "pr":
            # two-stage: (a, L0) with (W,A)=(9,10), then (W,A) at best (a,L0)
            As1 = [10.0]
            Ws1 = [9]
            a_grid = [0.5, 0.75, 1.0, 1.25, 1.5, 2.0, 3.0]
            L0_grid = [-0.5, 0.0, 0.5, 1.0]
            ncfg = len(a_grid) * len(L0_grid)
            tab = np.empty((ncfg, len(keys), Cn))
            ci = 0
            for a in a_grid:
                for L0 in L0_grid:
                    field = np.maximum(S_v * a + L0 / KG_M2_S_TO_MM_DAY, 0.0)
                    Warr = np.full(C, 9, dtype=int)
                    Aarr = np.full(C, 10.0)
                    for j in range(9):
                        p = np.round(365.0 * (j + 0.5) / 9).astype(int)
                        field = field.copy()
                        field[p] += Aarr / KG_M2_S_TO_MM_DAY
                    r = var_indices_year("pr", field, ths)
                    for ki, k in enumerate(keys):
                        tab[ci, ki] = r[k].reshape(-1)
                    ci += 1
            loss = np.zeros((ncfg, Cn))
            for ki, k in enumerate(keys):
                loss += (tab[:, ki] - mu[k][None, :]) ** 2 / scales[k] ** 2
            b1 = loss.argmin(axis=0)
            i_a = b1 // len(L0_grid)
            i_L = b1 % len(L0_grid)
            a_best = np.array(a_grid)[i_a].reshape(C)
            L0_best = np.array(L0_grid)[i_L].reshape(C)
            # stage 2: W, A per cell
            Ws = list(range(1, 19))
            As = [float(x) for x in np.arange(0.0, 60.01, 2.0)]
            ncfg2 = len(Ws) * len(As)
            tab2 = np.empty((ncfg2, len(keys), Cn))
            ci = 0
            for W in Ws:
                Warr = np.full(C, W, dtype=int)
                for A in As:
                    Aarr = np.full(C, A)
                    field = np.maximum(S_v * a_best + L0_best / KG_M2_S_TO_MM_DAY, 0.0)
                    for j in range(W):
                        p = np.round(365.0 * (j + 0.5) / W).astype(int)
                        field[p] += Aarr / KG_M2_S_TO_MM_DAY
                    r = var_indices_year("pr", field, ths)
                    for ki, k in enumerate(keys):
                        tab2[ci, ki] = r[k].reshape(-1)
                    ci += 1
            loss2 = np.zeros((ncfg2, Cn))
            for ki, k in enumerate(keys):
                loss2 += (tab2[:, ki] - mu[k][None, :]) ** 2 / scales[k] ** 2
            b2 = loss2.argmin(axis=0)
            iW = (b2 // len(As)).astype(int)
            iA = (b2 % len(As)).astype(int)
            calib["a"] = a_best
            calib["L0"] = L0_best
            calib["W"] = np.asarray(Ws)[iW].reshape(C).astype(int)
            calib["A"] = np.asarray(As)[iA].reshape(C)
            print(f"  pr: a med={np.median(a_best):.2f} L0 med={np.median(L0_best):+.2f} "
                  f"W {calib['W'].min()}-{calib['W'].max()} A {calib['A'].min():.0f}-{calib['A'].max():.0f} "
                  f"({time.time()-t0:.0f}s)")
            continue

        # temperature vars: independent per-cell (slope, level) on the daily field
        slope_grid = [float(x) for x in np.arange(-2.0, 12.01, 0.25)] if not quick else [0.0]
        d_grid = [float(x) for x in np.arange(-6.0, 6.01, 0.25)] if not quick else [0.0]
        F_pts = (0.05, 0.30, 0.60) if not quick else (0.1,)
        ncfg = len(slope_grid) * len(d_grid) * len(F_pts)
        tab = np.empty((ncfg, len(keys), Cn))
        ci = 0
        for sl in slope_grid:
            for d in d_grid:
                for Fp in F_pts:
                    field = S_v + sl * (Fp - model.Fbar) + d
                    r = var_indices_year(v, field, ths)
                    for ki, k in enumerate(keys):
                        tab[ci, ki] = r[k].reshape(-1)
                    ci += 1
        tgt_tab = np.empty((ncfg, len(keys), Cn))
        fi = 0
        for sl in slope_grid:
            for d in d_grid:
                for Fp in F_pts:
                    for ki, k in enumerate(keys):
                        tgt_tab[fi, ki] = a_int[k] + beta[k] * Fp
                    fi += 1
        loss = np.zeros((ncfg, Cn))
        for ki, k in enumerate(keys):
            loss += (tab[:, ki] - tgt_tab[:, ki]) ** 2 / scales[k] ** 2
        # baseline: V4 field (slope = m_v, d = 0)
        base_loss = np.zeros(Cn)
        for Fp in F_pts:
            field = S_v + m_v * (Fp - model.Fbar)
            r = var_indices_year(v, field, ths)
            for ki, k in enumerate(keys):
                base_loss += (r[k].reshape(-1) - a_int[k] - beta[k] * Fp) ** 2 / scales[k] ** 2
        b = loss.argmin(axis=0)
        best_loss = loss[b, np.arange(Cn)]
        use = best_loss <= base_loss
        # encode baseline as special index = ncfg
        b = np.where(use, b, ncfg)
        sl_best = np.empty(Cn)
        d_best = np.empty(Cn)
        isb = b == ncfg
        if isb.any():
            sl_best[isb] = m_v.reshape(-1)[isb]
            d_best[isb] = 0.0
        isg = ~isb
        if isg.any():
            b2 = b[isg]
            sl_best[isg] = np.asarray(slope_grid)[(b2 // (len(d_grid) * len(F_pts))).astype(int)]
            d_best[isg] = np.asarray(d_grid)[((b2 % (len(d_grid) * len(F_pts))) // len(F_pts)).astype(int)]
        if sigma_c > 0:
            sl_best = gaussian_filter(sl_best.reshape(12, 18), sigma_c, mode="nearest")
            d_best = gaussian_filter(d_best.reshape(12, 18), sigma_c, mode="nearest")
        calib["slope"][v] = sl_best.reshape(12, 18)
        calib["d"][v] = d_best.reshape(12, 18)
        calib["eps"][v] = np.zeros((12, 18))
        print(f"  {v}: slope med={np.median(sl_best):+.3f} d med={np.median(d_best):+.3f} "
              f"frac_baseline={isb.mean():.2f} ({time.time()-t0:.0f}s)")
        # stage 2: local (d, eps) refinement with wobble (saturated-cell fix)
        w = WOB[v]
        if w is not None:
            sl_c = sl_best.reshape(12, 18)
            d_c = d_best.reshape(12, 18).copy()
            d_best2 = d_c.copy()
            eps_c = np.zeros((12, 18))
            base_loss_c = base_loss
            dj = [0.0, 0.25, -0.25, 0.5, -0.5, 0.75, -0.75]
            nF = len(F_pts)
            n_ref = len(EPS_GRID) * len(dj) * nF
            tabr = np.empty((n_ref, len(keys), Cn))
            ci = 0
            for eps in EPS_GRID:
                for dd in dj:
                    for Fp in F_pts:
                        field = S_v + sl_c * (Fp - model.Fbar) + (d_c + dd) + eps * w[:, None, None]
                        r = var_indices_year(v, field, ths)
                        for ki, k in enumerate(keys):
                            tabr[ci, ki] = r[k].reshape(-1)
                        ci += 1
                        del field, r
            lossr = np.zeros((len(EPS_GRID), len(dj), len(keys), Cn))
            for ki, k in enumerate(keys):
                tabr_f = tabr[:, ki].reshape(len(EPS_GRID), len(dj), nF, Cn)
                tgt_f = np.stack([a_int[k] + beta[k] * Fp for Fp in F_pts])  # (nF, Cn)
                lossr[:, :, ki, :] = ((tabr_f - tgt_f[None, None]) ** 2).mean(axis=2) / scales[k] ** 2
            lossr_ed = lossr.sum(axis=2)  # (neps, ndj, nkeys, Cn) -> (neps, ndj, Cn)
            flat_idx = lossr_ed.reshape(len(EPS_GRID) * len(dj), Cn).argmin(axis=0)
            i_eps = (flat_idx // len(dj)).astype(int)
            i_dd = (flat_idx % len(dj)).astype(int)
            d_new = d_c + np.asarray(dj)[i_dd].reshape(12, 18)
            eps_new = np.asarray(EPS_GRID)[i_eps].reshape(12, 18)
            best1 = loss[b, np.arange(Cn)]
            best2 = lossr_ed.reshape(-1, Cn)[br_ed * 1 + np.arange(Cn) * 0, np.arange(Cn)] if False else None
            # loss at the chosen (eps, dd) per cell:
            chosen = lossr_ed.reshape(-1, Cn)[flat_idx, np.arange(Cn)]  # (Cn,)
            acc = (chosen < best1) & (chosen < base_loss_c)
            d_best2 = np.where(acc.reshape(12, 18), d_new, d_best2)
            eps_c = np.where(acc.reshape(12, 18), eps_new, 0.0)
            calib["d"][v] = d_best2
            calib["eps"][v] = eps_c
            print(f"  {v}: wobble used in {(acc.reshape(12,18).mean()*100):.0f}% of cells, "
                  f"eps med={np.median(eps_c[eps_c>0]) if (eps_c>0).any() else 0:.2f} ({time.time()-t0:.0f}s)")
    return calib


class V5:
    def __init__(self, model: t4.V4, calib: dict):
        self.model = model
        self.calib = calib

    def _base(self, v: str, scen: str) -> np.ndarray:
        m = self.model
        c = self.calib
        Ft = m.F_test[scen]
        sl = 0.0 if v == "pr" else c["slope"][v]
        d = c["d"][v]
        eps = c["eps"][v]
        w = WOB[v]
        off = sl * (Ft - m.Fbar)[:, None, None] + d[None, :, :]
        if w is not None and (eps != 0).any():
            arr = m.S[v][None, :, :, :] + off[:, None, :, :] + (eps[None, None] * w[:, None, None])
        else:
            arr = m.S[v][None, :, :, :] + off[:, None, :, :]  # (86,365,12,18)
        flat = np.empty((31389, 12, 18), dtype=np.float32)
        i = 0
        for y in range(86):
            nn = 364 if y == 85 else 365
            flat[i:i + nn] = arr[y, :nn].astype(np.float32)
            i += nn
        return flat

    def predict_daily(self, scen: str) -> dict[str, np.ndarray]:
        out = {v: self._base(v, scen) for v in VARS}
        c = self.calib
        # pr: scale + L0 shift + events
        a = c["a"]
        L0 = c["L0"]
        out["pr"] = out["pr"] * a[None, :, :] + L0[None, :, :] / KG_M2_S_TO_MM_DAY
        W = np.clip(c["W"].astype(int), 0, 182)
        A = c["A"]
        if W.max() > 0:
            pat = np.zeros((365, 12, 18), dtype=np.float32)
            for j in range(int(W.max())):
                active = np.argwhere(W > j)
                ii, jj = active[:, 0], active[:, 1]
                p = np.round(365.0 * (j + 0.5) / W[ii, jj]).astype(int)
                pat[p, ii, jj] += (A[ii, jj] / KG_M2_S_TO_MM_DAY).astype(np.float32)
            tile = np.concatenate([pat] * 85 + [pat[:364]], axis=0)
            out["pr"] = np.maximum(out["pr"] + tile, 0.0).astype(np.float32)
        return out



def build_model(train_ssps, hist_start=1961, sigma=2.0):
    model = t4.V4(hist_start=hist_start, sigma=sigma)
    model.fit(train_ssps)
    ds2 = nc.Dataset(str(lite_file("hist_late")))
    model.base = base_period_slice({v: np.asarray(ds2.variables[v][0], dtype=np.float64) for v in VARS})
    ds2.close()
    model.th_r95p = percentile_doy_window(model.base["pr"] * 86400.0, 0.95)
    return model


def eval_v5(v5: V5, scen: str) -> tuple[float, dict]:
    gt = t4.gt_indices_npz(Path("models/cache"), scen)
    daily = v5.predict_daily(scen)
    idx = compute_indices(daily, v5.model.base)
    S, per = score_lite(gt, idx)
    return S, per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", default="ssp370")
    ap.add_argument("--sigmas", default="2.0")
    ap.add_argument("--calib-sigma", default="1.5")
    ap.add_argument("--no-pr", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--save-best", action="store_true")
    a = ap.parse_args()

    ho = None if a.holdout == "none" else a.holdout
    train_ssps = [s for s in t4.SSPS if s != ho] if ho else t4.SSPS
    print(f"holdout={ho} train={train_ssps}")

    best = None
    for sig in [float(x) for x in a.sigmas.split(",")]:
        model = build_model(train_ssps, sigma=sig)
        calib = calibrate(model, train_ssps, sigma_c=float(a.calib_sigma),
                          quick=a.quick)
        v5 = V5(model, calib)
        test_ssps = [ho] if ho else ["ssp245"]
        for s in test_ssps:
            S, per = eval_v5(v5, s)
            print(f"\n== sigma={sig} holdout={s}: S={S:+.5f} ==")
            for k in SUB_COLUMNS:
                print(f"  {k:8s} {per[k]:+.4f}")
        if ho and (best is None or S > best[0]):
            best = (S, sig)
    print(f"\nbest sigma (CV): {best}")


if __name__ == "__main__":
    main()
