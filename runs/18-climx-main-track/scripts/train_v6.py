"""V6: per-cell target curves in F + per-year daily-field level matching.

For each cell and index, estimate the scenario-mean index curve g_c(F) by a
local median over pooled (F, index) points from the training SSPs.  The test
target for year y is T(y) = g_c(F_test(y)).  Then build a daily field per
year whose official indices match T(y):

- tasmax: S + d(y) + eps(y)*w_max(doy)  -> TXx, SU, ID, WSDI
- tasmin: S + d(y) + eps(y)*w_min(doy)  -> TNn, TR, FD, CSDI
- tas:    S + d(y)                       -> GSL
- pr:     S*a + L0 + W(y) events x A(y)  -> Rx5day, CDD, CWD, R95p, R10mm, SDII

Per-year (d, eps) / (W, A) chosen by weighted argmin over a response table,
weights w_k = 1/(var(T_k) + c_k) so low-signal indices cannot blow up.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import faulthandler
faulthandler.enable()
import numpy as np
import netCDF4 as nc
from scipy.ndimage import gaussian_filter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from climx_common import lite_file, SUB_COLUMNS, make_submission_parquet, nearest_upsample_lite_to_full
from climx_indices import compute_indices, base_period_slice, percentile_doy_window, KG_M2_S_TO_MM_DAY
from rnNSE_lite import score_lite
import train_v4 as t4

VARS = t4.VARS
FLOOR = {
    "TXx": 0.25, "TNn": 0.25, "SU": 250.0, "TR": 250.0, "FD": 250.0, "ID": 250.0,
    "WSDI": 400.0, "CSDI": 400.0, "GSL": 4.0,
    "Rx5day": 25.0, "CDD": 4.0, "CWD": 4.0, "R95p": 100.0, "R10mm": 1.0, "SDII": 0.01,
}
VARIDX = {
    "tasmax": ["TXx", "SU", "ID", "WSDI"],
    "tasmin": ["TNn", "TR", "FD", "CSDI"],
    "tas": ["GSL"],
    "pr": ["Rx5day", "CDD", "CWD", "R95p", "R10mm", "SDII"],
}
D_GRID = np.arange(-6.0, 14.01, 0.25)          # K
EPS_GRID = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0])   # K
TA = np.arange(365)
N = 86
N_DAYS = 31389


# ----------------------------------------------------------------------
# single-cell fast annual indices (one 365-day year, arrays over a cfg axis)
# ----------------------------------------------------------------------

def _run_stats(m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """m: (C, 365) bool. Returns (longest_run,) per row (6-day runs handled by caller)."""
    T = m.shape[1]
    pos = np.arange(T)[None, :]
    prev = np.empty_like(m)
    prev[:, 0] = False
    prev[:, 1:] = m[:, :-1]
    rs = m & ~prev
    starts = np.where(rs, pos, 0)
    last_start = np.maximum.accumulate(starts, axis=1)
    runlen = pos - last_start + 1
    nxt = np.concatenate([m[:, 1:], np.zeros((m.shape[0], 1), dtype=bool)], axis=1)
    is_end = m & ~nxt
    total = np.where(is_end & (runlen >= 6), runlen, 0).sum(axis=1)
    longest = np.where(m, runlen, 0).max(axis=1)
    return total.astype(np.float64), longest.astype(np.float64)


def _first_run_ge(m: np.ndarray, min_len: int) -> np.ndarray:
    """m: (C, T). First day index of first run of >=min_len True; T if none."""
    T = m.shape[1]
    pos = np.arange(T)[None, :]
    prev = np.empty_like(m)
    prev[:, 0] = False
    prev[:, 1:] = m[:, :-1]
    rs = m & ~prev
    starts = np.where(rs, pos, 0)
    last_start = np.maximum.accumulate(starts, axis=1)
    runlen = pos - last_start + 1
    nxt = np.concatenate([m[:, 1:], np.zeros((m.shape[0], 1), dtype=bool)], axis=1)
    is_end = m & ~nxt
    valid_end = is_end & (runlen >= min_len)
    cand = np.where(valid_end, last_start, T)
    return cand.min(axis=1)


def indices_temp(field: np.ndarray, th_wsdi: np.ndarray, th_csdi: np.ndarray) -> np.ndarray:
    """field: (C, 365). Returns (C, 4): tasmax -> (TXx,SU,ID,WSDI); tasmin -> (TNn,TR,FD,CSDI)."""
    out = np.empty((field.shape[0], 4))
    if th_wsdi is not None:
        out[:, 0] = field.max(axis=1)
        out[:, 1] = (field > 298.15).sum(axis=1)
        out[:, 2] = (field < 273.15).sum(axis=1)
        out[:, 3] = _run_stats((field > th_wsdi[None, :]).astype(bool))[0]
    else:
        out[:, 0] = field.min(axis=1)
        out[:, 1] = (field > 293.15).sum(axis=1)
        out[:, 2] = (field < 273.15).sum(axis=1)
        out[:, 3] = _run_stats((field < th_csdi[None, :]).astype(bool))[0]
    return out


def indices_gsl(field: np.ndarray) -> np.ndarray:
    """field (C, 365) tas. Returns (C,) GSL (xclim season semantics)."""
    pos = np.arange(365)[None, :]
    cond = field >= 278.15
    m_start = cond & (pos < 181 + 6 - 1)
    beg = _first_run_ge(m_start, 6)
    has_start = beg < 365
    m_end = (~cond) & (pos >= beg[:, None]) & (pos >= 181)
    end = _first_run_ge(m_end, 6)
    length = np.where(has_start, np.where(end < 365, end - beg, 365 - beg), 0)
    return length.astype(np.float64)


def indices_pr(field: np.ndarray, th_r95p: np.ndarray) -> np.ndarray:
    """field (C, 365) pr in kg m-2 s-1. Returns (C, 6):
    (Rx5day, CDD, CWD, R95p, R10mm, SDII). Rx5day uses periodic cross-year windows."""
    C = field.shape[0]
    if th_r95p.ndim == 3:
        th_r95p = th_r95p.reshape(-1, 365)
    if th_r95p.shape == (365,):
        th_r95p = np.tile(th_r95p, (C, 1))
    pr = field * KG_M2_S_TO_MM_DAY
    two = np.concatenate([pr, pr], axis=1)
    csum = np.cumsum(two, axis=1)
    pr5 = np.full_like(two, np.nan)
    pr5[:, 4] = csum[:, 4]
    pr5[:, 5:] = csum[:, 5:] - csum[:, :-5]
    out = np.empty((C, 6))
    out[:, 0] = np.nanmax(pr5[:, 365:730], axis=1)
    out[:, 1] = _run_stats((pr < 1.0).astype(bool))[1]
    out[:, 2] = _run_stats((pr >= 1.0).astype(bool))[1]
    out[:, 3] = np.where(pr > th_r95p, pr, 0.0).sum(axis=1)
    out[:, 4] = (pr >= 10.0).sum(axis=1)
    ws = np.where(pr >= 1.0, pr, 0.0).sum(axis=1)
    wc = (pr >= 1.0).sum(axis=1)
    out[:, 5] = np.where(wc == 0, 0.0, ws / wc)
    return out


# ----------------------------------------------------------------------
# target curves
# ----------------------------------------------------------------------

def fit_targets(train_ssps: list[str], F: dict, F_test: np.ndarray,
                h: float = 0.15, min_n: int = 8, h_max: float = 0.8,
                smooth: float = 2.0, tau_age: float = 0.0) -> dict[str, np.ndarray]:
    """T[k] (N, 12, 18): weighted median over pooled train points.
    With tau_age > 0, weight also decays with age difference (SSPs branched
    from a common 2015 state, so same-age years share internal state)."""
    n = len(F_test)
    Ftr = np.concatenate([F[s][:85] for s in train_ssps])
    ages = np.concatenate([np.arange(85.0) for s in train_ssps])
    T = {}
    for k in SUB_COLUMNS:
        Itr = np.stack([t4.gt_indices_npz(Path("models/cache"), s)[k][:85].reshape(85, -1)
                        for s in train_ssps], axis=1).transpose(1, 0, 2).reshape(-1, 216)
        Tk = np.zeros((n, 216))
        for t in range(n):
            hh = h
            while True:
                sup = np.abs(Ftr - F_test[t]) < hh
                if sup.sum() >= min_n or hh >= h_max:
                    break
                hh *= 1.5
            if not sup.any():
                w = np.zeros(len(Ftr))
                w[np.argmin(np.abs(Ftr - F_test[t]))] = 1.0
            else:
                w = np.where(sup, 1.0, 0.0)
                if tau_age > 0:
                    w = w * np.exp(-((ages - t) / tau_age) ** 2)
            order = np.argsort(Itr, axis=0)  # (npts, 216)
            vs = np.take_along_axis(Itr, order, axis=0)
            ws = np.take_along_axis(w[:, None], order, axis=0)
            cw = np.cumsum(ws, axis=0)
            tot = cw[-1]
            half = tot / 2
            med_idx = np.argmax(cw >= half[None, :], axis=0)
            Tk[t] = np.take_along_axis(vs, med_idx[None, :], axis=0)[0]
        Tk = gaussian_filter(Tk, (smooth, 0.01), mode="nearest") if smooth > 0 else Tk
        T[k] = Tk.reshape(n, 12, 18)
    return T


# ----------------------------------------------------------------------
# model
# ----------------------------------------------------------------------

class V6:
    def __init__(self, S, wob_max, wob_min, pr_years, base, th_wsdi, th_csdi, th_r95p):
        self.S = S
        self.wob_max = wob_max
        self.wob_min = wob_min
        self.pr_years = pr_years
        self.base = base
        self.th_wsdi = th_wsdi
        self.th_csdi = th_csdi
        self.th_r95p = th_r95p

    @staticmethod
    def build_S_wob(train_ssps, hist_start=1961):
        t0 = time.time()
        data = {}
        for v in VARS:
            def _load(key):
                ds = nc.Dataset(str(lite_file(key)))
                a = np.asarray(ds.variables[v][0], dtype=np.float64)
                ds.close()
                return a
            h1 = _load("hist_early")
            h2 = _load("hist_late")
            ssps = [_load(s) for s in train_ssps]
            data[v] = dict(hist=np.concatenate([h1, h2], axis=0), ssps=ssps)
        cnt = 0
        acc = {v: np.zeros((365, 12, 18)) for v in VARS}
        years = {v: [] for v in VARS}
        HE_LEN = 82 * 365  # hist_early covers 1850-1931
        for y in range(hist_start, 2014):
            y0 = (y - 1932) * 365 + HE_LEN
            for v in VARS:
                a = data[v]["hist"][y0:y0 + 365]
                acc[v] += a
                years[v].append(a)
            cnt += 1
        n_hist = 2014 - hist_start
        for s in train_ssps:
            for v in VARS:
                a = data[v]["ssps"][train_ssps.index(s)][:85 * 365]
                acc[v] += a.reshape(85, 365, 12, 18).sum(axis=0)
                for y in range(85):
                    years[v].append(a[y * 365:(y + 1) * 365])
        cnt = n_hist + 85 * len(train_ssps)
        S = {v: acc[v] / cnt for v in VARS}

        def _wob(v):
            norms = np.array([float(np.abs(y - S[v]).max()) for y in years[v]])
            top = np.argsort(norms)[::-1][:10]
            pat = np.zeros((365, 12, 18))
            for y in top:
                pat += years[v][int(y)] - S[v]
            pat /= len(top)
            m = np.abs(pat).max(axis=0)
            m[m < 1e-6] = 1.0
            return pat / m[None]

        wob_max = _wob("tasmax")
        wob_min = _wob("tasmin")
        pr_years = np.stack(years["pr"], axis=0).astype(np.float32)  # (223,365,12,18)
        del data, years, acc
        print(f"  S/wob built {time.time() - t0:.0f}s")
        return S, wob_max, wob_min, pr_years

    # ---- pr stage 1: per-cell (a, L0) at the mid-F target ----
    def _temp_field(self, T, S_v, wob, th_wsdi, th_csdi, keys):
        """Per-year (d, eps) selection for a temperature var. Returns (31389,12,18)."""
        neps, nd = len(EPS_GRID), len(D_GRID)
        tgt = np.stack([T[k] for k in keys], axis=-1)  # (N,12,18,nk)
        varr = np.stack([T[k].var(axis=0) + FLOOR[k] for k in keys], axis=2)
        wgt = 1.0 / varr  # (12,18,nk)
        nk = len(keys)
        flat = np.empty((N_DAYS, 12, 18), dtype=np.float32)
        for i in range(12):
            for j in range(18):
                Sc = S_v[:, i, j]
                wc = wob[:, i, j]
                th_ = th_wsdi[:, i, j] if th_wsdi is not None else th_csdi[:, i, j]
                dmat = Sc[None, None, :] + D_GRID[None, :, None]
                emat = (EPS_GRID[:, None, None] * wc[None, None, :])
                tab = indices_temp((dmat + emat).reshape(neps * nd, 365),
                                   th_ if th_wsdi is not None else None,
                                   None if th_wsdi is not None else th_)
                tab = tab.reshape(neps, nd, nk)
                loss = (wgt[i, j][None, None, None] *
                        (tab[None] - tgt[:, i, j, :][:, None, None]) ** 2)
                loss = loss.sum(axis=3)  # (N, neps, nd)
                idx_ = np.argmin(loss.reshape(loss.shape[0], -1), axis=1)
                ie = (idx_ // nd).astype(int)
                id_ = (idx_ % nd).astype(int)
                add = EPS_GRID[ie][:, None] * wc[None, :]
                for t in range(N):
                    nn = 364 if t == N - 1 else 365
                    flat[t * 365:t * 365 + nn, i, j] = (
                        Sc[:nn] + D_GRID[id_[t]] + add[t, :nn]).astype(np.float32)
        return flat

    def predict_daily(self, T, a_pr=None, L0_pr=None):
        keys_max = VARIDX["tasmax"]
        keys_min = VARIDX["tasmin"]
        out = {}
        t0 = time.time()
        out["tasmax"] = self._temp_field(T, self.S["tasmax"], self.wob_max,
                                         self.th_wsdi, None, keys_max)
        print(f"  tasmax {time.time() - t0:.0f}s", flush=True)
        t0 = time.time()
        out["tasmin"] = self._temp_field(T, self.S["tasmin"], self.wob_min,
                                         None, self.th_csdi, keys_min)
        print(f"  tasmin {time.time() - t0:.0f}s", flush=True)
        t0 = time.time()
        S_v = self.S["tas"]
        flat = np.empty((N_DAYS, 12, 18), dtype=np.float32)
        for i in range(12):
            for j in range(18):
                tab = indices_gsl(S_v[:, i, j][None] + D_GRID[:, None])
                sel = D_GRID[np.argmin((tab[None] - T["GSL"][:, i, j][:, None]) ** 2, axis=1)]
                for t in range(N):
                    nn = 364 if t == N - 1 else 365
                    flat[t * 365:t * 365 + nn, i, j] = (
                        S_v[:nn, i, j] + sel[t]).astype(np.float32)
        out["tas"] = flat
        print(f"  tas {time.time() - t0:.0f}s", flush=True)
        t0 = time.time()
        out["pr"] = self._pr_field(T, a_pr, L0_pr)
        print(f"  pr {time.time() - t0:.0f}s", flush=True)
        return out

    PR_A_GRID = (0.55, 0.7, 0.85, 1.0, 1.15, 1.3, 1.5)

    def _pr_field(self, T, a_pr=None, L0_pr=None):
        keys = ["R95p", "CDD", "CWD", "Rx5day", "R10mm"]
        PRCAP = {"Rx5day": 10.0, "CDD": 8.0, "CWD": 20.0, "R95p": 50.0, "R10mm": 2.0}
        tgtT = np.stack([T[k] for k in keys], axis=-1)  # (N,12,18,5)
        wgt = np.stack([1.0 / np.maximum(T[k].mean(axis=0) ** 2, PRCAP[k] ** 2)
                        for k in keys], axis=-1).reshape(216, 5)
        ny = self.pr_years.shape[0]
        # template index table: (nA*ny, 216, 5)
        tab = np.empty((len(self.PR_A_GRID) * ny, 216, 5))
        t0 = time.time()
        for x, a in enumerate(self.PR_A_GRID):
            for y0 in range(ny):
                f = np.maximum(self.pr_years[y0].astype(np.float64) * a, 0.0)
                r = indices_pr(f.transpose(2, 0, 1).reshape(216, 365),
                               self.th_r95p.reshape(216, 365))
                tab[x * ny + y0] = r[:, [3, 1, 2, 0, 4]]  # keys order
        print(f"    pr template table {time.time() - t0:.0f}s", flush=True)
        out = np.empty((N_DAYS, 12, 18), dtype=np.float32)
        ci = np.arange(12)[:, None].repeat(18, axis=1).reshape(216)
        cj = np.arange(18)[None, :].repeat(12, axis=0).reshape(216)
        for t in range(N):
            tgtv = tgtT[t].reshape(216, 5)
            loss = (wgt[None] * (tab - tgtv[None]) ** 2).sum(axis=2)
            b = loss.argmin(axis=0)  # (216,) in [0, nA*ny)
            ia = (b // ny).astype(int)
            iy = (b % ny).astype(int)
            a_sel = np.asarray(self.PR_A_GRID)[ia].reshape(216)
            flat = (iy[:, None] * 365 + np.arange(365)[None, :]) * 216
            flat += ci[:, None] * 18 + cj[:, None]
            year_f = self.pr_years.reshape(-1)[flat]  # (216, 365)
            year_f = np.maximum(year_f.astype(np.float64) * a_sel[:, None], 0.0)
            nn = 364 if t == N - 1 else 365
            out[t * 365:t * 365 + nn] = year_f[:, :nn].T.reshape(nn, 12, 18).astype(np.float32)
        return out


def build_model(train_ssps, hist_start=1961):
    t0 = time.time()
    S, wob_max, wob_min, pr_years = V6.build_S_wob(train_ssps, hist_start)
    ds2 = nc.Dataset(str(lite_file("hist_late")))
    base = base_period_slice({v: np.asarray(ds2.variables[v][0], dtype=np.float64) for v in VARS})
    ds2.close()
    th_wsdi = percentile_doy_window(base["tasmax"], 0.90)
    th_csdi = percentile_doy_window(base["tasmin"], 0.10)
    th_r95p = percentile_doy_window(base["pr"] * 86400.0, 0.95)
    m = V6(S, wob_max, wob_min, pr_years, base, th_wsdi, th_csdi, th_r95p)
    print(f"  model built {time.time() - t0:.0f}s")
    return m


def load_F():
    co2_ref = float(np.asarray(nc.Dataset(str(lite_file("ssp126"))).variables["CO2_LBC"][0, 0]))
    F = {}
    for s in t4.SSPS + ["test_forcings"]:
        c = np.asarray(nc.Dataset(str(lite_file(s))).variables["CO2_LBC"][0], dtype=np.float64)
        F[s] = np.log(c / co2_ref)
    return F


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", default="ssp370")
    ap.add_argument("--h", type=float, default=0.15)
    ap.add_argument("--min-n", type=int, default=8)
    ap.add_argument("--smooth", type=float, default=2.0)
    ap.add_argument("--tau-age", type=float, default=0.0)
    ap.add_argument("--hist-start", type=int, default=1961)
    a = ap.parse_args()

    ho = None if a.holdout == "none" else a.holdout
    train_ssps = [s for s in t4.SSPS if s != ho] if ho else t4.SSPS
    print(f"holdout={ho} train={train_ssps}")
    F = load_F()
    model = build_model(train_ssps, hist_start=a.hist_start)
    scen = ho if ho else "test_forcings"
    F_test = F[scen]
    t0 = time.time()
    T = fit_targets(train_ssps, F, F_test, h=a.h, min_n=a.min_n, smooth=a.smooth, tau_age=a.tau_age)
    print(f"  targets {time.time() - t0:.0f}s")
    t0 = time.time()
    daily = model.predict_daily(T, None, None)
    print(f"  fields {time.time() - t0:.0f}s")
    idx = compute_indices(daily, model.base)
    if ho:
        gt = t4.gt_indices_npz(Path("models/cache"), ho)
        S, per = score_lite(gt, idx)
        print(f"\n== V6 CV holdout={ho}: S={S:+.5f} ==")
        for k in SUB_COLUMNS:
            print(f"  {k:8s} {per[k]:+.4f}")
    else:
        outd = Path("submissions")
        outd.mkdir(exist_ok=True)
        idx_full = {k: nearest_upsample_lite_to_full(idx[k].astype(np.float32)) for k in SUB_COLUMNS}
        p = make_submission_parquet(idx_full, outd / "submission_v6.parquet")
        print("wrote", p)


if __name__ == "__main__":
    main()
