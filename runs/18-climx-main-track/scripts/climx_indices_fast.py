"""Fast implementation of the 15 annual indices (ClimX).

Produces identical results to climx_indices.compute_indices.
Run-based indices (GSL, WSDI, CSDI, CDD, CWD) are vectorized over blocks of
years (all cells at once); cheap indices are vectorized over all 86 years.
Last year (2100) has 364 days and is padded per-index with identity values.
"""
from __future__ import annotations

import numpy as np

from climx_indices import year_slices, percentile_doy_window

KG_M2_S_TO_MM_DAY = 86400.0
N_DAYS = 31389
N_YEARS = 86


def _run_start_pos0_3d(m: np.ndarray) -> np.ndarray:
    """m: (Y, T, C) bool -> (Y, T, C) int32 start position of the run
    containing each position (0 where no run started yet)."""
    pos = np.arange(m.shape[1], dtype=np.int32)[None, :, None]
    prev = np.empty(m.shape, dtype=bool)
    prev[:, 0] = False
    prev[:, 1:] = m[:, :-1]
    rs = m & ~prev
    starts = np.where(rs, pos, 0)
    return np.maximum.accumulate(starts, axis=1)


def longest_run_annual_3d(m: np.ndarray) -> np.ndarray:
    T = m.shape[1]
    pos = np.arange(T, dtype=np.int32)[None, :, None]
    start = _run_start_pos0_3d(m.astype(bool))
    runlen = np.where(m, pos - start + 1, 0)
    return runlen.max(axis=1).astype(np.float64)


def windowed_run_sum_annual_3d(m: np.ndarray, min_len: int) -> np.ndarray:
    T = m.shape[1]
    pos = np.arange(T, dtype=np.int32)[None, :, None]
    mb = m.astype(bool)
    start = _run_start_pos0_3d(mb)
    runlen = pos - start + 1
    nxt = np.concatenate([mb[:, 1:], np.zeros((mb.shape[0], 1) + mb.shape[2:], dtype=bool)], axis=1)
    is_end = mb & ~nxt
    total = np.where(is_end & (runlen >= min_len), runlen, 0).sum(axis=1)
    return total.astype(np.float64)


def first_run_ge_3d(m: np.ndarray, min_len: int) -> np.ndarray:
    T = m.shape[1]
    pos = np.arange(T, dtype=np.int32)[None, :, None]
    mb = m.astype(bool)
    start = _run_start_pos0_3d(mb)
    runlen = pos - start + 1
    nxt = np.concatenate([mb[:, 1:], np.zeros((mb.shape[0], 1) + mb.shape[2:], dtype=bool)], axis=1)
    is_end = mb & ~nxt
    full_len = np.where(is_end, runlen, 0)
    valid_end = is_end & (full_len >= min_len)
    cand = np.where(valid_end, start, T)
    return cand.min(axis=1)


def growing_season_length_3d(tas_y: np.ndarray, thresh: float = 278.15,
                             window: int = 6, mid_idx: int = 181) -> np.ndarray:
    """tas_y: (Y, T, C) K -> (Y, C) GSL."""
    T = tas_y.shape[1]
    pos = np.arange(T, dtype=np.int32)[None, :, None]
    cond = tas_y >= thresh
    m_start = cond & (pos < mid_idx + window - 1)
    beg = first_run_ge_3d(m_start, window)
    has_start = beg < T
    cold = ~cond
    m_end = cold & (pos >= beg[:, None, ...]) & (pos >= mid_idx)
    end = first_run_ge_3d(m_end, window)
    length = np.where(has_start,
                      np.where(end < T, end - beg, T - beg),
                      0)
    return length.astype(np.float64)


def _padY_f(a: np.ndarray, pad, C: int) -> np.ndarray:
    T = a.shape[0]
    o = np.empty((N_YEARS * 365, C), dtype=a.dtype)
    o[:T] = a
    o[T:] = pad
    return o.reshape(N_YEARS, 365, C)


def _padY_b(m: np.ndarray, C: int) -> np.ndarray:
    T = m.shape[0]
    o = np.zeros((N_YEARS * 365, C), dtype=bool)
    o[:T] = m
    return o.reshape(N_YEARS, 365, C)


def compute_indices_fast(daily: dict, base_daily: dict, year_block: int = 16) -> dict:
    """daily: var -> (31389, H, W). base_daily: var -> (10950,H,W) raw or (365,H,W) doy threshold.
    Returns name -> (86, H, W), identical to climx_indices.compute_indices."""
    T = next(iter(daily.values())).shape[0]
    ys = year_slices(T)
    tasmax = daily["tasmax"].astype(np.float64)
    tasmin = daily["tasmin"].astype(np.float64)
    tas = (daily["tas"].astype(np.float64) if "tas" in daily
           else (tasmax + tasmin) / 2.0)
    pr = daily["pr"].astype(np.float64) * KG_M2_S_TO_MM_DAY

    def _doy_th(a, q, scale=1.0):
        a = a.astype(np.float64)
        if a.shape[0] == 365:
            return a
        return percentile_doy_window(a * scale, q)

    th_wsdi = _doy_th(base_daily["tasmax"], 0.90)
    th_csdi = _doy_th(base_daily["tasmin"], 0.10)
    th_r95p = _doy_th(base_daily["pr"], 0.95, KG_M2_S_TO_MM_DAY)

    lat_lon = tasmax.shape[1:]
    C = int(np.prod(lat_lon))

    tm = tasmax.reshape(T, C)
    tn = tasmin.reshape(T, C)
    ta = tas.reshape(T, C)
    pa = pr.reshape(T, C)

    doy = np.arange(T) % 365
    th_w = th_wsdi.reshape(365, C)[doy]
    th_c = th_csdi.reshape(365, C)[doy]
    th_p = th_r95p.reshape(365, C)[doy]
    mask_wsdi = tm > th_w
    mask_csdi = tn < th_c
    mask_pr95 = pa > th_p
    pr_lt1 = pa < 1.0
    pr_ge1 = pa >= 1.0
    pr_ge10 = pa >= 10.0

    pr5 = np.full_like(pa, np.nan)
    csum = np.cumsum(pa, axis=0)
    pr5[4] = csum[4]
    pr5[5:] = csum[5:] - csum[:-5]

    names = ["TXx", "TNn", "SU", "TR", "FD", "ID", "GSL", "WSDI", "CSDI",
             "Rx5day", "CDD", "CWD", "R95p", "R10mm", "SDII"]
    out = {k: np.empty((N_YEARS,) + lat_lon, dtype=np.float64) for k in names}

    # ---- cheap indices: vectorized across all years (last year padded) ----
    out["TXx"].reshape(-1)[:] = _padY_f(tm, -np.inf, C).max(axis=1).reshape(-1)
    out["TNn"].reshape(-1)[:] = _padY_f(tn, np.inf, C).min(axis=1).reshape(-1)
    out["SU"].reshape(-1)[:] = _padY_b(tm > 298.15, C).sum(axis=1).reshape(-1)
    out["TR"].reshape(-1)[:] = _padY_b(tn > 293.15, C).sum(axis=1).reshape(-1)
    out["FD"].reshape(-1)[:] = _padY_b(tn < 273.15, C).sum(axis=1).reshape(-1)
    out["ID"].reshape(-1)[:] = _padY_b(tm < 273.15, C).sum(axis=1).reshape(-1)
    out["Rx5day"].reshape(-1)[:] = np.nanmax(_padY_f(pr5, np.nan, C), axis=1).reshape(-1)
    out["R95p"].reshape(-1)[:] = _padY_f(np.where(mask_pr95, pa, 0.0), 0.0, C).sum(axis=1).reshape(-1)
    out["R10mm"].reshape(-1)[:] = _padY_b(pr_ge10, C).sum(axis=1).reshape(-1)
    wet_sum = _padY_f(np.where(pr_ge1, pa, 0.0), 0.0, C).sum(axis=1)
    wet_cnt = _padY_b(pr_ge1, C).sum(axis=1)
    out["SDII"].reshape(-1)[:] = np.where(wet_cnt == 0, 0.0, wet_sum / wet_cnt).reshape(-1)
    del pr5, csum, wet_sum, wet_cnt

    # ---- run-based indices: year blocks x all cells ----
    mw = _padY_b(mask_wsdi, C)
    mc = _padY_b(mask_csdi, C)
    p1 = _padY_b(pr_lt1, C)
    p2 = _padY_b(pr_ge1, C)
    # tas padded (last day of 2100) with neutral value (does not affect GSL:
    # padded day is beyond the real 364 days; GSL result on 365>=364 days with
    # an extra day could only matter if runs extend to day 364 (index 364).
    # To be exact, pad with the year's last real day value is NOT safe either;
    # instead compute GSL on the unpadded block for the last year.
    for y0 in range(0, N_YEARS, year_block):
        y1 = min(y0 + year_block, N_YEARS)
        out["WSDI"][y0:y1].reshape(-1)[:] = windowed_run_sum_annual_3d(mw[y0:y1], 6).reshape(-1)
        out["CSDI"][y0:y1].reshape(-1)[:] = windowed_run_sum_annual_3d(mc[y0:y1], 6).reshape(-1)
        out["CDD"][y0:y1].reshape(-1)[:] = longest_run_annual_3d(p1[y0:y1]).reshape(-1)
        out["CWD"][y0:y1].reshape(-1)[:] = longest_run_annual_3d(p2[y0:y1]).reshape(-1)
        gy1 = min(y1, N_YEARS - 1)  # full 365-day years only
        if gy1 > y0:
            tb = ta[ys[y0][0]:ys[gy1 - 1][1]].reshape(gy1 - y0, 365, C)
            out["GSL"][y0:gy1].reshape(-1)[:] = growing_season_length_3d(tb).reshape(-1)
    # last year (364 days) exactly
    s_, e_ = ys[-1]
    out["GSL"][-1].reshape(-1)[:] = growing_season_length_3d(
        ta[s_:e_].reshape(1, 364, C)).reshape(-1)
    return out
