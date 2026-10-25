"""Annual ETCCDI indices used by ClimX, computed from daily fields.

Pure-numpy mirror of ref/ClimX/src/indices_xclim.py (the 15 selected indices),
verified against the official xclim pipeline (xclim 0.62.0).
Works on (T, ...) arrays, noleap calendar; year 2100 has 364 days.
"""
from __future__ import annotations

import numpy as np

KG_M2_S_TO_MM_DAY = 86400.0


def year_slices(n_days: int) -> list[tuple[int, int]]:
    sl, idx = [], 0
    for y in range(2015, 2101):
        n = 364 if y == 2100 else 365
        sl.append((idx, idx + n))
        idx += n
    assert idx == n_days, (idx, n_days)
    return sl


def _quantile_xclim_sorted(s: np.ndarray, p: float, alpha: float = 1.0 / 3.0,
                           beta: float = 1.0 / 3.0) -> np.ndarray:
    """Quantile of samples sorted along axis 0, using xclim's exact
    Hyndman-Fan index scheme (xclim.core.utils._nan_quantile, NaN-free).
    s: (n, ...) sorted along axis 0 -> returns s[0].shape."""
    n = s.shape[0]
    v = n * p + (alpha + p * (1.0 - alpha - beta)) - 1.0
    if v <= 0.0:
        return s[0]
    if v >= n - 1:
        return s[-1]
    lo = int(np.floor(v))
    g = v - lo
    return s[lo] * (1.0 - g) + s[lo + 1] * g


def percentile_doy_window(da: np.ndarray, quantile: float, window: int = 5) -> np.ndarray:
    """Percentile per day-of-year over a centered rolling window, exactly as
    xclim.core.calendar.percentile_doy (min_periods=1, continuous series).
    da: (T, ...) -> (365, ...)."""
    T = da.shape[0]
    n_years = T // 365
    half = window // 2
    out = np.empty((365,) + da.shape[1:], dtype=np.float64)
    cap = n_years * window
    for d in range(365):
        n_samples = 0
        buf = np.empty((cap,) + da.shape[1:], dtype=np.float64)
        for y in range(n_years):
            c = y * 365 + d
            lo = max(c - half, 0)
            hi = min(c + half, T - 1)
            buf[n_samples:n_samples + (hi - lo + 1)] = da[lo:hi + 1]
            n_samples += hi - lo + 1
        s = np.sort(buf[:n_samples], axis=0)
        out[d] = _quantile_xclim_sorted(s, quantile)
    return out


def _run_start_pos0(m: np.ndarray) -> np.ndarray:
    """m: (T, ...) bool. 0-based start position of the run containing t
    (0 where no run has started yet)."""
    T = m.shape[0]
    pos = np.arange(T)[:, None]
    prev = np.empty_like(m)
    prev[0] = False
    prev[1:] = m[:-1]
    rs = m & ~prev
    starts = np.where(rs, pos, 0)
    return np.maximum.accumulate(starts, axis=0)


def longest_run_annual(m: np.ndarray) -> np.ndarray:
    """m: (T_year, ...) 0/1 -> (...) longest run of 1s."""
    T = m.shape[0]
    pos = np.arange(T)[:, None]
    start = _run_start_pos0(m.astype(bool))
    runlen = np.where(m, pos - start + 1, 0)
    return runlen.max(axis=0).astype(np.float64)


def windowed_run_sum_annual(m: np.ndarray, min_len: int) -> np.ndarray:
    """m: (T_year, ...) 0/1 -> (...) total days in runs of length >= min_len."""
    T = m.shape[0]
    pos = np.arange(T)[:, None]
    mb = m.astype(bool)
    start = _run_start_pos0(mb)
    runlen = pos - start + 1
    nxt = np.concatenate([mb[1:], np.zeros((1,) + mb.shape[1:], dtype=bool)], axis=0)
    is_end = mb & ~nxt
    total = np.where(is_end & (runlen >= min_len), runlen, 0).sum(axis=0)
    return total.astype(np.float64)


def first_run_ge(m: np.ndarray, min_len: int) -> np.ndarray:
    """m: (T_year, ...) bool -> (...) index of the first day of the first run
    of consecutive True with length >= min_len; T_year if none."""
    T = m.shape[0]
    pos = np.arange(T)[:, None]
    mb = m.astype(bool)
    start = _run_start_pos0(mb)
    runlen = pos - start + 1
    nxt = np.concatenate([mb[1:], np.zeros((1,) + mb.shape[1:], dtype=bool)], axis=0)
    is_end = mb & ~nxt
    full_len = np.where(is_end, runlen, 0)
    valid_end = is_end & (full_len >= min_len)
    cand = np.where(valid_end, start, T)
    return cand.min(axis=0)


def growing_season_length_year(tas_year: np.ndarray, thresh: float = 278.15,
                               window: int = 6, mid_idx: int = 181) -> np.ndarray:
    """GSL for one year, matching xclim growing_season_length (freq='YS',
    mid_date='07-01', op='>='). tas_year: (T, ...) K. Returns (...)."""
    shape = tas_year.shape
    T = shape[0]
    tas_year = tas_year.reshape(T, -1)
    pos = np.arange(T)[:, None]
    cond = tas_year >= thresh
    # start: first run of >=window True, restricted to positions < mid_idx+window-1
    m_start = cond & (pos < mid_idx + window - 1)
    beg = first_run_ge(m_start, window)          # (...)
    has_start = beg < T
    # end: first run of >=window False, in positions >= beg and >= mid_idx
    cold = ~cond
    m_end = cold & (pos >= beg[None, ...]) & (pos >= mid_idx)
    end = first_run_ge(m_end, window)            # (...)
    length = np.where(has_start,
                      np.where(end < T, end - beg, T - beg),
                      0)
    return length.reshape(shape[1:]).astype(np.float64)


def compute_indices(daily: dict[str, np.ndarray], base_daily: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Compute the 15 annual indices.

    daily: var -> (T, lat, lon), T = days 2015-01-01..2100-12-30 (noleap; 2100 -> 364 days).
    base_daily: var -> (10950, lat, lon) over 1961-01-01..1990-12-31.
    Returns: name -> (86, lat, lon).
    """
    T = next(iter(daily.values())).shape[0]
    ys = year_slices(T)
    n_years = len(ys)
    tasmax = daily["tasmax"].astype(np.float64)
    tasmin = daily["tasmin"].astype(np.float64)
    tas = daily.get("tas")
    if tas is None:
        tas = (tasmax + tasmin) / 2.0
    else:
        tas = tas.astype(np.float64)
    pr = daily["pr"].astype(np.float64) * KG_M2_S_TO_MM_DAY  # mm/day

    def _doy_th(a: np.ndarray, q: float, scale: float = 1.0) -> np.ndarray:
        # accept raw base daily series (10950,) or precomputed doy thresholds (365,)
        a = a.astype(np.float64)
        if a.shape[0] == 365:
            return a  # already scaled (mm/day) when precomputed
        return percentile_doy_window(a * scale, q)

    th_wsdi = _doy_th(base_daily["tasmax"], 0.90)
    th_csdi = _doy_th(base_daily["tasmin"], 0.10)
    th_r95p = _doy_th(base_daily["pr"], 0.95, KG_M2_S_TO_MM_DAY)

    lat_lon = tasmax.shape[1:]
    C = int(np.prod(lat_lon))

    def _flat(a: np.ndarray) -> np.ndarray:
        return a.reshape(a.shape[0], C)

    names = ["TXx", "TNn", "SU", "TR", "FD", "ID", "GSL", "WSDI", "CSDI",
             "Rx5day", "CDD", "CWD", "R95p", "R10mm", "SDII"]
    out = {k: np.zeros((n_years,) + lat_lon, dtype=np.float64) for k in names}

    doy = np.arange(T) % 365
    mask_wsdi = tasmax > th_wsdi[doy]
    mask_csdi = tasmin < th_csdi[doy]
    mask_pr95 = pr > th_r95p[doy]
    pr_lt1 = pr < 1.0
    pr_ge1 = pr >= 1.0
    pr_ge10 = pr >= 10.0

    # 5-day cumulative pr over the CONTINUOUS series (trailing window, crosses
    # year boundaries; first 4 days of the whole series are NaN)
    pr5 = np.full_like(pr, np.nan)
    csum = np.cumsum(pr, axis=0)
    pr5[4] = csum[4]
    pr5[5:] = csum[5:] - csum[:-5]

    for y, (s_, e_) in enumerate(ys):
        seg = lambda a: a[s_:e_]
        out["TXx"][y] = seg(tasmax).max(axis=0)
        out["TNn"][y] = seg(tasmin).min(axis=0)
        out["SU"][y] = (seg(tasmax) > 298.15).sum(axis=0)
        out["TR"][y] = (seg(tasmin) > 293.15).sum(axis=0)
        out["FD"][y] = (seg(tasmin) < 273.15).sum(axis=0)
        out["ID"][y] = (seg(tasmax) < 273.15).sum(axis=0)
        out["GSL"][y] = growing_season_length_year(seg(tas)).reshape(lat_lon)
        out["WSDI"][y] = windowed_run_sum_annual(_flat(seg(mask_wsdi)), 6).reshape(lat_lon)
        out["CSDI"][y] = windowed_run_sum_annual(_flat(seg(mask_csdi)), 6).reshape(lat_lon)
        out["Rx5day"][y] = np.nanmax(seg(pr5), axis=0)
        out["CDD"][y] = longest_run_annual(_flat(seg(pr_lt1))).reshape(lat_lon)
        out["CWD"][y] = longest_run_annual(_flat(seg(pr_ge1))).reshape(lat_lon)
        out["R95p"][y] = np.where(seg(mask_pr95), seg(pr), 0.0).sum(axis=0)
        out["R10mm"][y] = seg(pr_ge10).sum(axis=0)
        wet_sum = np.where(seg(pr_ge1), seg(pr), 0.0).sum(axis=0)
        wet_cnt = seg(pr_ge1).sum(axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            out["SDII"][y] = np.where(wet_cnt == 0, 0.0, wet_sum / wet_cnt)
    return out


def base_period_slice(hist_late: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """1961-01-01..1990-12-31 (30*365 days) from hist_late (1932-2014)."""
    s = (1961 - 1932) * 365
    e = (1990 - 1931) * 365
    assert e - s == 30 * 365
    return {k: v[s:e] for k, v in hist_late.items()}
