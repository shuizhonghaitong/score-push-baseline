"""Extended per-trial features: fine spectral bins, temporal dynamics,
CSP features. EEG only."""
from __future__ import annotations

import numpy as np

FS = 250
CHANNELS = ['Fz', 'C3', 'Cz', 'C4', 'PO7', 'Pz', 'PO8', 'Oz']

def _psd(x: np.ndarray) -> tuple:
    n = x.shape[0]
    xw = (x - x.mean(axis=0)) * np.hanning(n)[:, None]
    spec = np.abs(np.fft.rfft(xw, axis=0)) ** 2
    freqs = np.fft.rfftfreq(n, d=1.0 / FS)
    return spec, freqs

def band_power_matrix(spec, freqs, lo, hi):
    m = (freqs >= lo) & (freqs < hi)
    return spec[m, :].mean(axis=0)

FINE_BINS = [(f, f + 1.0) for f in range(1, 30)]          # 1..29 Hz
SUBWINDOWS = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]     # seconds in cue


def _bp(x: np.ndarray, lo: float, hi: float) -> np.ndarray:
    """Band power per channel for a segment x (n, ch)."""
    s, f = _psd(x)
    m = (f >= lo) & (f < hi)
    return s[m, :].mean(axis=0)


def base_features(trial: dict) -> np.ndarray:
    cue = trial['cue']
    base = trial['baseline']
    nch = cue.shape[1]
    feat = []
    # fine-bin log power over full 5s cue (per channel)
    for (lo, hi) in FINE_BINS:
        feat.append(np.log1p(_bp(cue, lo, hi)))
    # subwindow log power for mu & beta (central 3 ch averaged)
    for (a, b) in SUBWINDOWS:
        for bnd in ((8, 12), (13, 30)):
            feat.append(np.log1p(_bp(cue[a * FS:b * FS], bnd[0], bnd[1])[[1, 2, 3]].mean()))
    # ERD vs 3s pre-cue baseline, fine bins, all channels
    if base.shape[0] >= 2 * FS:
        for (lo, hi) in FINE_BINS:
            feat.append(np.log((_bp(cue, lo, hi) + 1e-8) / (_bp(base, lo, hi) + 1e-8)))
    else:
        feat.extend([0.0] * (len(FINE_BINS) * nch))
    # relative power full-cue (channel averaged)
    total = np.stack([_bp(cue, f, f + 1) for f in range(1, 30)]).sum(axis=0) + 1e-8
    for (lo, hi) in FINE_BINS:
        feat.append(_bp(cue, lo, hi).mean() / total.mean())
    # dynamics: (second half - first half)/first half per fine bin (central)
    half = cue.shape[0] // 2
    for (lo, hi) in FINE_BINS:
        p1 = _bp(cue[:half], lo, hi)
        p2 = _bp(cue[half:2 * half], lo, hi)
        feat.append((p2[[1, 2, 3]] - p1[[1, 2, 3]]) / (p1[[1, 2, 3]] + 1e-8))
    # artifact proxies: raw variance & amplitude per channel, delta power
    feat.append(cue.var(axis=0))
    feat.append(np.abs(cue).mean(axis=0))
    feat.append(_bp(cue, 0.5, 4))
    # central vs posterior contrast, fine bins
    for (lo, hi) in FINE_BINS:
        p = _bp(cue, lo, hi)
        feat.append(p[[1, 2, 3]].sum() - p[[4, 5, 6, 7]].sum())
    return np.concatenate([np.atleast_1d(np.asarray(x, dtype=np.float32)) for x in feat])


# ----------------------------- CSP ---------------------------------

def csp_transform(X: np.ndarray, y: np.ndarray, n_filters: int = 8,
                  lam: float = 1e-4) -> np.ndarray:
    """X: (n, n_ch, n_t) -> features (n, n_filters) log variances.

    Standard two-class CSP on mean per-trial covariances, regularized.
    """
    n, c, t = X.shape
    Xc = X - X.mean(axis=2, keepdims=True)
    covs = np.einsum('nct,ndt->ncd', Xc, Xc) / (t - 1)
    C0 = covs[y == 0].mean(axis=0)
    C1 = covs[y == 1].mean(axis=0)
    C = (C0 + C1) / 2.0
    Creg = C + (lam / 2) * np.trace(C) / c * np.eye(c)
    D, V = np.linalg.eigh(Creg)
    W = V @ np.diag(1.0 / np.sqrt(D + 1e-12)) @ V.T
    Xw = np.einsum('nct,mc->ntm', Xc, W)           # (n, t, c)
    S0 = (Xw[y == 0] ** 2).mean(axis=(0, 1))
    S1 = (Xw[y == 1] ** 2).mean(axis=(0, 1))
    S = (S0 + S1) / 2.0 + lam * np.eye(c)
    M = V.T @ np.linalg.inv(S) @ V
    D2, V2 = np.linalg.eigh(M)
    idx = np.argsort(D2)[::-1]
    A = V @ V2[:, idx]
    Y = np.einsum('nct,mc->ntm', Xc, A[:, :n_filters])
    return np.log(np.var(Y, axis=1))
