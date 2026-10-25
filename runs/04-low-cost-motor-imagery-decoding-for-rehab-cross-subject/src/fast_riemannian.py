"""Vectorized FB-RTS features (matches run-03 riemannian_features closely,
using a fixed Ledoit-Wolf-style shrinkage instead of per-trial LedoitWolf)."""
import numpy as np
from scipy import signal

FS = 250.0
BANDS = [(4, 8), (8, 12), (12, 16), (16, 20), (20, 25), (25, 30)]
_BUTTER = {}

def _butter(l, h, order=4):
    key = (float(l), float(h))
    if key not in _BUTTER:
        nyq = FS / 2
        _BUTTER[key] = signal.butter(order, [max(0.01, l / nyq), min(0.99, h / nyq)], btype='bandpass')
    return _BUTTER[key]

def band_filter(x, l, h):
    b, a = _butter(l, h)
    axis = 1 if x.ndim == 3 else 0
    return signal.filtfilt(b, a, x, axis=axis)

def _matpow8(A, p):
    w, v = np.linalg.eigh(A)
    return v @ np.diag(np.maximum(w, 1e-12) ** p) @ v.T

def _matlog8(A):
    w, v = np.linalg.eigh(A)
    return v @ np.diag(np.log(np.maximum(w, 1e-12))) @ v.T

def _triu_vec(L):
    idx = np.triu_indices(8)
    v = L[idx]
    v[1:] *= np.sqrt(2.0)
    return v

def fast_riemannian_features(cue, bands=BANDS, ref=None, lam=0.05):
    """cue: (n, t, c) CAR+notch processed. Returns (feats, ref)."""
    n, t, c = cue.shape
    if ref is None:
        refs = []
        for (l, h) in bands:
            cf = band_filter(cue, l, h)
            cov = np.einsum('nta,ntb->nab', cf, cf) / (t - 1)
            refs.append(cov.mean(axis=0))
        ref = np.stack(refs)
    feats = []
    for (l, h), r in zip(bands, ref):
        cf = band_filter(cue, l, h)
        cov = np.einsum('nta,ntb->nab', cf, cf) / (t - 1)
        mu = np.trace(cov, axis1=1, axis2=2) / c
        cov = (1 - lam) * cov + lam * mu[:, None, None] * np.eye(c)[None]
        G = _matpow8(r, -0.5)
        W = np.einsum('ab,nbc,cd->nad', G, cov, G)
        w, v = np.linalg.eigh(W)
        L = (v * np.log(np.maximum(w, 1e-12))[:, None, :]) @ v.transpose(0, 2, 1)
        feats.append(np.stack([_triu_vec(L[i]) for i in range(n)]))
    return np.concatenate(feats, axis=1), ref
