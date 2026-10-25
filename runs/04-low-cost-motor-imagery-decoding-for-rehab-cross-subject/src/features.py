"""Feature extractors (ML-track compliant: no neural networks)."""
import numpy as np
from scipy import signal
from sklearn.covariance import LedoitWolf
from common import FS

BANDS = [(4, 8), (8, 12), (12, 16), (16, 20), (20, 25), (25, 30)]
BANDS7 = [(4, 6), (6, 8), (8, 10), (10, 12), (12, 16), (16, 20), (20, 30)]
BUTTER = {}
def _butter(l, h, order=4):
    key = (float(l), float(h))
    if key not in BUTTER:
        nyq = FS / 2
        BUTTER[key] = signal.butter(order, [max(0.01, l / nyq), min(0.99, h / nyq)], btype='bandpass')
    return BUTTER[key]

def band_filter(x, l, h):
    b, a = _butter(l, h)
    axis = 1 if x.ndim == 3 else 0
    return signal.filtfilt(b, a, x, axis=axis)

def band_power_features(cue, base, bands=BANDS, off=0.0):
    if off > 0:
        k = int(off * FS)
        cue = cue[:, k:, :]
    """Log band power, cue absolute + relative to baseline. (N, 4*nbands)"""
    N = cue.shape[0]
    feats = []
    for (l, h) in bands:
        cf = band_filter(cue, l, h)
        bf = band_filter(base, l, h)
        pc = np.log(np.maximum((cf ** 2).mean(axis=1), 1e-12))  # (N, 8)
        pb = np.log(np.maximum((bf ** 2).mean(axis=1), 1e-12))
        feats.append(pc)
        feats.append(np.log(np.maximum((cf ** 2).mean(axis=1), 1e-12) / np.maximum((bf ** 2).mean(axis=1), 1e-12)))
    return np.concatenate(feats, axis=1)

def _spdm(cue):
    """Spatio-temporal covariance (N, 8, 8)."""
    return np.array([np.cov(x.T) for x in cue])

def _matrix_log(A):
    w, v = np.linalg.eigh(A)
    return v @ np.diag(np.log(np.maximum(w, 1e-12))) @ v.T

def riemannian_features(cue, bands=BANDS, ref=None, shrink=True, off=0.0):
    if off > 0:
        k = int(off * FS)
        cue = cue[:, k:, :]
    """FB-RTS: per-band band-filtered covariances projected onto log-Euclidean
    tangent space at reference mean. Returns (feats, ref)."""
    est = LedoitWolf()
    covs = []
    for (l, h) in bands:
        cf = band_filter(cue, l, h)
        covs.append(np.array([est.fit(x).covariance_ for x in cf]))
    if ref is None:
        ref = np.stack([np.mean(c, axis=0) for c in covs])
    feats = []
    for ci, r in zip(covs, ref):
        Ginv_sqrt = _matrix_op(r, -0.5)
        f = []
        for C in ci:
            W = Ginv_sqrt @ C @ Ginv_sqrt
            L = _matrix_log(W)
            idx = np.triu_indices(8)
            v = L[idx]
            v[1:] *= np.sqrt(2.0)
            f.append(v)
        feats.append(np.stack(f))
    return np.concatenate(feats, axis=1), ref

def _matrix_op(A, p):
    w, v = np.linalg.eigh(A)
    return v @ np.diag(w ** p) @ v.T

def csp_features(cue, y, bands=((8,12),(12,16)), k=4, off=0.0):
    if off > 0:
        k2 = int(off * FS)
        cue = cue[:, k2:, :]
    """CSP spatial filters per band; diagonal-bandpower features. Returns (feats, filters)."""
    est = LedoitWolf()
    filters = []
    feats = []
    for (l, h) in bands:
        cf = band_filter(cue, l, h)
        C1 = np.mean([np.cov(x.T) for x in cf[y == 1]], axis=0) if np.sum(y == 1) > 2 else np.eye(8)
        C0 = np.mean([np.cov(x.T) for x in cf[y == 0]], axis=0) if np.sum(y == 0) > 2 else np.eye(8)
        C1, C0 = est.fit(C1).covariance_, est.fit(C0).covariance_
        w, V = np.linalg.eigh(np.linalg.pinv(C0) @ C1)
        order = np.argsort(w)
        V = V[:, order]
        F = np.concatenate([V[:, -k:], V[:, :k]], axis=1)  # (8, 2k)
        filters.append(F)
        proj = [x @ F for x in cf]
        d = np.log(np.maximum(np.array([np.diag(np.cov(p.T)) for p in proj]), 1e-12))
        feats.append(d)
    return np.concatenate(feats, axis=1), filters

def csp_transform(cue, filters, bands=((8,12),(12,16)), off=0.0):
    if off > 0:
        k2 = int(off * FS)
        cue = cue[:, k2:, :]
    feats = []
    for (l, h), F in zip(bands, filters):
        cf = band_filter(cue, l, h)
        proj = [x @ F for x in cf]
        d = np.log(np.maximum(np.array([np.diag(np.cov(p.T)) for p in proj]), 1e-12))
        feats.append(d)
    return np.concatenate(feats, axis=1)

def timdomain_features(cue, l=4.0, h=30.0, fs_out=50.0, off=0.0):
    """Bandpass + decimate: (N, 8, T/fs_out*...)."""
    if off > 0:
        k2 = int(off * FS)
        cue = cue[:, k2:, :]
    xf = band_filter(cue, l, h)
    r = int(FS / fs_out)
    return xf[:, ::r, :].reshape(xf.shape[0], -1)
