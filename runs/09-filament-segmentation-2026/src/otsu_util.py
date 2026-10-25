"""Otsu threshold that works with headless/minimal OpenCV builds."""
from __future__ import annotations
import numpy as np


def otsu_threshold(u8: np.ndarray) -> float:
    """Otsu threshold for a uint8 array (grayscale). Returns the threshold t
    such that mask = x > t (cv2.THRESH_BINARY convention: x > t)."""
    hist = np.bincount(u8.ravel(), minlength=256).astype(np.float64)
    total = hist.sum()
    if total == 0:
        return 0.0
    hist /= total
    bins = np.arange(256)
    w0 = np.cumsum(hist)
    m0 = np.cumsum(hist * bins)
    m_total = m0[-1]
    denom = w0 * (1.0 - w0)
    with np.errstate(divide="ignore", invalid="ignore"):
        mu_between = (m_total * w0 - m0) ** 2 / np.where(denom == 0, np.inf, denom)
    mu_between[0] = 0.0
    t = int(np.argmax(mu_between))
    return float(t)
