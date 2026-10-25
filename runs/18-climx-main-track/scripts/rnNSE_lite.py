"""Lite-grid (12x18) rnNSE, mirroring ref/ClimX/rnNSE.py scoring logic.

Indices: (86, 12, 18) arrays in SUB_COLUMNS order.
"""
from __future__ import annotations

import numpy as np

from climx_common import ar6_mask_full, lite_latlon, SUB_COLUMNS

_VAR_EPS_SQ = 1e-6


def _lite_mask() -> np.ndarray:
    full = ar6_mask_full()  # (192, 288)
    # block centers: rows 16i+8, cols 16j+8
    rows = 16 * np.arange(12) + 8
    cols = 16 * np.arange(18) + 8
    return full[rows[:, None], cols[None, :]].astype(np.int16)


def lite_lat() -> np.ndarray:
    lat, _ = lite_latlon()
    return lat


def score_lite(gt: dict[str, np.ndarray], pred: dict[str, np.ndarray], names=None):
    """gt/pred: name -> (86, 12, 18). Returns (S, per_name dict)."""
    names = names or SUB_COLUMNS
    region_mask = _lite_mask()
    lats = lite_lat()
    cos_w = np.cos(np.deg2rad(lats))
    unique_regions = np.unique(region_mask[region_mask >= 0])
    per_var = {}
    for col in names:
        g = gt[col].astype(np.float64)
        p = pred[col].astype(np.float64)
        if not np.isfinite(p).all():
            raise ValueError(f"non-finite in pred {col}")
        var_cell = g.var(axis=0, ddof=0)
        mse_cell = ((g - p) ** 2).mean(axis=0)
        valid = var_cell > _VAR_EPS_SQ
        r2 = np.where(valid, 1.0 - mse_cell / np.maximum(var_cell, _VAR_EPS_SQ), np.nan)
        nnse = np.where(valid, r2 / (2.0 - r2), np.nan)
        region_scores = []
        for rc in unique_regions:
            in_region = (region_mask == rc) & valid
            w = cos_w[:, None] * in_region
            total_w = w.sum()
            if total_w < 1e-12:
                continue
            val = nnse[in_region]
            if val.size == 0:
                continue
            region_scores.append(np.average(val, weights=w[in_region]))
        per_var[col] = float(np.mean(region_scores)) if region_scores else 0.0
    S = float(np.mean(list(per_var.values())))
    return S, per_var
