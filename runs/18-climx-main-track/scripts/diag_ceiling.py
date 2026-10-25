"""Ceiling analysis: score of a heavily-smoothed version of the holdout's own GT
(approximates a perfect scenario-mean predictor), plus per-cell var distribution."""
import sys, numpy as np
sys.path.insert(0, 'scripts')
import train_v4 as t4
from climx_common import SUB_COLUMNS
from scipy.ndimage import gaussian_filter
from pathlib import Path
from rnNSE_lite import score_lite

ssps = t4.SSPS
gt = {s: t4.gt_indices_npz(Path("models/cache"), s) for s in ssps}

for ho in ssps:
    g = {k: gt[ho][k][:85] for k in SUB_COLUMNS}
    out = {}
    for sigma in (5, 10, 15, 20):
        out[sigma] = {k: gaussian_filter(g[k], (sigma, 0.01, 0.01), mode="nearest") for k in SUB_COLUMNS}
    print(f"== holdout {ho}")
    for sigma, pred in out.items():
        pred86 = {k: np.concatenate([p, p[-1:]], axis=0) for k, p in pred.items()}
        S, per = score_lite(gt[ho], pred86)
        print(f"  smooth{sigma}: S={S:+.4f}  " + " ".join(f"{k[:4]}:{per[k]:+.2f}" for k in SUB_COLUMNS))
    # per-cell var distribution (85y)
    print("  per-cell var (q25/med/q75):")
    for k in SUB_COLUMNS:
        var = g[k].var(axis=0)
        print(f"    {k:8s} q25={np.quantile(var,0.25):10.3f} med={np.median(var):10.3f} q75={np.quantile(var,0.75):10.3f} frac<1e-2={ (var<1e-2).mean():.2f}")
