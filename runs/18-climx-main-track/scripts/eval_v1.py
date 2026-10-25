"""Evaluate a v1 prediction set against a scenario's GT indices (lite rnNSE)."""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import netCDF4 as nc

sys.path.insert(0, str(Path(__file__).resolve().parent))
from climx_common import TARGET_VARS, lite_file
from climx_indices import base_period_slice
from climx_indices_official import compute_indices_official
from rnNSE_lite import score_lite


def gt_indices(scenario: str) -> dict[str, np.ndarray]:
    ds = nc.Dataset(str(lite_file(scenario)))
    daily = {v: np.asarray(ds.variables[v][0], dtype=np.float64) for v in ["tas", "tasmax", "tasmin", "pr"]}
    ds.close()
    dslate = nc.Dataset(str(lite_file("hist_late")))
    base = base_period_slice({v: np.asarray(dslate.variables[v][0], dtype=np.float64) for v in ["tas", "tasmax", "tasmin", "pr"]})
    dslate.close()
    return compute_indices_official(daily, base)


def main(pred_path: str, scenario: str):
    t0 = time.time()
    run = Path(__file__).resolve().parents[1]
    p = np.load(run / pred_path)
    need = ["tas", "tasmax", "tasmin", "pr"]
    daily_pred = {v: p[v].astype(np.float64) for v in need if v in p.files}
    dslate = nc.Dataset(str(lite_file("hist_late")))
    base = base_period_slice({v: np.asarray(dslate.variables[v][0], dtype=np.float64) for v in ["tas", "tasmax", "tasmin", "pr"]})
    dslate.close()
    idx_pred = compute_indices_official(daily_pred, base)
    idx_gt = gt_indices(scenario)
    S, per = score_lite(idx_gt, idx_pred)
    Sg, perg = score_lite(idx_gt, idx_gt)
    print(f"== {scenario} (pred from {pred_path})  total={S:.5f} (self={Sg:.5f})")
    for k in per:
        print(f"  {k:8s} {per[k]:+.4f}")
    print(f"elapsed {time.time()-t0:.0f}s")
    return S, per


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
