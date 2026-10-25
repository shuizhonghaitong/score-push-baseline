#!/usr/bin/env python3
"""Per-stem failure analysis of a run's val predictions vs GT (RLE dumps)."""
import json, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from tune_postprocess import _decode, load_npz_rles, load_gt  # noqa: E402


def best_iou(a: np.ndarray, masks: list[np.ndarray]) -> float:
    if not masks:
        return 0.0
    best = 0.0
    for m in masks:
        i = int(np.logical_and(a, m).sum())
        u = int(np.logical_or(a, m).sum())
        v = i / u if u else 0.0
        if v > best:
            best = v
    return float(best)


def clip_idx(b: float, scale: int, max_idx: int) -> int:
    return max(0, min(max_idx, int(b * scale)))


def main() -> None:
    run_dir = Path(sys.argv[1])
    model = sys.argv[2] if len(sys.argv) > 2 else "unet"
    pred_npz = run_dir / f"val_predictions_{model}.npz"
    stems, pred = load_npz_rles(pred_npz)
    stems_gt, gt = load_gt(run_dir / "val_gt.npz")
    n_gt = n_pred = n_tp = n_fn = n_fp = 0
    tp_hist = [0, 0, 0, 0]   # [0.5,0.6,0.7,0.8,1.0]
    fn_hist = [0, 0, 0, 0]   # [0,0.1,0.2,0.3,0.5)
    gt_best, pr_best = [], []
    for s in stems:
        gm = gt.get(s, [])
        pm = [_decode(r) for r in pred.get(s, [])]
        for a in gm:
            b = best_iou(a, pm)
            n_gt += 1
            gt_best.append(b)
            if b > 0.5:
                n_tp += 1
                tp_hist[clip_idx(b - 0.5, 20, 3)] += 1
            else:
                n_fn += 1
                fn_hist[clip_idx(b, 10, 3)] += 1
        for a in pm:
            b = best_iou(a, gm)
            n_pred += 1
            pr_best.append(b)
            if b <= 0.5:
                n_fp += 1
    gt_best = np.array(gt_best); pr_best = np.array(pr_best)
    out = {
        "model": model, "stems": len(stems),
        "gt_instances": n_gt, "pred_instances": n_pred,
        "tp": n_tp, "fn": n_fn, "fp": n_fp,
        "tp_iou_hist_[0.5-0.6-0.7-0.8-1.0]": tp_hist,
        "fn_best_iou_hist_[0-0.1-0.2-0.3-0.5)": fn_hist,
        "gt_best_iou_p25_50_75": [float(np.percentile(gt_best, p)) for p in (25, 50, 75)],
        "pred_best_iou_p25_50_75": [float(np.percentile(pr_best, p)) for p in (25, 50, 75)],
        "pred_iou_lt0.1_frac": float((pr_best < 0.1).mean()),
    }
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
