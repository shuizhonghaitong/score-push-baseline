"""Local Panoptic-Quality evaluator faithful to the official Self-Evaluation notebook.

Faithful to:
- get_overlap_df / process_entry: GT grouped by annotator-image "<annotator>-<image_id>",
  predictions grouped by image_id; one row per annotator-image.
- get_pq_score: TP pair iff IoU > 0.5 (strict); a prediction may be TP against multiple
  GTs and vice versa; FP = prediction with no hit in the row; FN = GT with no hit in the row.
- PQ = sum(TP IoU) / (|TP| + 0.5*|FP| + 0.5*|FN|) across all rows.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from pycocotools import mask as mask_util


def _decode_one(counts, height, width):
    """Robust single RLE decode across pycocotools versions / count encodings."""
    if counts is None:
        return np.zeros((height, width), dtype=np.uint8)
    candidates = []
    candidates.append(counts)
    if isinstance(counts, str):
        try:
            import base64
            candidates.append(base64.b64decode(counts))
        except Exception:
            pass
    last = None
    for cand in candidates:
        try:
            m = mask_util.decode({"size": [height, width], "counts": cand})
            return (m > 0).astype(np.uint8)
        except Exception as e:
            last = e
    raise last


def rles_to_layers(rles: list[str], height: int = 2048, width: int = 2048) -> np.ndarray:
    if not rles:
        return np.zeros((0, height, width), dtype=np.float32)
    stack = np.stack([_decode_one(r, height, width) for r in rles], axis=0)
    return stack.astype(np.float32)


def overlap_matrices(gt_layers: np.ndarray, pred_layers: np.ndarray):
    n_gt = gt_layers.shape[0]
    n_pred = pred_layers.shape[0]
    if n_gt == 0 or n_pred == 0:
        return (
            np.zeros((n_gt, n_pred), dtype=np.float64),
            np.zeros((n_gt, n_pred), dtype=np.float64),
        )
    gt_flat = gt_layers.reshape(n_gt, -1)
    pred_flat = pred_layers.reshape(n_pred, -1)
    inter = gt_flat @ pred_flat.T
    gt_area = gt_flat.sum(axis=1, keepdims=True)
    pred_area = pred_flat.sum(axis=1, keepdims=True).T
    union = gt_area + pred_area - inter
    iou = np.where(union > 0, inter / np.where(union > 0, union, 1), 0.0)
    dice = np.where(union > 0, 2.0 * inter / np.where(union > 0, union, 1), 0.0)
    return iou, dice


def pq_score(gt_df: pd.DataFrame, pred_df: pd.DataFrame, height=2048, width=2048,
             iou_threshold=0.5, verbose=False):
    """gt_df/pred_df: columns filament_id, segmentation_rle. Returns (pq, stats dict)."""
    gt_aid = gt_df["filament_id"].astype(str).str.split("_", n=1).str[0]
    pred_iid = pred_df["filament_id"].astype(str).str.split("_", n=1).str[0]
    pred_by_image = pred_df.groupby(pred_iid, sort=False)["segmentation_rle"].apply(list).to_dict()

    tp_sum = 0.0
    tp = fp = fn = 0
    per_image = {}
    for aid, sub in gt_df.groupby(gt_aid, sort=False):
        image_id = aid.split("-", 1)[1]
        gt_layers = rles_to_layers(sub["segmentation_rle"].tolist(), height, width)
        pred_rles = pred_by_image.get(image_id, [])
        pred_layers = rles_to_layers(pred_rles, height, width)
        n_gt, n_pred = gt_layers.shape[0], pred_layers.shape[0]
        if n_gt == 0:
            fp += n_pred
            continue
        if n_pred == 0:
            fn += n_gt
            per_image[image_id] = (0.0, n_gt, 0)
            continue
        iou, _ = overlap_matrices(gt_layers, pred_layers)
        hit = iou > iou_threshold
        tp_iou = iou[hit]
        tp += int(hit.sum())
        tp_sum += float(tp_iou.sum())
        fp += int((hit.sum(axis=0) == 0).sum())
        fn += int((hit.sum(axis=1) == 0).sum())
        per_image[image_id] = (float(tp_iou.mean()) if tp_iou.size else 0.0, n_gt, n_pred)
    denom = tp + 0.5 * fp + 0.5 * fn
    pq = tp_sum / denom if denom > 0 else 0.0
    stats = {"pq": pq, "tp": tp, "fp": fp, "fn": fn, "tp_iou_sum": tp_sum,
             "mean_tp_iou": (tp_sum / tp) if tp else 0.0, "n_images": len(per_image)}
    if verbose:
        print(stats)
    return pq, stats


if __name__ == "__main__":
    import sys
    gt = pd.read_csv(sys.argv[1])
    pred = pd.read_csv(sys.argv[2])
    pq, stats = pq_score(gt, pred, verbose=True)
    print(f"PQ = {pq:.4f}")
