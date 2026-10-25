"""Offline post-processing tuner.

Inputs (from a completed kernel run, downloaded to output/<run>/):
  - val_gt.npz                      per-stem newline-joined b64 RLE GT (per record, stem-ordered)
  - val_predictions_{unet,mrcnn,merge}.npz  per-stem newline-joined b64 RLE preds
  - val_stems.json
  - submission_{unet,mrcnn,merge}.csv       per-candidate TEST submissions
  - run_report.json

Goal: find the best per-observation combination rule (from the per-candidate
instances) on the val set, then apply it to the test CSVs to build a new
submission CSV.  Also supports per-candidate min_px filtering and instance
merging.

Usage:
  python src/tune_postprocess.py output/v1-gpu --out tuned_submission.csv
"""
from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

IMG = 2048


def _decode(rle_b64: str) -> np.ndarray:
    # Raw RLE counts (new format) first; base64 fallback for legacy npz/csv.
    # NOTE: this Cython pycocotools silently returns a corrupt mask for bad
    # input (e.g. empty counts from b64-decoding a bool array), so guard the
    # type explicitly.
    if not isinstance(rle_b64, str):
        raise TypeError(f"_decode expects an RLE str, got {type(rle_b64).__name__}")
    from pycocotools import mask as mu
    try:
        return (mu.decode({"size": [IMG, IMG], "counts": rle_b64.encode("latin-1")}) > 0)
    except Exception:
        counts = base64.b64decode(rle_b64)
        return (mu.decode({"size": [IMG, IMG], "counts": counts}) > 0)


def _encode(mask: np.ndarray) -> str:
    # Official submission format: raw pycocotools compressed RLE counts.
    from pycocotools import mask as mu
    rle = mu.encode(np.asfortranarray(mask.astype(np.uint8)))
    counts = rle["counts"]
    if isinstance(counts, bytes):
        return counts.decode("latin-1")
    return str(counts)


def load_npz_rles(path: Path) -> tuple[list[str], dict[str, list[str]]]:
    d = np.load(path, allow_pickle=False)
    stems = [str(s) for s in d["stems"]]
    out = {}
    for i, s in enumerate(stems):
        v = str(d[f"pred_{i:03d}"])
        out[s] = v.split("\n") if v else []
    return stems, out


def load_gt(path: Path) -> tuple[list[str], dict[str, list[np.ndarray]]]:
    d = np.load(path, allow_pickle=False)
    stems = [str(s) for s in d["stems"]]
    out = {}
    for i, s in enumerate(stems):
        v = str(d[f"gt_{i:03d}"])
        out[s] = [_decode(r) for r in v.split("\n")] if v else []
    return stems, out


def pq_obs(gt_masks: list[np.ndarray], pred_masks: list[np.ndarray], thr: float = 0.5) -> dict:
    tp_sum, tp, fp, fn = 0.0, 0, 0, 0
    n_gt, n_pred = len(gt_masks), len(pred_masks)
    if n_gt and n_pred:
        gtf = np.stack(gt_masks).reshape(n_gt, -1).astype(np.float32)
        ptf = np.stack(pred_masks).reshape(n_pred, -1).astype(np.float32)
        inter = gtf @ ptf.T
        union = gtf.sum(1, keepdims=True) + ptf.sum(1, keepdims=True).T - inter
        iou = np.where(union > 0, inter / np.maximum(union, 1), 0.0)
        hit = iou > thr
        tp_iou = iou[hit]
        tp = int(hit.sum())
        tp_sum = float(tp_iou.sum())
        fp = int((hit.sum(0) == 0).sum())
        fn = int((hit.sum(1) == 0).sum())
    else:
        fp, fn = n_pred, n_gt
    den = tp + 0.5 * fp + 0.5 * fn
    return {"pq": tp_sum / den if den else 0.0, "tp": tp, "fp": fp, "fn": fn, "tp_sum": tp_sum}


def merge_masks(masks: list[np.ndarray]) -> list[np.ndarray]:
    masks = [m for m in masks if m.any()]
    if len(masks) < 2:
        return masks
    H, W = masks[0].shape
    L = np.zeros((H, W), np.int32)
    for i, m in enumerate(masks):
        L[m] = i + 1
    n = len(masks)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i, m in enumerate(masks):
        for lab in np.unique(L[m]):
            j = int(lab) - 1
            if j != i:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[ri] = rj
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return [masks[g[0]] if len(g) == 1 else np.logical_or.reduce([masks[i] for i in g])
            for g in groups.values()]


def apply_rule(preds_by_model: dict[str, list[np.ndarray]], rule: str) -> list[np.ndarray]:
    """rule in {union, unet, mrcnn, merge, union_nosmall, best_iou}."""
    unet = preds_by_model.get("unet", [])
    mrcnn = preds_by_model.get("mrcnn", [])
    merged = preds_by_model.get("merge", [])
    if rule == "unet":
        ms = unet
    elif rule == "mrcnn":
        ms = mrcnn
    elif rule == "merge":
        ms = merged
    else:
        ms = unet + mrcnn
        if rule == "union_nosmall":
            ms = [m for m in ms if m.sum() >= 64]
        ms = merge_masks(ms)
    if rule == "best_iou":
        # assign each mrcnn instance to the best-IoU unet instance (replace);
        # keep mrcnn instances with no good unet partner
        out = list(unet)
        for p in mrcnn:
            best_iou, best_idx = 0.0, -1
            for i, u in enumerate(unet):
                iou = _iou(u, p)
                if iou > best_iou:
                    best_iou, best_idx = iou, i
            if best_idx >= 0 and best_iou > 0.3:
                out[best_idx] = p
            elif best_idx < 0:
                out.append(p)
        return out
    return ms


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = int(np.logical_and(a, b).sum())
    union = int(np.logical_or(a, b).sum())
    return inter / union if union else 0.0


def main() -> None:
    run_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "output/v1-gpu")
    out_path = Path(sys.argv[2]) if len(sys.argv) > 2 else run_dir / "tuned_submission.csv"

    stems, gt = load_gt(run_dir / "val_gt.npz")
    cands = {}
    for name in ("unet", "mrcnn", "merge"):
        p = run_dir / f"val_predictions_{name}.npz"
        if p.exists():
            _, cands[name] = load_npz_rles(p)
    print(f"val stems={len(stems)} candidates={ {k: len(v) for k, v in cands.items()} }")

    rules = ["unet", "mrcnn", "merge", "union", "union_nosmall", "best_iou"]
    results = {}
    for rule in rules:
        tp_sum = tp = fp = fn = 0
        for s in stems:
            preds = {k: [r for r in v.get(s, [])] for k, v in cands.items()}
            pixel = {k: [_decode(r) for r in rs] for k, rs in preds.items()}
            out = apply_rule(pixel, rule)
            r = pq_obs(gt[s], out)
            tp_sum += r["tp_sum"]; tp += r["tp"]; fp += r["fp"]; fn += r["fn"]
        den = tp + 0.5 * fp + 0.5 * fn
        results[rule] = {"pq": tp_sum / den if den else 0.0, "tp": tp, "fp": fp, "fn": fn}
    for rule, r in sorted(results.items(), key=lambda kv: -kv[1]["pq"]):
        print(f"  {rule:14s} PQ={r['pq']:.4f} tp={r['tp']} fp={r['fp']} fn={r['fn']}")
    best = max(results, key=lambda k: results[k]["pq"])
    print(f"best rule: {best} PQ={results[best]['pq']:.4f}")

    # apply best rule to the test CSVs
    if best in ("unet", "mrcnn", "merge"):
        df = pd.read_csv(run_dir / f"submission_{best}.csv")
    else:
        per_stem: dict[str, dict[str, list[str]]] = {}
        for name in ("unet", "mrcnn"):
            dfn = pd.read_csv(run_dir / f"submission_{name}.csv")
            for fid, rle in zip(dfn["filament_id"], dfn["segmentation_rle"]):
                per_stem.setdefault(str(fid).rsplit("_", 1)[0], {}).setdefault(name, []).append(str(rle))
        out_rows = []
        for stem in sorted(per_stem):
            by_model = per_stem[stem]
            pixel = {name: [_decode(r) for r in by_model.get(name, [])] for name in ("unet", "mrcnn")}
            if best == "best_iou":
                rule_masks = apply_rule(pixel, "best_iou")
            elif best == "union":
                rule_masks = merge_masks(pixel["unet"] + pixel["mrcnn"])
            else:  # union_nosmall
                rule_masks = merge_masks([m for m in pixel["unet"] + pixel["mrcnn"] if m.sum() >= 64])
            for k, m in enumerate(rule_masks, 1):
                out_rows.append({"filament_id": f"{stem}_{k}", "segmentation_rle": _encode(m)})
        df = pd.DataFrame(out_rows, columns=["filament_id", "segmentation_rle"])
    df.to_csv(out_path, index=False)
    print(f"wrote {out_path} rows={len(df)}")


if __name__ == "__main__":
    main()
