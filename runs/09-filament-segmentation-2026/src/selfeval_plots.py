"""Reproduce the official Self-Evaluation plots (IoU/Dice distributions, m2n)
for a (gt_df, pred_df) pair, and save PNGs + a summary JSON.

Usage: python selfeval_plots.py --gt gt.csv --pred pred.csv --outdir plots
"""
from __future__ import annotations

import argparse
import base64
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from evaluate import _decode_counts


def rles_to_layers(rles, height=2048, width=2048):
    if not rles:
        return np.zeros((0, height, width), dtype=np.float32)
    return np.stack([_decode_counts(r, height, width) for r in rles], axis=0).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", type=Path, required=True)
    ap.add_argument("--pred", type=Path, required=True)
    ap.add_argument("--outdir", type=Path, default=Path("plots"))
    args = ap.parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    gt_df = pd.read_csv(args.gt)
    pred_df = pd.read_csv(args.pred)
    gt_aid = gt_df["filament_id"].astype(str).str.split("_", n=1).str[0]
    pred_iid = pred_df["filament_id"].astype(str).str.split("_", n=1).str[0]
    pred_by = pred_df.groupby(pred_iid, sort=False)["segmentation_rle"].apply(list).to_dict()

    iou_scores, dice_scores = [], []
    m2n_gt, m2n_pred = Counter(), Counter()
    pq_rows = []
    for aid, sub in gt_df.groupby(gt_aid, sort=False):
        image_id = aid.split("-", 1)[1]
        gt = rles_to_layers(sub["segmentation_rle"].tolist())
        pred = rles_to_layers(pred_by.get(image_id, []))
        n_gt, n_pred = gt.shape[0], pred.shape[0]
        if n_gt == 0 or n_pred == 0:
            continue
        gtf, ptf = gt.reshape(n_gt, -1), pred.reshape(n_pred, -1)
        inter = gtf @ ptf.T
        union = gtf.sum(1, keepdims=True) + ptf.sum(1, keepdims=True).T - inter
        iou = np.where(union > 0, inter / np.maximum(union, 1), 0.0)
        dice = np.where(union > 0, 2 * inter / np.maximum(union, 1), 0.0)
        iou_scores.extend(iou[iou > 0].tolist())
        dice_scores.extend(dice[dice > 0].tolist())
        hit_pos = iou > 0
        for i in range(n_gt):
            m2n_gt[int(hit_pos[i].sum())] += 1
        for j in range(n_pred):
            m2n_pred[int(hit_pos[:, j].sum())] += 1
        hit = iou > 0.5
        ti = iou[hit]
        fp = int((hit.sum(0) == 0).sum())
        fn = int((hit.sum(1) == 0).sum())
        pq_rows.append((float(ti.sum()) if ti.size else 0.0, int(hit.sum()), fp, fn))

    tp = sum(r[1] for r in pq_rows)
    fp = sum(r[2] for r in pq_rows)
    fn = sum(r[3] for r in pq_rows)
    pq = sum(r[0] for r in pq_rows) / (tp + 0.5 * fp + 0.5 * fn) if (tp + 0.5 * fp + 0.5 * fn) else 0.0
    summary = {"pq": pq, "tp": tp, "fp": fp, "fn": fn,
               "mean_iou_pos": float(np.mean(iou_scores)) if iou_scores else None,
               "mean_dice_pos": float(np.mean(dice_scores)) if dice_scores else None}
    (args.outdir / "summary.json").write_text(json.dumps(summary, indent=1))

    for name, scores in (("iou", iou_scores), ("dice", dice_scores)):
        fig, ax = plt.subplots(figsize=(6, 4))
        s = np.asarray(scores)
        if s.size:
            ax.hist(s, bins=50)
            ax.axvline(s.mean(), color="red", linestyle="--", label=f"mean={s.mean():.3f}")
            ax.legend()
        ax.set_xlabel(name)
        ax.set_title(f"{name} distribution (positive overlaps)")
        fig.tight_layout()
        fig.savefig(args.outdir / f"{name}_dist.png", dpi=110)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    for counter, lab, col in ((m2n_gt, "GT->pred partners", "steelblue"), (m2n_pred, "pred->GT partners", "indianred")):
        ks = sorted(counter)
        ax.bar([str(k) for k in ks], [counter[k] for k in ks], label=lab, color=col)
    ax.set_xlabel("number of overlapping partners")
    ax.set_title("Many-to-one / one-to-many relations")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.outdir / "m2n.png", dpi=110)
    plt.close(fig)
    print(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
