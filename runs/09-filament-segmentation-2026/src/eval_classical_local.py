"""Local evaluation of the classical baseline on the esairlab MAGFiLO data.

Uses the same grouped observation split (seed 42) and official PQ as the
Kaggle kernels, so the val PQ here approximates the kernel's val PQ.
"""
from __future__ import annotations
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from baseline_classical import segment  # noqa: E402
from evaluate import pq_score  # noqa: E402
from otsu_util import otsu_threshold  # noqa: E402

RUN = Path(__file__).resolve().parents[1]
ROOT = RUN / "data/extracted"
ANN = ROOT / "MAGFiLO_1.0_Annotations.json"
IMGDIR = ROOT / "MAGFiLO_1.0_Images"
IMG = 2048


def build_records():
    import pycocotools.coco as coco
    c = coco.COCO(str(ANN))
    recs = []
    for rec_id, info in c.imgs.items():
        stem = Path(info["file_name"]).stem
        recs.append({"record_id": str(rec_id), "stem": stem,
                     "img_path": IMGDIR / f"{stem}.jpeg",
                     "ann_items": list(c.imgToAnns[rec_id]), "coco": c})
    return recs


def grouped_split(records, val_frac=0.2, seed=42):
    stems = sorted({r["stem"] for r in records})
    rng = random.Random(seed)
    rng.shuffle(stems)
    n_val = max(1, int(round(len(stems) * val_frac)))
    val_stems = set(stems[:n_val])
    train = [r for r in records if r["stem"] not in val_stems]
    val = [r for r in records if r["stem"] in val_stems]
    return train, val


def masks_of(rec):
    return [rec["coco"].annToMask(a) > 0 for a in rec["ann_items"]]


def main():
    t0 = time.time()
    recs = build_records()
    train, val = grouped_split(recs)
    val_stems = sorted({r["stem"] for r in val})
    print(f"recs={len(recs)} train={len(train)} val={len(val)} val_obs={len(val_stems)}", flush=True)

    def build_base(q):
        base = {}
        for s in val_stems:
            img = cv2.imread(str(IMGDIR / f"{s}.jpeg"), cv2.IMREAD_GRAYSCALE)
            blurred = cv2.medianBlur(img, 151).astype(np.int16)
            resid = blurred - img.astype(np.int16)
            floor = np.quantile(resid, 1 - q)
            src_u8 = np.clip(np.clip(resid, floor, resid.max()), 0, 255).astype(np.uint8)
            thr = otsu_threshold(src_u8)
            base[s] = (resid >= max(thr, 8))
        return base

    def preds_for(q, k_close, min_px, base):
        out = {}
        for s in val_stems:
            m = base[s].astype(np.uint8)
            if k_close > 0:
                k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_close, k_close))
                m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
            parts = []
            n, lab = cv2.connectedComponents(m, connectivity=8)
            for i in range(1, n):
                part = lab == i
                if part.sum() >= min_px:
                    parts.append(part)
            out[s] = parts
        return out

    val_by_stem = {}
    for r in val:
        val_by_stem.setdefault(r["stem"], []).append(r)

    def pq_of(preds):
        tp_iou, fp, fn, tp = 0.0, 0, 0, 0
        for s, rs in val_by_stem.items():
            pm = preds[s]
            npred = len(pm)
            if npred:
                plab = np.zeros((IMG, IMG), np.int32)
                for j, m in enumerate(pm, 1):
                    plab[m] = j
                p_flat = plab.ravel()
            else:
                p_flat = None
            for r in rs:
                ms = masks_of(r)
                ng = len(ms)
                if ng == 0:
                    fp += npred; continue
                if npred == 0:
                    fn += ng; continue
                glab = np.zeros((IMG, IMG), np.int32)
                for i, m in enumerate(ms, 1):
                    glab[m] = i
                g_flat = glab.ravel()
                nz = (g_flat > 0) & (p_flat > 0)  # intersection pixels only (no -1 wraparound)
                gi = g_flat[nz] - 1
                pi = p_flat[nz] - 1
                inter = np.zeros((ng, npred), np.int64)
                np.add.at(inter, (gi, pi), 1)
                g_area = np.bincount(g_flat[g_flat > 0] - 1, minlength=ng)
                p_area = np.bincount(p_flat[p_flat > 0] - 1, minlength=npred)
                union = g_area[:, None] + p_area[None, :] - inter
                ious = np.where(union > 0, inter / np.maximum(union, 1), 0.0)
                hits = ious > 0.5
                tp_iou += ious[hits].sum()
                tp += int(hits.sum())
                fn += int((~hits.any(axis=1)).sum())
                fp += int((~hits.any(axis=0)).sum())
        denom = tp + 0.5 * fp + 0.5 * fn
        return tp_iou / denom if denom else 0.0, tp, fp, fn

    results = []
    for q in (0.08, 0.10, 0.15):
        print(f"building base masks q={q}...", flush=True)
        base = build_base(q)
        for kc in (3, 5):
            for mp in (64, 256, 512):
                preds = preds_for(q, kc, mp, base)
                pq, tp, fp, fn = pq_of(preds)
                results.append((pq, q, kc, mp, tp, fp, fn))
                print(f"q={q} k_close={kc} min_px={mp}: PQ={pq:.4f} TP={tp} FP={fp} FN={fn} t={time.time()-t0:.0f}s", flush=True)
        del base
    results.sort(reverse=True)
    print("BEST:", results[0])
    json.dump([{"pq": r[0], "q": r[1], "k_close": r[2], "min_px": r[3], "TP": r[4], "FP": r[5], "FN": r[6]}
               for r in results], open(RUN / "report/classical_sweep.json", "w"), indent=1)


if __name__ == "__main__":
    main()
