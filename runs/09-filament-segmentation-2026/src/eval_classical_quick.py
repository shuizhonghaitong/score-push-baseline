"""Quick classical-baseline PQ estimate on a 40-observation val subsample."""
from __future__ import annotations
import sys, time
from pathlib import Path
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_classical_local import build_records, grouped_split, masks_of, IMGDIR, IMG
from otsu_util import otsu_threshold

def main():
    t0 = time.time()
    recs = build_records()
    train, val = grouped_split(recs)
    stems = sorted({r["stem"] for r in val})
    stems = stems[:: max(1, len(stems) // 40)][:40]
    val_s = [r for r in val if r["stem"] in set(stems)]
    print(f"subsample stems={len(stems)} recs={len(val_s)}", flush=True)
    by_stem = {}
    for r in val_s:
        by_stem.setdefault(r["stem"], []).append(r)

    def pq_for(base):
        tp_iou = fp = fn = tp = 0.0
        for s, rs in by_stem.items():
            m = base[s].astype(np.uint8)
            k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, k)
            n, lab = cv2.connectedComponents(m, connectivity=8)
            pm = [lab == i for i in range(1, n) if (lab == i).sum() >= 128]
            npred = len(pm)
            if npred:
                plab = np.zeros((IMG, IMG), np.int32)
                for j, mm in enumerate(pm, 1):
                    plab[mm] = j
                p_flat = plab.ravel()
            else:
                p_flat = None
            for r in rs:
                ms = masks_of(r); ng = len(ms)
                if ng == 0: fp += npred; continue
                if npred == 0: fn += ng; continue
                glab = np.zeros((IMG, IMG), np.int32)
                for i, mm in enumerate(ms, 1):
                    glab[mm] = i
                g_flat = glab.ravel()
                nz = (g_flat > 0) & (p_flat > 0)  # intersection pixels only (no -1 wraparound)
                inter = np.zeros((ng, npred), np.int64)
                np.add.at(inter, (g_flat[nz] - 1, p_flat[nz] - 1), 1)
                g_area = np.bincount(g_flat[g_flat > 0] - 1, minlength=ng)
                p_area = np.bincount(p_flat[p_flat > 0] - 1, minlength=npred)
                union = g_area[:, None] + p_area[None, :] - inter
                ious = np.where(union > 0, inter / np.maximum(union, 1), 0.0)
                hits = ious > 0.5
                tp_iou += ious[hits].sum(); tp += int(hits.sum())
                fn += int((~hits.any(axis=1)).sum()); fp += int((~hits.any(axis=0)).sum())
        d = tp + 0.5 * fp + 0.5 * fn
        return tp_iou / d if d else 0.0, tp, fp, fn

    base = {}
    for s in stems:
        img = cv2.imread(str(IMGDIR / f"{s}.jpeg"), cv2.IMREAD_GRAYSCALE)
        resid = cv2.medianBlur(img, 151).astype(np.int16) - img.astype(np.int16)
        floor = np.quantile(resid, 0.90)
        u8 = np.clip(np.clip(resid, floor, resid.max()), 0, 255).astype(np.uint8)
        base[s] = resid >= max(otsu_threshold(u8), 8)
        print(f"  {s} t={time.time()-t0:.0f}s", flush=True)
    pq, tp, fp, fn = pq_for(base)
    print(f"QUICK CLASSICAL PQ={pq:.4f} TP={tp} FP={fp} FN={fn} t={time.time()-t0:.0f}s")

if __name__ == "__main__":
    main()
