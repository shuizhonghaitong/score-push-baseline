"""Classical (non-ML) solar filament segmentation baseline.

Pipeline per image:
1. Background estimation with a large median blur; foreground = dark residuals.
2. Robust threshold: Otsu on the residual histogram restricted to the dark side.
3. Morphological closing/opening to clean noise and connect thin structures.
4. 8-connected components with a minimum-area filter.

Runs locally (CPU). Produces a submission-style per-image instance list and
can score itself against COCO annotations with the official PQ formula.
"""
from __future__ import annotations

import argparse
import base64
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from otsu_util import otsu_threshold  # noqa: E402



def segment(image: np.ndarray, dark_quantile: float = 0.10, min_px: int = 128,
            k_close: int = 3, k_open: int = 1) -> list[np.ndarray]:
    blurred = cv2.medianBlur(image, 151)
    resid = blurred.astype(np.int16) - image.astype(np.int16)  # positive = darker than background
    # keep only plausibly-dark pixels for Otsu stability
    floor = np.quantile(resid, 1 - dark_quantile)
    hist_src = np.clip(resid, floor, resid.max() if resid.size else 0)
    # Otsu on uint8 version of the dark-side residual
    src_u8 = np.clip(hist_src, 0, 255).astype(np.uint8)
    thr = otsu_threshold(src_u8)
    mask = (resid >= max(thr, 8)).astype(np.uint8)
    if k_close > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_close, k_close))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k)
    if k_open > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_open, k_open))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    n, lab = cv2.connectedComponents(mask, connectivity=8)
    out = []
    for i in range(1, n):
        part = lab == i
        if part.sum() >= min_px:
            out.append(part.astype(bool))
    return out


def run(image_dir: Path, out_csv: Path, **kw) -> pd.DataFrame:
    rows = []
    for p in sorted(image_dir.glob("*.jpeg")):
        img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
        for k, m in enumerate(segment(img, **kw), 1):
            from pycocotools import mask as mu
            rle = mu.encode(np.asfortranarray(m.astype(np.uint8)))
            c = rle["counts"]
            if isinstance(c, bytes):
                c = base64.b64encode(c).decode("ascii")
            rows.append({"filament_id": f"{p.stem}_{k}", "segmentation_rle": c})
    df = pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])
    df.to_csv(out_csv, index=False)
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    df = run(args.images, args.out)
    print(f"rows={len(df)} -> {args.out}")
