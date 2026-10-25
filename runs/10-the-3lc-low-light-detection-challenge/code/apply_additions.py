"""Append reviewed boxes to a labels dir (in place on the given dir).

Input additions JSON: [{"stem": ..., "cls": int, "box": [x1,y1,x2,y2] (pixels)}]
Boxes are converted to normalized YOLO using each image's size. Skips near
duplicates (same class, IoU > 0.5) to stay idempotent.
"""
import argparse
import json
from pathlib import Path

import cv2


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--labels-dir", required=True)
    ap.add_argument("--images-dir", required=True)
    ap.add_argument("--additions", required=True)
    args = ap.parse_args()

    labels_dir = Path(args.labels_dir)
    images_dir = Path(args.images_dir)
    additions = json.loads(Path(args.additions).read_text())
    by_stem = {}
    for a in additions:
        by_stem.setdefault(a["stem"], []).append(a)

    added = skipped = 0
    for stem, items in by_stem.items():
        lf = labels_dir / f"{stem}.txt"
        existing = []
        if lf.exists():
            for line in lf.read_text().splitlines():
                p = line.split()
                if len(p) >= 5:
                    c = int(float(p[0]))
                    xc, yc, w, h = (float(x) for x in p[1:5])
                    existing.append((c, xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2,
                                     [c, xc, yc, w, h]))
        img = cv2.imread(str(images_dir / f"{stem}.jpg"))
        H, W = img.shape[:2]
        lines = [e[5] for e in existing]
        for a in items:
            c = int(a["cls"])
            x1, y1, x2, y2 = [float(v) for v in a["box"]]
            w, h = x2 - x1, y2 - y1
            if w <= 0 or h <= 0:
                skipped += 1
                continue
            for e in existing:
                if e[0] == c and iou((x1, y1, x2, y2), (e[1], e[2], e[3], e[4])) > 0.5:
                    skipped += 1
                    break
            else:
                xc = min(max((x1 + x2) / 2 / W, 0), 1)
                yc = min(max((y1 + y2) / 2 / H, 0), 1)
                nw = min(max(w / W, 0), 1)
                nh = min(max(h / H, 0), 1)
                if nw < 1e-4 or nh < 1e-4:
                    skipped += 1
                    continue
                lines.append([c, xc, yc, nw, nh])
                existing.append((c, x1, y1, x2, y2, None))
                added += 1
        lf.write_text("".join(f"{c} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n" for c, xc, yc, w, h in lines))
    print(json.dumps({"added": added, "skipped": skipped, "images_touched": len(by_stem)}), flush=True)


if __name__ == "__main__":
    main()
