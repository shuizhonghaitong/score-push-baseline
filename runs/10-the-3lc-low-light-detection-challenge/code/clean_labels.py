"""Deterministic, policy-based label cleanup (no model involvement).

- clips normalized coordinates to [0, 1]
- drops degenerate boxes (w or h < min_side)
- dedups same-class boxes with IoU > iou_dup (keeps the larger area one)
- validates class ids are 0..11

Writes to a NEW labels dir; originals are preserved.
"""
import argparse
import json
from pathlib import Path


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
    ap.add_argument("--src", required=True)
    ap.add_argument("--dst", required=True)
    ap.add_argument("--min-side", type=float, default=0.002)
    ap.add_argument("--iou-dup", type=float, default=0.95)
    args = ap.parse_args()

    src, dst = Path(args.src), Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)
    stats = {"images": 0, "boxes_in": 0, "boxes_out": 0, "clipped": 0,
             "degenerate": 0, "dups": 0, "badclass": 0, "changed_images": 0}
    for lf in sorted(src.glob("*.txt")):
        boxes = []
        for line in lf.read_text().splitlines():
            p = line.split()
            if len(p) < 5:
                continue
            stats["boxes_in"] += 1
            c = int(float(p[0]))
            if not 0 <= c <= 11:
                stats["badclass"] += 1
                continue
            xc, yc, w, h = (float(x) for x in p[1:5])
            changed = False
            vals = [xc, yc, w, h]
            newvals = [min(max(v, 0.0), 1.0) for v in vals]
            if newvals != vals:
                changed = True
                stats["clipped"] += 1
            xc, yc, w, h = newvals
            if w < args.min_side or h < args.min_side:
                stats["degenerate"] += 1
                continue
            x1, y1, x2, y2 = xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2
            boxes.append((c, x1, y1, x2, y2, w * h, [c, xc, yc, w, h]))
        # dedup
        kept = []
        for box in sorted(boxes, key=lambda b: -b[5]):
            c, x1, y1, x2, y2, area, _ = box
            dup = False
            for k in kept:
                if k[0] == c and iou((x1, y1, x2, y2), (k[1], k[2], k[3], k[4])) > args.iou_dup:
                    stats["dups"] += 1
                    dup = True
                    break
            if not dup:
                kept.append(box)
        out = [k[6] for k in kept]
        stats["boxes_out"] += len(out)
        if len(out) != len([1 for _ in lf.read_text().splitlines() if _.split()]):
            stats["changed_images"] += 1
        (dst / lf.name).write_text("".join(
            f"{c} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n" for c, xc, yc, w, h in out))
        stats["images"] += 1
    print(json.dumps(stats), flush=True)


if __name__ == "__main__":
    main()
