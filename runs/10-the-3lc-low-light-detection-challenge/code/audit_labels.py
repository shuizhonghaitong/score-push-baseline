"""Audit model predictions against YOLO ground truth on a split.

Outputs:
  - per-image JSON records with predictions, GT boxes, IoU matching
  - candidate lists:
      missing_high: conf >= c_high and max IoU vs same-class GT < iou_thr
      missing_mid:  c_mid <= conf < c_high and max IoU < iou_thr
      unmatched_gt: GT boxes with max IoU vs any pred < iou_thr
  - a CSV summary of candidates (for review)
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("YOLO_CONFIG_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "state", "ultralytics"))


def iou(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def load_gt(labels_dir: Path, stem, W, H):
    boxes = []
    lf = labels_dir / f"{stem}.txt"
    if lf.exists():
        for line in lf.read_text().splitlines():
            p = line.split()
            if len(p) < 5:
                continue
            c = int(float(p[0]))
            xc, yc, bw, bh = (float(x) for x in p[1:5])
            boxes.append([c, (xc - bw / 2) * W, (yc - bh / 2) * H,
                          (xc + bw / 2) * W, (yc + bh / 2) * H])
    return boxes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--images-dir", required=True)
    ap.add_argument("--labels-dir", required=True)
    ap.add_argument("--out", required=True, help="output json path")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--max-det", type=int, default=300)
    ap.add_argument("--c-high", type=float, default=0.5)
    ap.add_argument("--c-mid", type=float, default=0.3)
    ap.add_argument("--iou-thr", type=float, default=0.1)
    ap.add_argument("--device", default="0")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    import torch
    if not torch.cuda.is_available():
        print("FATAL: CUDA unavailable", flush=True)
        sys.exit(3)
    from PIL import Image
    from ultralytics import YOLO

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    images_dir = Path(args.images_dir)
    labels_dir = Path(args.labels_dir)
    files = sorted(images_dir.glob("*.jpg"))
    if args.limit:
        files = files[:args.limit]

    model = YOLO(args.weights)
    t0 = time.time()
    preds = model.predict(source=str(images_dir), imgsz=args.imgsz, conf=args.conf,
                          max_det=args.max_det, device=args.device, verbose=False)
    pred_by_stem = {}
    for p in preds:
        stem = Path(p.path).stem
        boxes = p.boxes
        if boxes is None or len(boxes) == 0:
            pred_by_stem[stem] = []
            continue
        cls = boxes.cls.tolist()
        confs = boxes.conf.tolist()
        xyxy = boxes.xyxy.tolist()
        pred_by_stem[stem] = [[int(c), float(cf), [float(v) for v in bb]]
                              for c, cf, bb in zip(cls, confs, xyxy)]

    records = []
    n_missing_high = n_missing_mid = n_unmatched_gt = 0
    for f in files:
        with Image.open(f) as im:
            w, h = im.size
        gts = load_gt(labels_dir, f.stem, w, h)
        preds = pred_by_stem.get(f.stem, [])
        gt_unmatched = []
        for i, g in enumerate(gts):
            gc, gbox = g[0], g[1:5]
            best = max((iou(gbox, pb[2]) for pb in preds), default=0.0)
            if best < args.iou_thr:
                gt_unmatched.append(i)
        rec_preds = []
        for pi, (pc, pc_f, pbox) in enumerate(preds):
            best_same = max((iou(pbox, g[1:5]) for g in gts if g[0] == pc), default=0.0)
            best_any = max((iou(pbox, g[1:5]) for g in gts), default=0.0)
            if best_same < args.iou_thr:
                if pc_f >= args.c_high:
                    n_missing_high += 1
                elif pc_f >= args.c_mid:
                    n_missing_mid += 1
            rec_preds.append({"pi": pi, "cls": pc, "conf": round(pc_f, 4),
                              "xyxy": pbox, "iou_same": round(best_same, 4),
                              "iou_any": round(best_any, 4)})
        n_unmatched_gt += len(gt_unmatched)
        records.append({"stem": f.stem, "w": w, "h": h, "n_gt": len(gts),
                        "gt_unmatched_idx": gt_unmatched, "preds": rec_preds})
        if len(records) % 500 == 0:
            print(f"  {len(records)}/{len(files)} ({time.time()-t0:.0f}s)", flush=True)

    out.write_text(json.dumps({"args": vars(args), "n_images": len(records),
                               "n_missing_high": n_missing_high,
                               "n_missing_mid": n_missing_mid,
                               "n_unmatched_gt": n_unmatched_gt,
                               "records": records}, separators=(",", ":")))
    print(json.dumps({"out": str(out), "n_images": len(records),
                      "n_missing_high": n_missing_high, "n_missing_mid": n_missing_mid,
                      "n_unmatched_gt": n_unmatched_gt, "elapsed_s": round(time.time()-t0, 1)}), flush=True)


if __name__ == "__main__":
    main()
