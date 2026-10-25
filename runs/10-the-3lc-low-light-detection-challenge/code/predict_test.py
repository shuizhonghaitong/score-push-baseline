"""Run a trained YOLO model on the 715 test images and write a validated
submission.csv (competition format: id, image_id, prediction_string).
Matches the plugin's locked inference: imgsz=640, max_det=300, conf from arg.
"""
import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("YOLO_CONFIG_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "state", "ultralytics"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--test-dir", required=True)
    ap.add_argument("--submission-template", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--max-det", type=int, default=300)
    ap.add_argument("--device", default="0")
    ap.add_argument("--include-classes", default=None, help="comma list of class ids to keep (default all 0-11)")
    args = ap.parse_args()

    import torch
    if not torch.cuda.is_available():
        print("FATAL: CUDA unavailable", flush=True)
        sys.exit(3)

    from ultralytics import YOLO

    keep = None
    if args.include_classes:
        keep = set(int(x) for x in args.include_classes.split(","))

    model = YOLO(args.weights)
    test_dir = Path(args.test_dir)
    rows = []
    with open(args.submission_template) as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append((int(row["id"]), row["image_id"]))
    rows.sort(key=lambda r: r[0])

    t0 = time.time()
    preds = model.predict(
        source=str(test_dir),
        imgsz=args.imgsz,
        conf=args.conf,
        max_det=args.max_det,
        device=args.device,
        verbose=False,
    )
    by_stem = {}
    for p in preds:
        stem = Path(p.path).stem
        boxes = p.boxes
        parts = []
        if boxes is not None and len(boxes):
            cls = boxes.cls.tolist()
            confs = boxes.conf.tolist()
            xywhn = boxes.xywhn.tolist()
            for c, cf, (xc, yc, w, h) in zip(cls, confs, xywhn):
                c = int(c)
                if keep is not None and c not in keep:
                    continue
                parts.append(f"{c} {cf:.6f} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
        by_stem[stem] = " ".join(parts) if parts else "no box"

    missing = [stem for _, stem in rows if stem not in by_stem]
    if missing:
        print(f"FATAL: no predictions for {len(missing)} test images, e.g. {missing[:5]}", flush=True)
        sys.exit(4)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    total_boxes = 0
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "image_id", "prediction_string"])
        for rid, stem in rows:
            s = by_stem[stem]
            if s != "no box":
                total_boxes += len(s.split()) // 6
            w.writerow([rid, stem, s])
    elapsed = time.time() - t0
    print(json.dumps({"out": str(out), "rows": len(rows), "total_boxes": total_boxes,
                      "empty_images": sum(1 for _, s in by_stem.items() if s == "no box"),
                      "elapsed_s": round(elapsed, 1)}), flush=True)


if __name__ == "__main__":
    main()
