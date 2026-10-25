"""COCO mAP@0.5 evaluation mirroring the competition metric.

Takes a YOLO model + labeled split (images + YOLO labels), runs inference
at the given conf/max_det (scoring caps boxes at 100/image), converts both
to COCO absolute-pixel boxes, and runs pycocotools bbox eval at IoU=0.5.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("YOLO_CONFIG_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "state", "ultralytics"))


def load_yolo_labels(labels_dir: Path, image_sizes: dict):
    """Return COCO-style (images, anns) for a labeled split."""
    images, anns = [], []
    ann_id = 1
    for img_id, (w, h) in image_sizes.items():
        images.append({"id": img_id, "width": w, "height": h, "file_name": f"{img_id}.jpg"})
    label_files = sorted(Path(labels_dir).glob("*.txt"))
    for lf in label_files:
        stem = lf.stem
        if stem not in image_sizes:
            continue
        w, h = image_sizes[stem]
        img_id = stem
        with open(lf) as f:
            for line in f:
                parts = line.split()
                if len(parts) < 5:
                    continue
                cls = int(float(parts[0]))
                xc, yc, bw, bh = (float(x) for x in parts[1:5])
                x1 = (xc - bw / 2) * w
                y1 = (yc - bh / 2) * h
                ww = bw * w
                hh = bh * h
                if ww <= 0 or hh <= 0:
                    continue
                anns.append({
                    "id": ann_id, "image_id": img_id, "category_id": cls,
                    "bbox": [x1, y1, ww, hh], "area": ww * hh,
                    "iscrowd": 0,
                })
                ann_id += 1
    return images, anns


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", required=True)
    ap.add_argument("--images-dir", required=True)
    ap.add_argument("--labels-dir", required=True)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--max-det", type=int, default=100, help="score-side cap")
    ap.add_argument("--pred-max-det", type=int, default=300)
    ap.add_argument("--device", default="0")
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    import torch
    if not torch.cuda.is_available():
        print("FATAL: CUDA unavailable", flush=True)
        sys.exit(3)
    import numpy as np
    from PIL import Image
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    from ultralytics import YOLO

    images_dir = Path(args.images_dir)
    image_sizes = {}
    for p in sorted(images_dir.glob("*.jpg")):
        with Image.open(p) as im:
            image_sizes[p.stem] = (im.width, im.height)

    model = YOLO(args.weights)
    t0 = time.time()
    preds = model.predict(source=str(images_dir), imgsz=args.imgsz, conf=args.conf,
                          max_det=args.pred_max_det, device=args.device, verbose=False)

    images, anns = load_yolo_labels(Path(args.labels_dir), image_sizes)
    cats = [{"id": i, "name": n} for i, n in enumerate(
        ["Bicycle", "Boat", "Bottle", "Bus", "Car", "Cat", "Chair", "Cup", "Dog", "Motorbike", "People", "Table"])]

    coco = COCO()
    coco.dataset = {"images": images, "annotations": anns, "categories": cats}
    coco.imgs = {im["id"]: im for im in images}
    coco.catToImg = {}
    coco.cat2imgs = {}
    for im in images:
        coco.catToImg.setdefault(im["id"], [])
    for an in anns:
        coco.cat2imgs.setdefault(an["category_id"], set()).add(an["image_id"])
    coco.anns = {a["id"]: a for a in anns}
    coco.imgToAnns = {}
    for a in anns:
        coco.imgToAnns.setdefault(a["image_id"], []).append(a["id"])

    det_anns = []
    det_id = 1
    for p in preds:
        stem = Path(p.path).stem
        if stem not in image_sizes:
            continue
        boxes = p.boxes
        if boxes is None or len(boxes) == 0:
            continue
        cls = boxes.cls.tolist()
        confs = boxes.conf.tolist()
        xyxy = boxes.xyxy.tolist()
        order = np.argsort([-c for c in confs])
        for i in list(order)[:args.max_det]:
            x1, y1, x2, y2 = xyxy[i]
            det_anns.append({
                "id": det_id, "image_id": stem, "category_id": int(cls[i]),
                "bbox": [x1, y1, x2 - x1, y2 - y1], "score": float(confs[i]),
            })
            det_id += 1

    if not det_anns:
        print(json.dumps({"mAP50": 0.0, "pred_boxes": 0, "elapsed_s": round(time.time()-t0, 1)}), flush=True)
        return

    gt = COCO()
    gt.dataset = {"images": images, "annotations": anns, "categories": cats}
    gt.createIndex()
    coco_det = gt.loadRes(det_anns)
    ev = COCOeval(gt, coco_det, "bbox")
    ev.iouThrs = [0.5]
    ev.params.maxDets = [1, 10, 100]
    ev.params.useCats = 1
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    map50 = float(ev.stats[1])

    per_class = {}
    for cat in cats:
        gt_anns = [a for a in anns if a["category_id"] == cat["id"]]
        if not gt_anns:
            continue
        gt_c = COCO()
        gt_c.dataset = {"images": images, "annotations": gt_anns, "categories": [cat]}
        gt_c.createIndex()
        cls_dets = [d for d in det_anns if d["category_id"] == cat["id"]]
        if not cls_dets:
            per_class[cat["name"]] = 0.0
            continue
        dt_c = gt_c.loadRes(cls_dets)
        evc = COCOeval(gt_c, dt_c, "bbox")
        evc.iouThrs = [0.5]
        evc.params.maxDets = [1, 10, 100]
        evc.params.useCats = 1
        evc.evaluate()
        evc.accumulate()
        evc.summarize()
        per_class[cat["name"]] = round(float(evc.stats[1]), 4)

    out = {
        "mAP50": round(map50, 4),
        "AP50_per_class": per_class,
        "pred_boxes": len(det_anns),
        "gt_boxes": len(anns),
        "n_images": len(images),
        "conf": args.conf,
        "elapsed_s": round(time.time() - t0, 1),
    }
    print("EVAL_JSON " + json.dumps(out), flush=True)
    if args.out_json:
        Path(args.out_json).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
