"""Train YOLOv11n for the 3LC low-light challenge (locked contract:
yolo11n.pt official COCO-pretrained init, imgsz=640, single model).
Hyperparameters are free. Runs offline in the local GPU pool job sandbox.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("YOLO_CONFIG_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "state", "ultralytics"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--weights", required=True)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--name", required=True)
    ap.add_argument("--project", default=None)
    ap.add_argument("--device", default="0")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--lr0", type=float, default=None)
    ap.add_argument("--lrf", type=float, default=None)
    ap.add_argument("--patience", type=int, default=None)
    ap.add_argument("--warmup-epochs", type=float, default=None)
    ap.add_argument("--close-mosaic", type=int, default=None)
    ap.add_argument("--augment", type=int, default=0)
    ap.add_argument("--cos-lr", type=int, default=None)
    ap.add_argument("--optimizer", default=None)
    ap.add_argument("--momentum", type=float, default=None)
    ap.add_argument("--weight-decay", type=float, default=None)
    ap.add_argument("--hsv-v", type=float, default=None)
    args = ap.parse_args()

    import torch
    if not torch.cuda.is_available():
        print("FATAL: CUDA unavailable in training job", flush=True)
        sys.exit(3)
    print("GPU:", torch.cuda.get_device_name(0), flush=True)

    from ultralytics import YOLO

    extra = {}
    for key in ("lr0", "lrf", "patience", "warmup_epochs", "close_mosaic", "cos_lr",
                "optimizer", "momentum", "weight_decay", "hsv_v"):
        val = getattr(args, key.replace("-", "_"))
        if val is not None:
            if key in ("cos_lr",):
                val = bool(val)
            extra[key] = val

    # Resolve to absolute paths: the host's global Ultralytics runs_dir setting
    # would otherwise hijack relative project paths.
    args.project = str(Path(args.project).resolve()) if args.project else str(Path("models/runs").resolve())
    args.data = str(Path(args.data).resolve())
    args.weights = str(Path(args.weights).resolve())

    model = YOLO(args.weights)
    t0 = time.time()
    results = model.train(
        data=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        name=args.name,
        project=args.project,
        device=args.device,
        workers=args.workers,
        plots=True,
        exist_ok=True,
        **extra,
    )
    elapsed = time.time() - t0
    metrics = results.results_dict
    summary = {
        "name": args.name,
        "epochs": args.epochs,
        "elapsed_s": round(elapsed, 1),
        "box_map": float(metrics.get("metrics/mAP50(B)", float("nan"))),
        "box_map50_95": float(metrics.get("metrics/mAP50-95(B)", float("nan"))),
        "per_class_ap50": [round(float(x), 4) for x in metrics.get("metrics/mAP50(B) per class", [])],
        "classes": ["Bicycle", "Boat", "Bottle", "Bus", "Car", "Cat", "Chair", "Cup", "Dog", "Motorbike", "People", "Table"],
    }
    print("SUMMARY_JSON " + json.dumps(summary), flush=True)
    sys.stdout.flush()


if __name__ == "__main__":
    main()
