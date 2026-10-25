"""Build labeled contact sheets of candidate boxes from an audit JSON, for
visual review. Cells show brightened crops around each candidate box with
the cell index, class, and confidence. Emits a CSV mapping cells to candidates.

Usage:
  make_sheets.py --audit audit.json --images-dir data/starter_kit/images/train \
      --mode missing_high --out-dir outputs/review/sheetN --per-sheet 64
"""
import argparse
import csv
import json
from pathlib import Path

import cv2
import numpy as np

NAMES = {0: "Bicycle", 1: "Boat", 2: "Bottle", 3: "Bus", 4: "Car", 5: "Cat",
         6: "Chair", 7: "Cup", 8: "Dog", 9: "Motorbike", 10: "People", 11: "Table"}


def brighten(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    out = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
    out = np.clip(out * 1.25, 0, 255).astype(np.uint8)
    return out


def crop_cell(img, box, cell_size=300, margin=0.5):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = [float(v) for v in box]
    bw, bh = x2 - x1, y2 - y1
    mx, my = bw * margin, bh * margin
    cx1, cy1 = x1 - mx, y1 - my
    cx2, cy2 = x2 + mx, y2 + my
    # scale so the box fills most of the cell
    scale = min(cell_size / max(cx2 - cx1, 1), cell_size / max(cy2 - cy1, 1))
    scale = min(scale, cell_size / max(x2 - x1, 1) * 2.0, 4.0)
    nw, nh = (cx2 - cx1) * scale, (cy2 - cy1) * scale
    ox, oy = (cell_size - nw) / 2, (cell_size - nh) / 2
    rx1 = int(np.clip(cx1 * scale + ox, 0, w))
    ry1 = int(np.clip(cy1 * scale + oy, 0, h))
    rx2 = int(np.clip(cx2 * scale + ox, 0, w))
    ry2 = int(np.clip(cy2 * scale + oy, 0, h))
    if rx2 - rx1 < 4 or ry2 - ry1 < 4:
        # fallback: center crop
        cxp, cyp = int((x1 + x2) / 2), int((y1 + y2) / 2)
        half = cell_size // 2
        rx1, ry1 = max(0, cxp - half), max(0, cyp - half)
        rx2 = min(w, rx1 + cell_size)
        ry2 = min(h, ry1 + cell_size)
    crop = img[ry1:ry2, rx1:rx2]
    crop = cv2.resize(crop, (cell_size, cell_size))
    # draw the actual box in cell coords
    bx1 = int((x1 - cx1) * scale + ox)
    by1 = int((y1 - cy1) * scale + oy)
    bx2 = int((x2 - cx1) * scale + ox)
    by2 = int((y2 - cy1) * scale + oy)
    cv2.rectangle(crop, (bx1, by1), (bx2, by2), (0, 255, 0), 2)
    return crop


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", required=False)
    ap.add_argument("--cand", required=False, help="flat candidate JSON: [{stem,cls,conf,xyxy}]")
    ap.add_argument("--images-dir", required=True)
    ap.add_argument("--mode", choices=["missing_high", "missing_mid", "unmatched_gt"], required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--per-sheet", type=int, default=64)
    ap.add_argument("--cols", type=int, default=8)
    ap.add_argument("--max-sheets", type=int, default=0)
    args = ap.parse_args()

    images_dir = Path(args.images_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    candidates = []
    if args.cand:
        for item in json.loads(Path(args.cand).read_text()):
            candidates.append((item["stem"], item))
    else:
        audit = json.loads(Path(args.audit).read_text())
        for rec in audit["records"]:
            for p in rec["preds"]:
                if args.mode == "missing_high" and p["conf"] >= audit["args"]["c_high"] and p["iou_same"] < audit["args"]["iou_thr"]:
                    candidates.append((rec["stem"], p))
                elif args.mode == "missing_mid" and audit["args"]["c_mid"] <= p["conf"] < audit["args"]["c_high"] and p["iou_same"] < audit["args"]["iou_thr"]:
                    candidates.append((rec["stem"], p))
    # sort by class then conf desc for review ergonomics
    candidates.sort(key=lambda t: (t[1]["cls"], -t[1]["conf"]))

    csv_path = out_dir / "candidates.csv"
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cell", "sheet", "stem", "cls", "cls_name", "conf", "x1", "y1", "x2", "y2"])
    sheet_paths = []
    for s_i in range(0, len(candidates), args.per_sheet):
        chunk = candidates[s_i:s_i + args.per_sheet]
        rows = (len(chunk) + args.cols - 1) // args.cols
        sheet = np.full((rows * 300, args.cols * 300, 3), 40, np.uint8)
        with open(csv_path, "a", newline="") as f:
            w = csv.writer(f)
            for i, (stem, p) in enumerate(chunk):
                img = cv2.imread(str(images_dir / f"{stem}.jpg"))
                if img is None:
                    continue
                cell = crop_cell(brighten(img), p["xyxy"])
                label = f"{i} {NAMES.get(p['cls'], p['cls'])} {p['conf']:.2f}"
                cv2.putText(cell, label, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
                stem_tag = stem[-8:]
                cv2.putText(cell, stem_tag, (5, 295), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
                r, c = divmod(i, args.cols)
                sheet[r * 300:(r + 1) * 300, c * 300:(c + 1) * 300] = cell
                x1, y1, x2, y2 = p["xyxy"]
                w.writerow([i, s_i // args.per_sheet, stem, p["cls"], NAMES.get(p["cls"]), f"{p['conf']:.4f}",
                            f"{x1:.1f}", f"{y1:.1f}", f"{x2:.1f}", f"{y2:.1f}"])
        sheet_path = out_dir / f"sheet_{s_i // args.per_sheet:02d}.jpg"
        cv2.imwrite(str(sheet_path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 85])
        sheet_paths.append(str(sheet_path))
        if args.max_sheets and len(sheet_paths) >= args.max_sheets:
            break
    print(json.dumps({"mode": args.mode, "n_candidates": len(candidates),
                      "sheets": len(sheet_paths), "csv": str(csv_path),
                      "first": str(sheet_paths[0]) if sheet_paths else None}), flush=True)


if __name__ == "__main__":
    main()
