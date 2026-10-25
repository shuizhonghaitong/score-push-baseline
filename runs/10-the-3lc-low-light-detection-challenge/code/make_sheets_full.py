"""Full-image contact sheets: each cell shows the brightened full image with
ALL existing GT boxes (thin red) and the candidate box (thick green).
Cell label: cell index, stem, pred class, conf, n_GT."""
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
    l = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(l)
    out = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
    return np.clip(out * 1.2, 0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cand", required=True)
    ap.add_argument("--images-dir", required=True)
    ap.add_argument("--labels-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--per-sheet", type=int, default=48)
    ap.add_argument("--cols", type=int, default=6)
    ap.add_argument("--h", type=int, default=320)
    args = ap.parse_args()

    cands = json.loads(Path(args.cand).read_text())
    cands.sort(key=lambda t: (t["cls"], -t["conf"]))
    images_dir, labels_dir = Path(args.images_dir), Path(args.labels_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "candidates.csv"

    rows = []
    for s_i in range(0, len(cands), args.per_sheet):
        chunk = cands[s_i:s_i + args.per_sheet]
        n_rows = (len(chunk) + args.cols - 1) // args.cols
        cw = int(args.h * 4 / 3)
        sheet = np.full((n_rows * args.h, args.cols * cw, 3), 30, np.uint8)
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["cell", "sheet", "stem", "cls", "cls_name", "conf", "x1", "y1", "x2", "y2", "n_gt"])
            for i, c in enumerate(chunk):
                img = cv2.imread(str(images_dir / f"{c['stem']}.jpg"))
                if img is None:
                    continue
                H, W = img.shape[:2]
                gts = []
                lf = labels_dir / f"{c['stem']}.txt"
                if lf.exists():
                    for line in lf.read_text().splitlines():
                        p = line.split()
                        if len(p) >= 5:
                            gts.append((int(float(p[0])), (float(p[1]) - float(p[3]) / 2) * W,
                                        (float(p[2]) - float(p[4]) / 2) * H,
                                        (float(p[1]) + float(p[3]) / 2) * W,
                                        (float(p[2]) + float(p[4]) / 2) * H))
                cell = brighten(img)
                ch, cw2 = cell.shape[:2]
                scale = min(args.h / ch, cw / cw2)
                nw, nh = int(cw2 * scale), int(ch * scale)
                cell = cv2.resize(cell, (nw, nh))
                canvas = np.full((args.h, cw, 3), 30, np.uint8)
                ox, oy = (cw - nw) // 2, (args.h - nh) // 2
                canvas[oy:oy + nh, ox:ox + nw] = cell
                cell = canvas
                for gc, gx1, gy1, gx2, gy2 in gts:
                    x1, y1 = int(gx1 * scale), int(gy1 * scale)
                    x2, y2 = int(gx2 * scale), int(gy2 * scale)
                    cv2.rectangle(cell, (x1, y1), (x2, y2), (0, 0, 200), 1)
                x1, y1, x2, y2 = c["xyxy"]
                cv2.rectangle(cell, (int(x1 * scale), int(y1 * scale)), (int(x2 * scale), int(y2 * scale)), (0, 255, 0), 2)
                label = f"{i} {NAMES.get(c['cls'])} {c['conf']:.2f} gt={len(gts)}"
                cv2.putText(cell, label, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
                cv2.putText(cell, c["stem"][-8:], (4, cell.shape[0] - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
                r, col = divmod(i, args.cols)
                sheet[r * args.h:(r + 1) * args.h, col * cw:(col + 1) * cw] = cell
                w.writerow([i, s_i // args.per_sheet, c["stem"], c["cls"], NAMES.get(c["cls"]),
                            f"{c['conf']:.4f}", f"{x1:.1f}", f"{y1:.1f}", f"{x2:.1f}", f"{y2:.1f}", len(gts)])
        sp = out_dir / f"full_{s_i // args.per_sheet:02d}.jpg"
        cv2.imwrite(str(sp), sheet, [cv2.IMWRITE_JPEG_QUALITY, 82])
        rows.append(str(sp))
    print(json.dumps({"n": len(cands), "sheets": len(rows), "first": rows[0] if rows else None}), flush=True)


if __name__ == "__main__":
    main()
