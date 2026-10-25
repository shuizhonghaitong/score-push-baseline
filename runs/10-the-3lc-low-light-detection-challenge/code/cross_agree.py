"""Cross-model agreement: accept missing-label candidates where two models agree.

Candidate = baseline high-conf box unmatched to GT (iou_any < iou_thr).
Accepted when the second model also predicts the same class (conf >= min_conf),
unmatched to GT (iou_any < iou_thr), with IoU >= min_agree_iou between models.
Consensus box = mean of the two boxes.
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
    ap.add_argument("--missing", required=True, help="baseline candidate list json")
    ap.add_argument("--tuned-audit", required=True, help="second model full audit json")
    ap.add_argument("--min-conf", type=float, default=0.5)
    ap.add_argument("--min-agree-iou", type=float, default=0.3)
    ap.add_argument("--iou-thr", type=float, default=0.1)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    missing = json.loads(Path(args.missing).read_text())
    tuned = json.loads(Path(args.tuned_audit).read_text())
    tuned_by_stem = {r["stem"]: r for r in tuned["records"]}

    accepted, rejected_counts = [], {"no_tuned_record": 0, "no_agreeing_pred": 0, "low_iou": 0}
    for cand in missing:
        stem, cls, cbox = cand["stem"], cand["cls"], cand["xyxy"]
        rec = tuned_by_stem.get(stem)
        if rec is None:
            rejected_counts["no_tuned_record"] += 1
            continue
        best_iou, best_pred = -1.0, None
        for p in rec["preds"]:
            if p["cls"] != cls or p["conf"] < args.min_conf or p["iou_any"] >= args.iou_thr:
                continue
            v = iou(cbox, p["xyxy"])
            if v > best_iou:
                best_iou, best_pred = v, p
        if best_pred is None:
            rejected_counts["no_agreeing_pred"] += 1
            continue
        if best_iou < args.min_agree_iou:
            rejected_counts["low_iou"] += 1
            continue
        bx = best_pred["xyxy"]
        box = [(a + b) / 2 for a, b in zip(cbox, bx)]
        accepted.append({
            "stem": stem, "cls": cls, "box": box,
            "base_conf": cand["conf"], "tuned_conf": best_pred["conf"],
            "iou_agree": round(best_iou, 4),
        })

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(accepted, indent=1))
    print(json.dumps({"accepted": len(accepted), "rejected": rejected_counts}), flush=True)


if __name__ == "__main__":
    main()
