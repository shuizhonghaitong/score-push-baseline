"""Build a submission CSV from per-puzzle packings.

packings: dict n -> list of (x, y, angle) (n items). Missing n => omitted
(charged grid ceiling by the metric).
"""
import csv, math
from score_local import grid_packing

def fmt(v):
    # .17g round-trips float64; must match the strings used for local
    # official-metric verification (scripts/verify_all.py).
    return f"{v:.17g}"

def build(packings, path, max_n=1000):
    import os; os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    rows = 0
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", "n", "square_id", "x", "y", "angle"])
        for n in range(1, max_n + 1):
            if n not in packings:
                continue
            entry = packings[n]
            strs = entry.get("strs") if isinstance(entry, dict) else None
            pts = entry["pts"] if isinstance(entry, dict) else entry
            for sid, (x, y, a) in enumerate(pts):
                if strs is not None:
                    w.writerow([rows, n, sid, strs[sid][0], strs[sid][1], strs[sid][2]])
                else:
                    w.writerow([rows, n, sid, fmt(x), fmt(y), fmt(a)])
                rows += 1
    return rows

if __name__ == "__main__":
    import sys
    packings = {n: grid_packing(n) for n in range(1, 1001)}
    rows = build(packings, "outputs/grid_submission.csv")
    print("rows:", rows)
