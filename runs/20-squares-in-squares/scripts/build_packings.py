"""Build the packing store from Friedman/Ellsworth SVG parses + showcase + grid.

Output: models/packings.json
  n -> {"pts": [[x, y, theta_deg], ...], "s": <decimal str>, "src": str, "feas": bool,
        "L": <measured float or null>}
For each SVG group (label "a, b, c" -> square-M.svg), every n in the group gets the
full M-square packing; for n < M we drop (M - n) squares and choose the drop set
greedily to minimize measured L.
"""
import json
import math
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from score_local import score_packing, puzzle_score, grid_packing


def parse_txt(path):
    pts = []
    s_val = None
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        m = re.match(r"Final s:\s*([0-9.]+)", line)
        if m:
            s_val = m.group(1)
            continue
        m = re.match(r"Square \d+:\s*x=([-+0-9.eE]+),\s*y=([-+0-9.eE]+),\s*deg=([-+0-9.eE]+)", line)
        if m:
            pts.append([float(m.group(1)), float(m.group(2)), float(m.group(3)) * math.pi / 180.0])
    return pts, s_val


def drop_squares(pts, keep):
    """Drop len(pts)-keep squares, greedily minimizing measured L (try dropping each
    square in order of 'exposedness': corner-ness heuristic + greedy check)."""
    if keep >= len(pts):
        return pts
    arr = [list(p) for p in pts]
    target = keep
    while len(arr) > target:
        best_i, best_L = None, None
        # candidates: squares with a corner on the overall bounding box (most exposed)
        cor = [ (i, score_packing(arr)[0]) for i in [] ]
        Lfull = None
        if len(arr) <= 60:
            cands = list(range(len(arr)))
        else:
            cands = sorted(range(len(arr)), key=lambda i: -(abs(arr[i][0]) + abs(arr[i][1])))[:60]
        for i in cands:
            trial = arr[:i] + arr[i+1:]
            L, ok, m, _ = score_packing(trial)
            if not ok:
                continue
            if best_L is None or L < best_L - 1e-12:
                best_L, best_i = L, i
        if best_i is None:
            # nothing can be dropped without overlap?? shouldn't happen; drop most extreme
            best_i = max(range(len(arr)), key=lambda i: abs(arr[i][0]) + abs(arr[i][1]))
        arr = arr[:best_i] + arr[best_i+1:]
    return arr


def main():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    items = json.load(open(os.path.join(base, "data/friedman/page_items.json")))
    show = {int(k): v for k, v in json.load(open("logs/deotte_payload.json"))["showcase"].items()}

    store = {}
    # grid fallback for every n
    for n in range(1, 1001):
        store[n] = {"pts": grid_packing(n), "s": str(math.ceil(math.sqrt(n))),
                    "src": "grid", "feas": True, "L": float(math.ceil(math.sqrt(n)))}

    # SVG groups (use the page's own svg filename, incl. lettered variants)
    for it in items:
        svg = it.get("svg")
        if not svg:
            continue
        f = os.path.join(base, "data/friedman/parse", svg[:-4] + ".txt")
        if not os.path.exists(f):
            print("missing parse for", f)
            continue
        pts, s_val = parse_txt(f)
        M = len(pts)
        for n in it["ns"]:
            if n > 1000:
                continue
            if n == M:
                chosen = pts
            else:
                chosen = drop_squares(pts, n)
            L, ok, m, _ = score_packing(chosen)
            # only use if better than current (grid) AND feasible with margin
            cur = store[n]["L"]
            if ok and L < cur - 1e-12:
                store[n] = {"pts": chosen, "s": s_val, "src": f"svg-{svg[:-4]}",
                            "feas": True, "L": L, "margin": float(m)}
            else:
                print(f"n={n}: svg not usable (ok={ok} L={L} cur={cur} m={m})")

    # showcase packings (better than svg for some n)
    for n, pts in show.items():
        if n > 1000:
            continue
        L, ok, m, _ = score_packing(pts)
        cur = store[n]["L"]
        if ok and L < cur - 1e-12:
            store[n] = {"pts": pts, "s": None, "src": "showcase", "feas": True, "L": L,
                        "margin": float(m)}
        elif ok:
            store[n]["margin"] = min(store[n].get("margin", float("inf")), float(m))
        else:
            print(f"showcase n={n} NOT feasible locally?! L={L} m={m}")

    json.dump({str(k): v for k, v in store.items()},
              open(os.path.join(base, "models/packings.json"), "w"))
    tot = 0.0
    nused = 0
    for n in range(1, 1001):
        tot += puzzle_score(n, store[n]["L"])
        if store[n]["src"] != "grid":
            nused += 1
    print(f"total: {tot:.6f}  non-grid puzzles: {nused}")


if __name__ == "__main__":
    main()
