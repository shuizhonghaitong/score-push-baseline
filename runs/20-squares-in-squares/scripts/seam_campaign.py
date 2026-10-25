"""Run kirilltym-style seam-pressure search on selected puzzles (CPU, qwen env torch).

Usage: python seam_campaign.py <target_n> [more_ns...]
Writes logs/seam_results/<n>.json per target.
"""
import csv
import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts/kirill_src"))
from packing_geometry import side, normalize
from portable_seam_search import seam_pressure
from cpu_contact_polish import polish

SUB = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs/submission_run20c.csv")
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs/seam_results")

def load_packing(n):
    rows = {}
    with open(SUB) as f:
        for r in csv.DictReader(f):
            if int(r["n"]) == n:
                rows[int(r["square_id"])] = [float(r["x"]), float(r["y"]), float(r["angle"])]
    assert len(rows) == n, (n, len(rows))
    return np.array([rows[i] for i in range(n)])

def main():
    ns = [int(x) for x in sys.argv[1:]]
    os.makedirs(OUT, exist_ok=True)
    for n in ns:
        t0 = time.time()
        p0 = load_packing(n)
        L0 = side(p0)
        best, best_report = p0, None
        for seed in (81173 + n, 424242 + n):
            try:
                proposals, report = seam_pressure(p0 if seed == 81173 + n else best,
                                                  backend="cpu", seed=seed, batch=6, steps=240, seconds=45)
            except Exception as e:
                print(f"n={n} seed={seed} seam_pressure failed: {e}", flush=True)
                continue
            for q in proposals[:3]:
                if side(q) > side(best) + 0.02:
                    continue
                try:
                    q2, _, _ = polish(q, rounds=60, max_seconds=10, trust_start=0.015)
                except Exception as e:
                    print(f"n={n} polish failed: {e}", flush=True)
                    continue
                if side(q2) < side(best) - 1e-9:
                    best, best_report = q2, report
        dt = time.time() - t0
        res = {"n": n, "before": L0, "after": side(best), "gain": (L0 - side(best)) / math.sqrt(n),
               "seconds": dt, "pts": best.tolist()}
        json.dump(res, open(os.path.join(OUT, f"{n}.json"), "w"))
        print(json.dumps({k: res[k] for k in ("n", "before", "after", "gain", "seconds")}), flush=True)

if __name__ == "__main__":
    main()
