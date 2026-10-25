"""Apply Deotte-style clearance repair to non-grid store entries, re-measure L.

repair(p, clearance): translate so min corner = 0, then scale centers by the
smallest f >= 1 such that every near pair (KDTree < 1.42) has projection gap
>= clearance on all 4 SAT axes.
"""
import json
import sys

import numpy as np
from scipy.spatial import cKDTree


def half_ext(ang):
    return 0.5 * (np.abs(np.cos(ang)) + np.abs(np.sin(ang)))


def side(p):
    he = half_ext(p[:, 2])
    return float(max(np.ptp(np.r_[p[:, 0] - he, p[:, 0] + he]),
                     np.ptp(np.r_[p[:, 1] - he, p[:, 1] + he])))


def repair(p, clearance=2e-8):
    p = np.array(p, dtype=np.float64).copy()
    he = half_ext(p[:, 2])
    p[:, 0] -= (p[:, 0] - he).min()
    p[:, 1] -= (p[:, 1] - he).min()
    pairs = cKDTree(p[:, :2]).query_pairs(1.42, output_type="ndarray")
    if len(pairs):
        i, j = pairs.T
        d = p[j, :2] - p[i, :2]
        a, b = p[i, 2], p[j, 2]
        proj = np.maximum.reduce([
            np.abs(d[:, 0] * np.cos(a) + d[:, 1] * np.sin(a)),
            np.abs(-d[:, 0] * np.sin(a) + d[:, 1] * np.cos(a)),
            np.abs(d[:, 0] * np.cos(b) + d[:, 1] * np.sin(b)),
            np.abs(-d[:, 0] * np.sin(b) + d[:, 1] * np.cos(b)),
        ])
        rad = 0.5 * (1.0 + np.abs(np.cos(a - b)) + np.abs(np.sin(a - b)))
        f = max(1.0, float(np.max((rad + clearance) / np.maximum(proj, 1e-30))))
        p[:, :2] *= f
        p[:, 0] -= (p[:, 0] - he).min()
        p[:, 1] -= (p[:, 1] - he).min()
    return p


def main():
    store = json.load(open("models/packings.json"))
    nrep = 0
    for k, v in store.items():
        if v["src"] == "grid":
            continue
        p = np.array(v["pts"], dtype=np.float64)
        pr = repair(p, clearance=2e-8)
        v["pts"] = pr.tolist()
        v["L"] = side(pr)
        v["repaired"] = True
        nrep += 1
    json.dump(store, open("models/packings.json", "w"))
    import math
    tot = sum(store[str(n)]["L"] / math.sqrt(n) - 1 for n in range(1, 1001))
    print(f"repaired {nrep}; total: {tot:.10f}")


if __name__ == "__main__":
    main()
