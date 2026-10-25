"""CPU-only geometry and file utilities for the Squares in Squares experiment."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
import math
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree, ConvexHull
from scipy.optimize import minimize_scalar


def grid(n):
    k = math.ceil(math.sqrt(n))
    return np.array([(i % k + .5, i // k + .5, 0.) for i in range(n)])


def aabbs(p):
    h = (np.abs(np.cos(p[:, 2])) + np.abs(np.sin(p[:, 2]))) / 2
    return p[:, :2] - h[:, None], p[:, :2] + h[:, None]


def side(p):
    lo, hi = aabbs(p)
    return float(np.max(hi.max(0) - lo.min(0)))


def normalize(p):
    p = np.array(p, dtype=float, copy=True)
    p[:, :2] -= aabbs(p)[0].min(0)
    return p


def rotate(p, angle):
    p = p.copy()
    c, s = math.cos(angle), math.sin(angle)
    p[:, :2] = p[:, :2] @ np.array([[c, s], [-s, c]])
    p[:, 2] += angle
    return normalize(p)


def vertices(p):
    u = np.array([[-.5, -.5], [.5, -.5], [.5, .5], [-.5, .5]])
    c, s = np.cos(p[:, 2]), np.sin(p[:, 2])
    return p[:, None, :2] + np.stack((c[:, None]*u[:, 0]-s[:, None]*u[:, 1],
                                     s[:, None]*u[:, 0]+c[:, None]*u[:, 1]), 2)


def pair_data(p, radius=1.55):
    pairs = cKDTree(p[:, :2]).query_pairs(radius, output_type="ndarray")
    if not len(pairs):
        return pairs, np.empty((0, 4)), np.empty(0)
    i, j = pairs.T
    d = p[j, :2] - p[i, :2]
    a, b = p[i, 2], p[j, 2]
    projections = np.column_stack((d[:, 0]*np.cos(a)+d[:, 1]*np.sin(a),
                                    -d[:, 0]*np.sin(a)+d[:, 1]*np.cos(a),
                                    d[:, 0]*np.cos(b)+d[:, 1]*np.sin(b),
                                    -d[:, 0]*np.sin(b)+d[:, 1]*np.cos(b)))
    support = (1 + np.abs(np.cos(a-b)) + np.abs(np.sin(a-b))) / 2
    return pairs, projections, support


def min_gap(p):
    _, projection, support = pair_data(p)
    return float(np.min(np.abs(projection).max(1)-support)) if len(support) else math.inf


def repair(p, clearance=3e-10):
    """Separate centres slightly without ever changing unit-square dimensions."""
    p = normalize(p)
    _, projection, support = pair_data(p)
    if len(support):
        ratio = float(np.max((support+clearance)/np.maximum(np.abs(projection).max(1), 1e-30)))
        if ratio > 1.005:
            return None
        p[:, :2] *= max(1., ratio)
    return normalize(p)


def load_bank(path="inputs/submission.csv"):
    import pandas as pd
    data = pd.read_csv(path, float_precision="round_trip")
    bank = {n: grid(n) for n in range(1, 1001)}
    for n, group in data.groupby("n"):
        n = int(n)
        assert len(group) == n
        bank[n] = group.sort_values("square_id")[["x", "y", "angle"]].to_numpy(float)
    return bank


def save_bank(path, bank):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{str(n): p for n, p in bank.items()})


def total_score(bank):
    return float(sum(side(bank[n])/math.sqrt(n)-1 for n in range(1, 1001)))


def optimal_rotation(p):
    """Minimize the enclosing square of the fixed full layout over global angle."""
    points = vertices(p).reshape(-1, 2)
    hull = points[ConvexHull(points).vertices]
    angles = np.linspace(-math.pi/4, math.pi/4, 721)
    c, s = np.cos(angles), np.sin(angles)
    xx = hull[:, 0, None]*c - hull[:, 1, None]*s
    yy = hull[:, 0, None]*s + hull[:, 1, None]*c
    lengths = np.maximum(np.ptp(xx, axis=0), np.ptp(yy, axis=0))
    indices = np.flatnonzero((lengths[1:-1] <= lengths[:-2]) &
                             (lengths[1:-1] <= lengths[2:])) + 1
    winner = (side(p), 0.)
    def objective(a):
        c, s = math.cos(a), math.sin(a)
        return max(np.ptp(hull[:, 0]*c-hull[:, 1]*s),
                   np.ptp(hull[:, 0]*s+hull[:, 1]*c))
    for k in indices:
        fit = minimize_scalar(objective, bounds=(angles[k-1], angles[k+1]),
                              method="bounded", options={"xatol": 1e-14})
        if fit.fun < winner[0]:
            winner = (float(fit.fun), float(fit.x))
    return rotate(p, winner[1]), winner[1]
