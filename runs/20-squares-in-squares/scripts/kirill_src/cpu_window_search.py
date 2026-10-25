"""Exact translated-window queries for a fixed angle, plus angle enumeration.

Unlike a few corner anchors, the sweep visits every critical window location.
It optimizes subsets without moving or resizing the inherited unit squares.
"""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMBA_NUM_THREADS", "1")
import argparse
import json
import math
import time
from pathlib import Path
import numpy as np
from numba import njit
from packing_geometry import (aabbs, grid, load_bank, normalize, optimal_rotation,
                              repair, rotate, save_bank, side, total_score)


@njit(cache=True)
def add_range(tree, lazy, size, left, right, delta):
    left += size
    right += size
    original_left, original_right = left, right - 1
    while left < right:
        if left & 1:
            tree[left] += delta
            lazy[left] += delta
            left += 1
        if right & 1:
            right -= 1
            tree[right] += delta
            lazy[right] += delta
        left >>= 1
        right >>= 1
    node = original_left >> 1
    while node:
        tree[node] = max(tree[node*2], tree[node*2+1]) + lazy[node]
        node >>= 1
    node = original_right >> 1
    while node:
        tree[node] = max(tree[node*2], tree[node*2+1]) + lazy[node]
        node >>= 1


@njit(cache=True)
def window_query(low, high, length, need):
    """Return a lower-left corner containing >=need complete square AABBs."""
    m = len(low)
    # A rectangle in the space of window lower-left corners for each square.
    start_x = high[:, 0] - length
    end_x = low[:, 0]
    start_y = high[:, 1] - length
    end_y = low[:, 1]
    ys = np.unique(np.concatenate((start_y, end_y)))
    yl = np.searchsorted(ys, start_y)
    yr = np.searchsorted(ys, end_y) + 1
    xs = np.concatenate((start_x, end_x))
    order = np.argsort(xs)
    size = 1
    while size < len(ys):
        size *= 2
    tree = np.zeros(2*size, np.int32)
    lazy = np.zeros(2*size, np.int32)
    k = 0
    while k < len(order):
        stop = k + 1
        x = xs[order[k]]
        while stop < len(order) and xs[order[stop]] == x:
            stop += 1
        for pos in range(k, stop):
            idx = order[pos]
            if idx < m:
                add_range(tree, lazy, size, yl[idx], yr[idx], 1)
        if tree[1] >= need:
            node = 1
            while node < size:
                if tree[node*2] >= tree[node*2+1]:
                    node *= 2
                else:
                    node = node*2 + 1
            y_index = node - size
            return True, x, ys[y_index]
        for pos in range(k, stop):
            idx = order[pos]
            if idx >= m:
                idx -= m
                add_range(tree, lazy, size, yl[idx], yr[idx], -1)
        k = stop
    return False, 0., 0.


def choose_subset(p, low, high, length, n, x, y):
    fits = ((low[:, 0] >= x-2e-12) & (low[:, 1] >= y-2e-12) &
            (high[:, 0] <= x+length+2e-12) &
            (high[:, 1] <= y+length+2e-12))
    indices = np.flatnonzero(fits)
    if len(indices) < n:
        raise ValueError("Sweep witness did not contain the required count")
    centre = np.array([x+length/2, y+length/2])
    radius = np.max(np.maximum(high[indices]-centre, centre-low[indices]), axis=1)
    indices = indices[np.argsort(radius, kind="stable")[:n]]
    return normalize(p[indices]), indices


def brute_query(low, high, length, need):
    for x in high[:, 0]-length:
        for y in high[:, 1]-length:
            if np.sum((low[:, 0] >= x-1e-12) & (low[:, 1] >= y-1e-12) &
                      (high[:, 0] <= x+length+1e-12) &
                      (high[:, 1] <= y+length+1e-12)) >= need:
                return True
    return False


def self_check():
    rng = np.random.default_rng(913)
    for m in range(2, 15):
        for _ in range(10):
            centres = rng.uniform(-3, 3, (m, 2))
            h = rng.uniform(.5, .71, m)
            low, high = centres-h[:, None], centres+h[:, None]
            length = float(rng.uniform(1.5, 5))
            need = int(rng.integers(1, m+1))
            got, x, y = window_query(low, high, length, need)
            assert got == brute_query(low, high, length, need), (m, length, need)
            if got:
                assert np.sum((low[:, 0] >= x-1e-12) & (low[:, 1] >= y-1e-12) &
                              (high[:, 0] <= x+length+1e-12) &
                              (high[:, 1] <= y+length+1e-12)) >= need
    # Deliberate tied starts/ends exercise closed-boundary contacts.
    p = grid(16)
    low, high = aabbs(p)
    for n in range(1, 17):
        assert window_query(low, high, math.ceil(math.sqrt(n)), n)[0]
    print("Window sweep agrees with independent enumeration on 130 random cases and exact grid contacts.", flush=True)


def search(angles, depth=2.5, tag="windows", limit=1000, input_path=None, baseline_path="inputs/submission.csv"):
    start = time.perf_counter()
    bank = load_bank(baseline_path)
    if input_path:
        z = np.load(input_path)
        for key in z.files:
            n = int(key)
            if side(z[key]) < side(bank[n]):
                bank[n] = z[key].copy()
    baseline = total_score(bank)
    baseline_sides = {n: side(p) for n, p in bank.items()}
    candidates = [(n, p.copy()) for n, p in bank.items()
                  if baseline_sides[n] < math.ceil(math.sqrt(n))-1e-7 and n <= limit]
    # Larger parents first: expose more potential subsets before local variants.
    candidates.sort(reverse=True, key=lambda x: x[0])
    history = []
    attempts = 0
    for angle_degrees in angles:
        angle = math.radians(angle_degrees)
        for index, (parent, raw) in enumerate(candidates):
            p = rotate(raw, angle)
            low, high = aabbs(p)
            parent_side = side(p)
            for n in range(parent, max(4, int(max(0., parent_side-depth)**2)), -1):
                incumbent = side(bank[n])
                if incumbent <= math.sqrt(n)+1e-8:
                    continue
                target = incumbent - 2e-7
                if parent_side-target > depth:
                    continue
                attempts += 1
                found, x, y = window_query(low, high, target, n)
                if not found:
                    continue
                lower, upper = math.sqrt(n), target
                for _ in range(24):
                    mid = (lower+upper)/2
                    ok, xx, yy = window_query(low, high, mid, n)
                    if ok:
                        upper, x, y = mid, xx, yy
                    else:
                        lower = mid
                q, indices = choose_subset(p, low, high, upper, n, x, y)
                # Subsets retain source feasibility. Dilation also protects rotation
                # and the newly serialized decimal coordinates.
                q = repair(q)
                actual = side(q) if q is not None else math.inf
                if actual < incumbent-1e-7:
                    bank[n] = q
                    record = dict(n=n, parent=parent, angle_degrees=angle_degrees,
                                  before=incumbent, after=actual,
                                  gain=(incumbent-actual)/math.sqrt(n),
                                  indices=indices.tolist(), window=[x, y, upper])
                    history.append(record)
                    print(json.dumps({k:v for k,v in record.items() if k != "indices"}), flush=True)
            if (index+1) % 30 == 0:
                print(json.dumps(dict(progress=index+1, parents=len(candidates), angle=angle_degrees,
                                      attempts=attempts, hits=len(history), seconds=time.perf_counter()-start,
                                      score=total_score(bank))), flush=True)
                save_bank(f"artifacts/{tag}_packings.npz", bank)
                Path(f"artifacts/{tag}_history.json").write_text(json.dumps(history, indent=2))
        print(json.dumps(dict(angle_complete=angle_degrees, score=total_score(bank),
                              elapsed_seconds=time.perf_counter()-start)), flush=True)
    save_bank(f"artifacts/{tag}_packings.npz", bank)
    Path(f"artifacts/{tag}_history.json").write_text(json.dumps(history, indent=2))
    print(json.dumps(dict(baseline=baseline, score=total_score(bank), hits=len(history),
                          attempts=attempts, seconds=time.perf_counter()-start)), flush=True)
    return bank, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--angles", nargs="+", type=float, default=[0.])
    parser.add_argument("--depth", type=float, default=2.5)
    parser.add_argument("--tag", default="windows")
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--input")
    parser.add_argument("--baseline", default="inputs/submission.csv")
    args = parser.parse_args()
    self_check()
    search(args.angles, args.depth, args.tag, args.limit, args.input, args.baseline)
