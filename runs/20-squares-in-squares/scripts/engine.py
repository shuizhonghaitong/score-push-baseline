"""Force-based packing optimizer: n unit squares (position + rotation) in
the smallest axis-aligned bounding square.

Feasibility at fixed L: all corners inside [0,L]^2, no pair of squares
overlapping (SAT, with a strict separation margin for float safety).

CLI: python engine.py --n 17 --target 4.675530 --time 60 [--seed 0] [--out f]
"""
import argparse
import json
import math
import time

import numpy as np

MARGIN = 5e-7  # required strict separation (metric rejects overlap > 1e-8)


# ---------------- geometry ----------------

def corners(pts):
    c = np.cos(pts[:, 2]); s = np.sin(pts[:, 2])
    h = 0.5
    dx1 = h * c - h * s
    dy1 = h * s + h * c
    dx2 = -h * c - h * s
    dy2 = -h * s + h * c
    cx, cy = pts[:, 0], pts[:, 1]
    out = np.empty((len(pts), 4, 2))
    out[:, 0, 0] = cx + dx1; out[:, 0, 1] = cy + dy1
    out[:, 1, 0] = cx + dx2; out[:, 1, 1] = cy + dy2
    out[:, 2, 0] = cx - dx1; out[:, 2, 1] = cy - dy1
    out[:, 3, 0] = cx - dx2; out[:, 3, 1] = cy - dy2
    return out


def aabbs(cor):
    return (cor[:, :, 0].min(1), cor[:, :, 0].max(1),
            cor[:, :, 1].min(1), cor[:, :, 1].max(1))


def candidate_pairs(a0, a1, b0, b1):
    """All pairs with overlapping AABBs (sweep with per-i window via searchsorted)."""
    n = len(a0)
    order = np.argsort(a0, kind="stable")
    xs0, xs1, ys0, ys1 = a0[order], a1[order], b0[order], b1[order]
    out_i, out_j = [], []
    for i_idx in range(n):
        lo = i_idx + 1
        hi = int(np.searchsorted(xs0, xs1[i_idx] + 1e-12, side="right"))
        if hi <= lo:
            continue
        win = np.arange(lo, hi)
        m = (ys0[win] <= ys1[i_idx] + 1e-12) & (ys0[i_idx] <= ys1[win] + 1e-12)
        for j_idx in win[m]:
            a, b = order[i_idx], order[j_idx]
            if a < b:
                out_i.append(a); out_j.append(b)
            else:
                out_i.append(b); out_j.append(a)
    return np.array(out_i, int), np.array(out_j, int)


def sat_overlaps(pts, pi, pj):
    """(P,4) overlaps: axes 0,1 of square i; axes 2,3 of square j."""
    if len(pi) == 0:
        return np.empty((0, 4))
    c1 = np.cos(pts[pi, 2]); s1 = np.sin(pts[pi, 2])
    c2 = np.cos(pts[pj, 2]); s2 = np.sin(pts[pj, 2])
    cor1 = corners(pts[pi]); cor2 = corners(pts[pj])
    ac = np.stack([c1, -s1, c2, -s2], 1)
    as_ = np.stack([s1, c1, s2, c2], 1)
    x1 = cor1[:, :, 0]; y1 = cor1[:, :, 1]
    x2 = cor2[:, :, 0]; y2 = cor2[:, :, 1]
    pr1 = ac[:, :, None] * x1[:, None, :] + as_[:, :, None] * y1[:, None, :]
    pr2 = ac[:, :, None] * x2[:, None, :] + as_[:, :, None] * y2[:, None, :]
    return np.minimum(pr1.max(2), pr2.max(2)) - np.maximum(pr1.min(2), pr2.min(2))


def check(pts, L):
    """Return (pen, out): max penetration (best axis) and max boundary violation."""
    cor = corners(pts)
    a0, a1, b0, b1 = aabbs(cor)
    pi, pj = candidate_pairs(a0, a1, b0, b1)
    pen = -math.inf
    if len(pi):
        ov = sat_overlaps(pts, pi, pj)
        pen = float(ov.min(1).max())
    x0, x1, y0, y1 = aabbs(cor)
    out = float(max(-x0.min(), x1.max() - L, -y0.min(), y1.max() - L))
    return pen, out


def measure_L(pts):
    cor = corners(pts)
    return float(max(cor[:, :, 0].max() - cor[:, :, 0].min(),
                     cor[:, :, 1].max() - cor[:, :, 1].min()))


# ---------------- relax ----------------

def _push_apart(pts, pi, pj, ov, step):
    pen = ov.min(1)
    mask = pen > -MARGIN
    if not mask.any():
        return
    pi2, pj2 = pi[mask], pj[mask]
    pen2 = pen[mask]; ax2 = ov[mask].argmin(1)
    own = ax2 < 2
    odd = (ax2 % 2) == 1
    ci = np.cos(pts[pi2, 2]); si = np.sin(pts[pi2, 2])
    cj = np.cos(pts[pj2, 2]); sj = np.sin(pts[pj2, 2])
    ax_x = np.where(own, np.where(odd, -ci, ci), np.where(odd, -cj, cj))
    ax_y = np.where(own, np.where(odd, si, si), np.where(odd, sj, sj))
    d = 0.5 * pen2 * step
    dx = np.zeros((len(pts), 2))
    v = np.stack([ax_x, ax_y], 1) * d[:, None]
    np.add.at(dx, pi2, v)
    np.add.at(dx, pj2, -v)
    pts[:, :2] = pts[:, :2] + dx


def _push_inside(pts, L, step):
    cor = corners(pts)
    x0, x1, y0, y1 = aabbs(cor)
    px = np.where(x1 > L, -(x1 - L), np.where(x0 < 0, -x0, 0.0)) * 0.5 * step
    py = np.where(y1 > L, -(y1 - L), np.where(y0 < 0, -y0, 0.0)) * 0.5 * step
    pts[:, 0] += px
    pts[:, 1] += py


def _rotate_kicks(pts, L, rng, max_kicks=24, dtheta=0.03):
    """For squares in violation, try small angle changes; keep improving ones."""
    cor = corners(pts)
    a0, a1, b0, b1 = aabbs(cor)
    pi, pj = candidate_pairs(a0, a1, b0, b1)
    involved = set()
    if len(pi):
        ov = sat_overlaps(pts, pi, pj)
        pen = ov.min(1)
        for idx in np.where(pen > -2 * MARGIN)[0]:
            involved.add(int(pi[idx])); involved.add(int(pj[idx]))
    x0, x1, y0, y1 = aabbs(cor)
    for i in np.where((x1 > L - 0.05) | (x0 < 0.05) | (y1 > L - 0.05) | (y0 < 0.05))[0]:
        involved.add(int(i))
    involved = list(involved)
    if not involved:
        return
    rng.shuffle(involved)
    for i in involved[:max_kicks]:
        base = _local_energy(pts, i, L)
        th = pts[i, 2]
        for sign in (1.0, -1.0):
            pts[i, 2] = th + sign * dtheta
            e = _local_energy(pts, i, L)
            if e < base - 1e-12:
                base = e
            else:
                pts[i, 2] = th
                break


def _local_energy(pts, i, L):
    cor = corners(pts[[i]])
    x0 = cor[0, :, 0].min(); x1 = cor[0, :, 0].max()
    y0 = cor[0, :, 1].min(); y1 = cor[0, :, 1].max()
    out = max(0.0, -x0, x1 - L, -y0, y1 - L)
    allcor = corners(pts)
    a0, a1, b0, b1 = aabbs(allcor)
    jmax = i
    for j in range(i + 1, len(pts)):
        if a0[j] > a1[i]:
            break
        if b0[j] <= b1[i] and b0[i] <= b1[j]:
            ov = sat_overlaps(pts, np.array([i, j]), np.array([j, i]))
            out = max(out, float(ov.min(1).max()))
            jmax = j
    for j in range(i - 1, -1, -1):
        if a1[j] < a0[i]:
            break
        if b0[j] <= b1[i] and b0[i] <= b1[j]:
            ov = sat_overlaps(pts, np.array([j, i]), np.array([i, j]))
            out = max(out, float(ov.min(1).max()))
    return out


def relax(pts, L, iters, step, jitter, rng, do_rotate=True, rotate_every=2):
    for it in range(iters):
        fr = it / max(1, iters - 1)
        j = jitter * (1.0 - min(1.0, fr / 0.85)) ** 2  # anneal to 0 in last 15%
        cor = corners(pts)
        a0, a1, b0, b1 = aabbs(cor)
        pi, pj = candidate_pairs(a0, a1, b0, b1)
        if len(pi):
            ov = sat_overlaps(pts, pi, pj)
            _push_apart(pts, pi, pj, ov, step)
        _push_inside(pts, L, step)
        if do_rotate and it % rotate_every == 0:
            _rotate_kicks(pts, L, rng)
        if j > 0:
            pts[:, :2] += rng.uniform(-j, j, (len(pts), 2))
    return pts


# ---------------- solve ----------------

def init_variants(n, seed):
    """Yield (name, pts) initial configurations."""
    rng = np.random.default_rng(seed)
    k = 1
    while k * k < n:
        k += 1
    K = k          # K*K >= n
    kfloor = K - 1 # kfloor*kfloor < n (for n>1)
    r = n - kfloor * kfloor if n > 1 else 1

    def block(rows):
        """grid of axis-aligned squares, list of (row, col)."""
        pts = []
        for (rr, cc) in rows:
            pts.append([cc + 0.5, rr + 0.5, 0.0])
        return np.array(pts, float)

    # V0: kfloor x kfloor grid + r extras in a row on top, axis-aligned
    rows = [(rr, cc) for rr in range(kfloor) for cc in range(kfloor)]
    for c in range(r):
        rows.append((kfloor, c))
    yield "grid_row", block(rows)

    # V1: same but extras rotated 45deg (diamonds), spacing sqrt(2)/1... width sqrt(2)
    rows = [(rr, cc) for rr in range(kfloor) for cc in range(kfloor)]
    pts = []
    for (rr, cc) in rows:
        pts.append([cc + 0.5, rr + 0.5, 0.0])
    for c in range(r):
        pts.append([c * 1.0 + 0.5, kfloor + 0.55, math.pi / 4])
    yield "diamond_row", np.array(pts, float)

    # V2: K x K grid minus holes, holes = top row first (remove from top-right)
    if n < K * K:
        m = K * K - n
        removed = set()
        # remove from top rows, right to left
        for rr in range(K - 1, -1, -1):
            for cc in range(K - 1, -1, -1):
                if len(removed) >= m:
                    break
                removed.add((rr, cc))
            if len(removed) >= m:
                break
        rows = [(rr, cc) for rr in range(K) for cc in range(K) if (rr, cc) not in removed]
        yield "grid_holes", block(rows)

    # V3: kfloor x kfloor grid, extras rotated by random angles in the top row
    for t in range(2):
        pts = []
        for rr in range(kfloor):
            for cc in range(kfloor):
                pts.append([cc + 0.5, rr + 0.5, 0.0])
        thetas = rng.uniform(0.3, 0.9, r)
        x = 0.5
        for c in range(r):
            w = math.cos(thetas[c]) + math.sin(thetas[c])  # width when rotated
            pts.append([x + w / 2, kfloor + 0.5 + (w - 1) * 0.2, float(thetas[c])])
            x += w
        yield f"rand_rot_{t}", np.array(pts, float)


def solve(n, target_L, time_budget, seed=0, verbose=False):
    """Try to find a feasible packing with L <= target_L.
    Returns (L_meas, pts) of the best found (may be infeasible-free: always
    feasible if returned with L_meas <= L tried), else None if only grid.
    """
    deadline = time.time() + time_budget
    k = 1
    while k * k < n:
        k += 1
    grid_L = float(k)
    if n == k * k:
        return None  # perfect square: grid is optimal, score 0

    best = None  # (L_meas, pts, feasible_L)
    # L ladder: target, then increasing up to grid
    ladder = [target_L]
    lo = target_L
    while lo < grid_L - 1e-9:
        lo = min(grid_L, lo + 0.03)
        ladder.append(lo)

    variants = list(init_variants(n, seed))
    vidx = 0
    for L in ladder:
        if time.time() > deadline:
            break
        per_var = max(3000, int(time_budget * 0.6 / (len(ladder) * len(variants))))
        for (vname, pts0) in variants[vidx % len(variants):] + variants[:vidx % len(variants)]:
            vidx += 1
            if time.time() > deadline:
                break
            pts = pts0.copy()
            # warm at grid L
            pts = relax(pts, grid_L, 250, 0.9, 0.0, np.random.default_rng(seed + vidx), do_rotate=False)
            t0 = time.time()
            rng = np.random.default_rng(seed * 1000 + vidx + int(L * 1000))
            jitter = 0.04 if L > target_L + 1e-9 else 0.02
            pts = relax(pts, L, per_var, 0.9, jitter, rng)
            pen, out = check(pts, L)
            feas = pen <= -MARGIN and out <= 1e-9
            if verbose:
                print(f"  n={n} L={L:.4f} v={vname}: pen={pen:.2e} out={out:.2e} feas={feas}")
            if feas:
                Lm = measure_L(pts)
                if best is None or Lm < best[0] - 1e-12:
                    best = (Lm, pts.copy(), L)
                # polish downward
                Ll = L
                for _ in range(60):
                    if time.time() > deadline:
                        break
                    Ln = Ll - 0.004
                    if Ln < math.sqrt(n) + 1e-9:
                        break
                    pts2 = relax(pts, Ln, 400, 0.9, 0.0, rng, do_rotate=False)
                    pen2, out2 = check(pts2, Ln)
                    if pen2 <= -MARGIN and out2 <= 1e-9:
                        Ll = Ln
                        pts = pts2
                        Lm = measure_L(pts)
                        if best is None or Lm < best[0] - 1e-12:
                            best = (Lm, pts.copy(), Ll)
                    else:
                        break
                if Ll <= target_L + 1e-9:
                    break
        if best is not None and best[2] <= target_L + 1e-9:
            break
    return best


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--target", type=float, required=True)
    ap.add_argument("--time", type=float, default=60.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=str, default="")
    ap.add_argument("-v", action="store_true")
    a = ap.parse_args()
    best = solve(a.n, a.target, a.time, seed=a.seed, verbose=a.v)
    if best is None:
        print(json.dumps({"n": a.n, "status": "no_result"}))
    else:
        Lm, pts, _ = best
        pen, out = check(pts, min(Lm + 1e-9, Lm))  # info
        rec = {"n": a.n, "L": Lm, "pts": [[float(v) for v in row] for row in pts]}
        if a.out:
            json.dump(rec, open(a.out, "w"))
        print(json.dumps({"n": a.n, "L": Lm}))
