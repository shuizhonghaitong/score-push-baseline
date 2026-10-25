"""engine v3: coordinate-descent minimization of overlap energy.

E = sum over pairs max(0, pen_ij)^2 + sum over squares max(0, out_i)^2
Each step: for a square i (cycling in random order), try candidate
translations (+optional small rotations), keep the one minimizing the
local violation. Monotone in expectation with reheats.
"""
import math
import time

import numpy as np
from engine import corners, aabbs, candidate_pairs, sat_overlaps, check, measure_L, MARGIN


def _local_viol(pts, i, L):
    """(v, pen, out) for square i: worst neighbor penetration & boundary."""
    cor = corners(pts[[i]])
    x0, x1 = cor[0, 0, 0], cor[0, 0, 0]  # placeholder
    x0 = cor[0, :, 0].min(); x1 = cor[0, :, 0].max()
    y0 = cor[0, :, 1].min(); y1 = cor[0, :, 1].max()
    out = max(0.0, -x0, x1 - L, -y0, y1 - L)
    allcor = corners(pts)
    a0, a1, b0, b1 = aabbs(allcor)
    pen = -1e9
    n = len(pts)
    for j in range(i + 1, n):
        if a0[j] > a1[i]:
            break
        if b0[j] <= b1[i] and b0[i] <= b1[j]:
            ov = sat_overlaps(pts, np.array([i, j]), np.array([j, i]))
            pen = max(pen, float(ov.min(1).max()))
    for j in range(i - 1, -1, -1):
        if a1[j] < a0[i]:
            break
        if b0[j] <= b1[i] and b0[i] <= b1[j]:
            ov = sat_overlaps(pts, np.array([j, i]), np.array([i, j]))
            pen = max(pen, float(ov.min(1).max()))
    return pen, out


def relax_cd(pts, L, iters, rng, d0=0.04, rot_step=0.05, reheat_every=200,
             reheat_scale=0.05, verbose=False, vlog=None):
    n = len(pts)
    order = np.arange(n)
    best_E = math.inf
    best_pts = pts.copy()
    for it in range(iters):
        if it % reheat_every == 0 and it > 0:
            # reheat: random kick of a few random squares
            k = max(1, n // 50)
            idx = rng.choice(n, size=k, replace=False)
            for i in idx:
                pts[i, 0] += rng.uniform(-reheat_scale, reheat_scale)
                pts[i, 1] += rng.uniform(-reheat_scale, reheat_scale)
                pts[i, 2] += rng.uniform(-0.02, 0.02)
        np.random_ = None
        rng.shuffle(order)
        for i in order:
            pen, out = _local_viol(pts, i, L)
            v = max(pen, out)
            if v <= -MARGIN and out <= 1e-9:
                continue
            d = max(d0, 0.35 * v)
            base = v
            cx, cy, th = pts[i, 0], pts[i, 1], pts[i, 2]
            best = None
            cands = [(0.0, 0.0, 0.0)]
            for sx in (-1.0, 1.0):
                for sy in (-1.0, 1.0):
                    cands.append((sx * d, sy * d, 0.0))
            if th != 0.0 or rng.random() < 0.5:
                cands.append((0.0, 0.0, rot_step))
                cands.append((0.0, 0.0, -rot_step))
            for dx, dy, dth in cands:
                pts[i, 0] = cx + dx
                pts[i, 1] = cy + dy
                pts[i, 2] = th + dth
                pen2, out2 = _local_viol(pts, i, L)
                v2 = max(pen2, out2)
                if best is None or v2 < best[0] - 1e-12:
                    best = (v2, dx, dy, dth)
            _, dx, dy, dth = best
            pts[i, 0] = cx + dx
            pts[i, 1] = cy + dy
            pts[i, 2] = th + dth
        if vlog and (it % vlog == 0 or it == iters - 1):
            pen, out = check(pts, L)
            E = max(pen, out)
            print(f"    it={it}: worst={E:.5f} (pen={pen:.4f} out={out:.4f})")
            if E < best_E:
                best_E = E
                best_pts = pts.copy()
    return pts, best_pts


def solve3(n, target_L, time_budget, seed=0, verbose=False, showcase=None,
           template_names=None):
    from engine2 import templates_for
    deadline = time.time() + time_budget
    K = int(math.isqrt(n - 1)) + 1
    while (K - 1) * (K - 1) >= n:
        K -= 1
    grid_L = float(K)
    if n == K * K:
        return None
    if showcase is not None and n in showcase:
        pts = np.array(showcase[n], float)
        Lm = measure_L(pts)
        pen, out = check(pts, Lm + 1e-9)
        if pen <= -MARGIN and out <= 1e-6:
            return (Lm, pts, Lm)

    best = None
    ladder = sorted(set(min(x, grid_L) for x in
                        [target_L, target_L + 0.06, target_L + 0.18, target_L + 0.5, grid_L]))
    weights = [0.5, 0.2, 0.15, 0.1, 0.05][:len(ladder)]
    ws = sum(weights)
    weights = [w / ws for w in weights]
    for li, L in enumerate(ladder):
        if time.time() > deadline:
            break
        t_budget = weights[li] * time_budget
        tmpl = templates_for(n, L, seed + li * 7)
        if template_names:
            tmpl = [t for t in tmpl if t[0] in template_names] or tmpl
        per_var = t_budget / len(tmpl)
        for vi, (name, pts0) in enumerate(tmpl):
            if time.time() > deadline:
                break
            t0 = time.time()
            rng = np.random.default_rng(seed * 99991 + li * 131 + vi * 17)
            pts = pts0.copy()
            its = max(400, int(time_budget * 30 / (len(ladder) * len(tmpl))))
            its = min(its, 30000)
            d0 = 0.05 if n <= 40 else 0.03
            pts, best_pts = relax_cd(pts, L, its, rng, d0=d0,
                                     verbose=verbose,
                                     vlog=200 if verbose else None)
            pen, out = check(best_pts, L)
            feas = pen <= -MARGIN and out <= 1e-9
            if verbose:
                print(f"  n={n} L={L:.4f} {name}: pen={pen:.2e} out={out:.2e} feas={feas} t={time.time()-t0:.1f}")
            if feas:
                Lm = measure_L(best_pts)
                if best is None or Lm < best[0] - 1e-9:
                    best = (Lm, best_pts.copy(), L)
                # polish
                Ll = L
                pts_cur = best_pts
                for _ in range(150):
                    if time.time() > deadline:
                        break
                    Ln = Ll - 0.003
                    if Ln < math.sqrt(n) + 1e-9:
                        break
                    rng2 = np.random.default_rng(rng.integers(1 << 30))
                    p2, b2 = relax_cd(pts_cur, Ln, 250, rng2, d0=0.02, reheat_scale=0.01)
                    pen2, out2 = check(b2, Ln)
                    if pen2 <= -MARGIN and out2 <= 1e-9:
                        Ll = Ln
                        pts_cur = b2
                        Lm = measure_L(pts_cur)
                        if best is None or Lm < best[0] - 1e-9:
                            best = (Lm, pts_cur.copy(), Ll)
                    else:
                        break
                if Ll <= target_L + 1e-9:
                    break
        if best is not None and best[2] <= target_L + 1e-9:
            break
    return best
