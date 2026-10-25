"""engine v2: template-driven force relaxation for squares-in-squares."""
import json
import math
import time

import numpy as np
from engine import corners, aabbs, candidate_pairs, sat_overlaps, check, measure_L, MARGIN


def _grid_pts(rows):
    return np.array([[c + 0.5, r + 0.5, 0.0] for (r, c) in rows], float)


def template_grid_row(n, K, L0):
    """k x k grid in lower-left + r extras in a row on top, centered in [0,L0]."""
    k = K - 1
    r = n - k * k
    rows = [(rr, cc) for rr in range(k) for cc in range(k)]
    for c in range(r):
        rows.append((k, c))
    return _grid_pts(rows)


def template_gobel(n, K, L0, rng):
    """k x k grid + r diamonds centered."""
    k = K - 1
    r = n - k * k
    pts = [(c + 0.5, rr + 0.5, 0.0) for rr in range(k) for c in range(k)]
    if r == 1:
        pts.append((L0 / 2, L0 / 2, math.pi / 4))
    else:
        # r diamonds in a row centered
        for i in range(r):
            x = L0 / 2 + (i - (r - 1) / 2) * 1.1
            pts.append((x, L0 / 2, math.pi / 4))
    return np.array(pts, float)


def template_core_frame(n, K, L0, m=None, theta=math.pi / 4, rng=None):
    """(m x m tilted core) + upright ring around it. For n = k^2 + r, k >= 4.
    Ring: (k+1) x (k+1) grid minus central m x m cells; trim ring from
    top-right to match the upright count n - m^2."""
    k = K - 1
    if m is None:
        m = max(2, k - 2)
    need_up = n - m * m
    # build full (k+1)^2 grid, mark central m x m hole (centered)
    hole_r0 = (K - m) // 2
    hole_c0 = (K - m) // 2
    cells = []
    for rr in range(K):
        for cc in range(K):
            if hole_r0 <= rr < hole_r0 + m and hole_c0 <= cc < hole_c0 + m:
                continue
            cells.append((rr, cc))
    # trim from top-right
    cells.sort(key=lambda t: (-(t[0] + 0.0) * K - t[1]))
    if len(cells) > need_up:
        cells = cells[:need_up]
    elif len(cells) < need_up:
        # add back from the hole edge (inner ring) - rare
        extra = []
        for rr in range(K):
            for cc in range(K):
                if (rr, cc) not in cells:
                    extra.append((rr, cc))
        cells += extra[:need_up - len(cells)]
    up = _grid_pts(cells)
    # tilted core: m x m at theta, spacing w_t, centered
    ct, st = math.cos(theta), math.sin(theta)
    w_t = ct + st
    s = w_t  # edge-to-edge spacing
    cx0 = L0 / 2 - (m - 1) * s / 2
    cy0 = L0 / 2 - (m - 1) * s / 2
    core = []
    for i in range(m):
        for j in range(m):
            core.append([cx0 + j * s, cy0 + i * s, theta])
    pts = np.vstack([up, np.array(core, float)])
    # shuffle core ids to avoid systematic order effects
    return pts


def templates_for(n, L0, seed):
    K = int(math.isqrt(n - 1)) + 1
    while (K - 1) * (K - 1) >= n:
        K -= 1
    rng = np.random.default_rng(seed)
    k = K - 1
    out = [("grid_row", template_grid_row(n, K, L0))]
    out.append(("gobel", template_gobel(n, K, L0, rng)))
    if k >= 4:
        out.append(("core45", template_core_frame(n, K, L0, m=k - 2, theta=math.pi / 4, rng=rng)))
        if k >= 6:
            m2 = max(2, k - 1)
            need = n - m2 * m2
            if need >= 0:
                out.append(("core45b", template_core_frame(n, K, L0, m=m2, theta=math.pi / 4, rng=rng)))
    out.append(("holes", template_holes(n, K, L0, rng)))
    return out


def template_holes(n, K, L0, rng):
    m = K * K - n
    removed = set()
    for rr in range(K - 1, -1, -1):
        for cc in range(K - 1, -1, -1):
            if len(removed) >= m:
                break
            removed.add((rr, cc))
        if len(removed) >= m:
            break
    rows = [(rr, cc) for rr in range(K) for cc in range(K) if (rr, cc) not in removed]
    return _grid_pts(rows)


def solve2(n, target_L, time_budget, seed=0, verbose=False, showcase=None):
    """Try to beat grid; return (L_meas, pts) best feasible with L_meas <= best tried, or None."""
    deadline = time.time() + time_budget
    K = int(math.isqrt(n - 1)) + 1
    while (K - 1) * (K - 1) >= n:
        K -= 1
    grid_L = float(K)
    if n == K * K:
        return None

    # showcase exact solution?
    if showcase is not None and n in showcase:
        from engine import check as _c
        pts = np.array(showcase[n], float)
        Lm = measure_L(pts)
        pen, out = _c(pts, Lm + 1e-9)
        if pen <= -MARGIN and out <= 1e-6:
            return (Lm, pts, Lm)

    best = None
    from engine import relax
    ladder = [target_L, target_L + 0.05, target_L + 0.15, target_L + 0.4, grid_L]
    ladder = sorted(set(min(x, grid_L) for x in ladder))
    # time weights: most to first (target)
    weights = [0.5, 0.2, 0.15, 0.1, 0.05][:len(ladder)]
    wsum = sum(weights)
    weights = [w / wsum for w in weights]

    for li, L in enumerate(ladder):
        if time.time() > deadline:
            break
        t_budget = weights[li] * time_budget
        tmpl = templates_for(n, L, seed + li * 7)
        per_var = max(1.0, t_budget / len(tmpl))
        for vi, (name, pts0) in enumerate(tmpl):
            if time.time() > deadline:
                break
            t0 = time.time()
            pts = pts0.copy()
            rng = np.random.default_rng(seed * 99991 + li * 131 + vi)
            # warm at grid L (settle overlaps of the template)
            pts = relax(pts, grid_L, 150, 0.9, 0.0, rng, do_rotate=False)
            its = max(300, int((time_budget * 0.9) / (len(ladder) * len(tmpl)) * 180))
            its = min(its, 20000)
            jitter = 0.03 if L > target_L + 1e-9 else 0.015
            pts = relax(pts, L, its, 0.9, jitter, rng)
            pen, out = check(pts, L)
            feas = pen <= -MARGIN and out <= 1e-9
            if verbose:
                print(f"  n={n} L={L:.4f} {name}: pen={pen:.2e} out={out:.2e} feas={feas} it={its} t={time.time()-t0:.1f}")
            if feas:
                Lm = measure_L(pts)
                if best is None or Lm < best[0] - 1e-9:
                    best = (Lm, pts.copy(), L)
                # polish downward
                Ll = L
                pts_cur = pts
                for _ in range(120):
                    if time.time() > deadline:
                        break
                    Ln = Ll - 0.004
                    if Ln < math.sqrt(n) + 1e-9:
                        break
                    pts2 = relax(pts_cur, Ln, 300, 0.9, 0.0, rng, do_rotate=False)
                    pen2, out2 = check(pts2, Ln)
                    if pen2 <= -MARGIN and out2 <= 1e-9:
                        Ll = Ln
                        pts_cur = pts2
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


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--target", type=float, required=True)
    ap.add_argument("--time", type=float, default=60.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("-v", action="store_true")
    a = ap.parse_args()
    best = solve2(a.n, a.target, a.time, seed=a.seed, verbose=a.v)
    if best is None:
        print(json.dumps({"n": a.n, "status": "no_result"}))
    else:
        Lm, pts, Lf = best
        print(json.dumps({"n": a.n, "L": Lm, "feasL": Lf}))
