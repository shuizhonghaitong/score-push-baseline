"""Local scorer mirroring the Kaggle metric: L = max corner x/y span; SAT overlap check.

Feasibility policy (safer than the metric): every pair must be separated with a
margin >= SEP_MARGIN in float SAT (metric rejects overlap > 1e-8; pairs within
1e-8 go to interval certification). We require clear separation to avoid
undecidable edge cases.
"""
import math

SEP_MARGIN = 1e-7

def corners(cx, cy, ang):
    c = math.cos(ang); s = math.sin(ang); h = 0.5
    dx1, dy1 = h*c - h*s, h*s + h*c
    dx2, dy2 = -h*c - h*s, -h*s + h*c
    return [(cx+dx1, cy+dy1), (cx+dx2, cy+dy2), (cx-dx1, cy-dy1), (cx-dx2, cy-dy2)]

def sat_overlap(c1, c2, axes):
    """Return min projection overlap (>0 => overlap, <0 => separated)."""
    min_overlap = float('inf')
    for ax, ay in axes:
        p1 = [ax*x + ay*y for x, y in c1]
        p2 = [ax*x + ay*y for x, y in c2]
        ov = min(max(p1), max(p2)) - max(min(p1), min(p2))
        if ov < -SEP_MARGIN:
            return ov
        min_overlap = min(min_overlap, ov)
    return min_overlap

def score_packing(squares):
    """squares: list of (x, y, angle). Return (L, feasible, min_margin, worst_pair)."""
    n = len(squares)
    cs = [corners(*sq) for sq in squares]
    xs = [x for c in cs for x, y in c]
    ys = [y for c in cs for x, y in c]
    L = max(max(xs) - min(xs), max(ys) - min(ys))
    if n < 2:
        return L, True, float('inf'), None
    axes_all = []
    for _, _, a in squares:
        c = math.cos(a); s = math.sin(a)
        axes_all.append((c, s))
    # unique-ish axes: for each pair use 4 axes (2 from each square)
    min_margin = float('inf'); worst = None
    # sweep by x-extent
    order = sorted(range(n), key=lambda i: min(x for x, y in cs[i]))
    ext = [(min(x for x, y in cs[i]), max(x for x, y in cs[i]),
            min(y for x, y in cs[i]), max(y for x, y in cs[i])) for i in range(n)]
    for i_idx in range(n):
        i = order[i_idx]
        for j_idx in range(i_idx + 1, n):
            j = order[j_idx]
            if ext[j][0] > ext[i][2]:
                break
            if not (ext[i][2] < ext[j][0] or ext[j][2] < ext[i][0] or ext[i][3] < ext[j][1] or ext[j][3] < ext[i][1]):
                a1 = squares[i][2]; a2 = squares[j][2]
                c1 = math.cos(a1); s1 = math.sin(a1)
                c2 = math.cos(a2); s2 = math.sin(a2)
                axes = [(c1, s1), (-s1, c1), (c2, s2), (-s2, c2)]
                ov = sat_overlap(cs[i], cs[j], axes)
                if ov > min_margin:
                    min_margin = ov; worst = (i, j, ov)
                if ov > SEP_MARGIN:
                    return L, False, min_margin, worst
    return L, True, min_margin, worst

def puzzle_score(n, L):
    return L / math.sqrt(n) - 1.0

def grid_packing(n):
    """Axis-aligned ceil(sqrt(n)) grid. Returns list of (x, y, angle)."""
    k = math.isqrt(n - 1) if n > 1 else 1
    while k * k < n:
        k += 1
    out = []
    for i in range(n):
        r, c = divmod(i, k)
        out.append((c + 0.5, r + 0.5, 0.0))
    return out
