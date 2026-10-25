"""v3 seam campaign: wider clamps, stronger shrink, multi-config per n.

Usage: python seam_campaign3.py <n> [more_ns...]   (qwen env python)
Writes logs/seam_results/3_<n>.json
"""
import json, math, os, sys, time
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts/kirill_src"))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from packing_geometry import side, normalize, repair
from seam_wide import seam_pressure
from cpu_contact_polish import polish
from seam_campaign import load_packing, OUT

# (tag, kwargs for seam_pressure)
CONFIGS = [
    ("sh0002", dict(shrink=0.002, steps=1200, batch=6, pos_max=0.25, ang_max=0.05, kick_lo=0.0002, kick_hi=0.02,  kick_ang=0.0008)),
    ("sh0010", dict(shrink=0.010, steps=1500, batch=8, pos_max=0.25, ang_max=0.08, kick_lo=0.0002, kick_hi=0.03,  kick_ang=0.002)),
    ("sh0030", dict(shrink=0.030, steps=2000, batch=8, pos_max=0.35, ang_max=0.12, kick_lo=0.001,  kick_hi=0.05,  kick_ang=0.005)),
    ("wide",   dict(shrink=0.010, steps=1500, batch=8, pos_max=0.50, ang_max=0.15, kick_lo=0.001,  kick_hi=0.05,  kick_ang=0.005)),
]
SEEDS = (30011, 50023, 70037)

def main():
    ns = [int(x) for x in sys.argv[1:]]
    for n in ns:
        t0 = time.time()
        p0 = load_packing(n)
        L0 = side(p0)
        best = p0
        logs = []
        for tag, kw in CONFIGS:
            for seed in SEEDS:
                kw = dict(kw); kw.update(backend="cpu", seed=seed + n, seconds=300)
                try:
                    proposals, report = seam_pressure(p0, **kw)
                except Exception as e:
                    logs.append({"tag": tag, "seed": seed, "error": str(e)[:200]})
                    continue
                logs.append({"tag": tag, "seed": seed, "side_best": (report["proposal_sides"][0] if report["proposal_sides"] else None), "seconds": round(report["seconds"],1)})
                for q in proposals[:3]:
                    if side(q) > side(best) + 0.02:
                        continue
                    try:
                        q2, _, _ = polish(q, rounds=80, max_seconds=15, trust_start=0.02)
                    except Exception:
                        continue
                    if q2 is not None and side(q2) < side(best) - 1e-9:
                        best = q2
        accepted = False
        final_pts = best.tolist()
        if side(best) < L0 - 1e-9:
            r = repair(best, clearance=2e-8)
            if r is not None and side(r) < L0 - 1e-9:
                best, final_pts, accepted = r, r.tolist(), True
        res = {"n": n, "before": L0, "after": side(best), "gain": (L0 - side(best)) / math.sqrt(n),
               "accepted": accepted, "seconds": time.time() - t0, "logs": logs, "pts": final_pts}
        json.dump(res, open(os.path.join(OUT, f"3_{n}.json"), "w"))
        print(json.dumps({k: res[k] for k in ("n", "before", "after", "gain", "accepted", "seconds")}), flush=True)

if __name__ == "__main__":
    main()
