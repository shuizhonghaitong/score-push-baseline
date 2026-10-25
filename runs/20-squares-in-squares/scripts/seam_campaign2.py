"""Escalated seam campaign: longer runs, more seeds, wider kicks."""
import json, math, os, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts/kirill_src"))
from packing_geometry import side, normalize
from portable_seam_search import seam_pressure
from cpu_contact_polish import polish

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from seam_campaign import load_packing, OUT

def main():
    ns = [int(x) for x in sys.argv[1:]]
    for n in ns:
        t0 = time.time()
        p0 = load_packing(n)
        L0 = side(p0)
        best = p0
        for seed in (81173 + n, 424242 + n, 777 + n):
            cur = p0 if seed == 81173 + n else best
            try:
                proposals, report = seam_pressure(cur, backend="cpu", seed=seed, batch=6, steps=900, seconds=240)
            except Exception as e:
                print(f"n={n} seed={seed} failed: {e}", flush=True)
                continue
            for q in proposals[:4]:
                if side(q) > side(best) + 0.02:
                    continue
                try:
                    q2, _, _ = polish(q, rounds=80, max_seconds=15, trust_start=0.02)
                except Exception as e:
                    print(f"n={n} polish failed: {e}", flush=True)
                    continue
                if side(q2) < side(best) - 1e-9:
                    best = q2
        res = {"n": n, "before": L0, "after": side(best), "gain": (L0 - side(best)) / math.sqrt(n),
               "seconds": time.time() - t0, "pts": best.tolist()}
        json.dump(res, open(os.path.join(OUT, f"2_{n}.json"), "w"))
        print(json.dumps({k: res[k] for k in ("n", "before", "after", "gain", "seconds")}), flush=True)

if __name__ == "__main__":
    main()
