"""One-shot CV: v1 (no block features) vs v2 (+block-anomaly feature), 3 folds."""
import sys, time
from pathlib import Path
import numpy as np

t0 = time.time()
sys.path.insert(0, "scripts")
import fullres_linear as fl

scens = ["ssp126", "ssp370", "ssp585"]
missing = [s for s in scens if not (fl.CACHE / f"full_{s}.npz").exists()]
if missing:
    print(f"[cv2] missing npz: {missing} — aborting", flush=True); sys.exit(2)
missing += [s for s in scens if not (fl.CACHE / f"lite_cv_idx_v9_{s}.npz").exists()]
if missing:
    print(f"[cv2] missing lite cv: {missing} — aborting", flush=True); sys.exit(2)

data = {s: fl.load_full(s) for s in scens}
print(f"[cv2] loaded in {time.time()-t0:.1f}s", flush=True)
lite_cv = {}
for s in scens:
    z = np.load(fl.CACHE / f"lite_cv_idx_v9_{s}.npz")
    lite_cv[s] = {k: z[k].astype(np.float32) for k in fl.IDX_NAMES}
    z.close()

v1, v2 = {}, {}
for ho in scens:
    tr = [s for s in scens if s != ho]
    t1 = time.time()
    p1 = fl.emulate(tr, data[ho]["lbc"], None, ridge=1e-3)
    S1, per1 = fl.fullres_nnse(data[ho]["idx"], p1)
    t2 = time.time()
    p2 = fl.emulate(tr, data[ho]["lbc"], None, ridge=1e-3,
                    block_pred={s: lite_cv[s] for s in tr},
                    test_block_pred=lite_cv[ho])
    S2, per2 = fl.fullres_nnse(data[ho]["idx"], p2)
    v1[ho], v2[ho] = per1, per2
    print(f"[cv2] ho={ho} v1 S={S1:+.5f} ({time.time()-t1:.0f}s)  v2 S={S2:+.5f} ({time.time()-t2:.0f}s)", flush=True)
    for k in fl.IDX_NAMES:
        d = per2[k] - per1[k]
        print(f"    {k:8s} v1 {per1[k]:+.4f}  v2 {per2[k]:+.4f}  d {d:+.4f}", flush=True)

w = {"ssp126": 0.4, "ssp370": 0.4, "ssp585": 0.2}
for tag, D in (("v1", v1), ("v2", v2)):
    avg = {k: float(np.mean([D[ho][k] for ho in scens])) for k in fl.IDX_NAMES}
    est = {k: sum(w[ho] * D[ho][k] for ho in scens) for k in fl.IDX_NAMES}
    print(f"\n== {tag}: 3-fold mean / weighted(0.4,0.4,0.2) ==")
    for k in fl.IDX_NAMES:
        print(f"    {k:8s} {avg[k]:+.4f}  {est[k]:+.4f}")
    print(f"    MEAN    {float(np.mean(list(avg.values()))):+.4f}  {float(np.mean(list(est.values()))):+.4f}")

out = Path("models/cache/cv_fullres_v2_result.txt")
with open(out, "w") as f:
    f.write("ho | idx | v1 | v2\n")
    for ho in scens:
        for k in fl.IDX_NAMES:
            f.write(f"{ho} {k} {v1[ho][k]:+.4f} {v2[ho][k]:+.4f}\n")
    for tag, D in (("v1", v1), ("v2", v2)):
        est = {k: sum(w[ho] * D[ho][k] for ho in scens) for k in fl.IDX_NAMES}
        f.write(f"{tag} weighted MEAN {float(np.mean(list(est.values()))):+.4f}\n")
print(f"[cv2] wrote {out} total {time.time()-t0:.0f}s", flush=True)
