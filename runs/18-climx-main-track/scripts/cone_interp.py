"""Cone-interpolation prediction for ssp245:
pred_k(t) = I_k(2015) + w(t)*(I_126(t)-I_126(2015)) + (1-w(t))*(I_370(t)-I_370(2015))
w(t) = per-year least-squares blend weight of LBC_245 inside span{LBC_126, LBC_370}.
"""
import sys, time
import numpy as np
sys.path.insert(0, "scripts")
import fullres_linear as fl
from climx_common import make_submission_parquet
from pathlib import Path

t0 = time.time()
d126 = fl.load_full("ssp126")
d370 = fl.load_full("ssp370")
lbc245 = fl.test_lbc()

# standardize LBC dims over the 3 scenarios' ranges for a well-scaled per-year fit
L = np.vstack([d126["lbc"], d370["lbc"], lbc245])  # (258,4)
Lm, Ls = L.mean(0), L.std(0) + 1e-30
a126 = (d126["lbc"] - Lm) / Ls
a370 = (d370["lbc"] - Lm) / Ls
a245 = (lbc245 - Lm) / Ls
w = []
for t in range(fl.N_Y):
    d1, d2, x = a126[t], a370[t], a245[t]
    # solve x = w*d1 + (1-w)*d2  ->  x - d2 = w*(d1 - d2)
    dd = d1 - d2
    wgt = float(np.dot(x - d2, dd) / max(float(np.dot(dd, dd)), 1e-30))
    w.append(np.clip(wgt, 0.0, 1.0))
w = np.array(w)
print(f"[cone] per-year w (126 share): {np.round(w, 3)}", flush=True)

pred = {}
for k in fl.IDX_NAMES:
    i1 = d126["idx"][k]  # (86,192,288) f32
    i2 = d370["idx"][k]
    i1 = i1.astype(np.float64); i2 = i2.astype(np.float64)
    d1y = i1 - i1[0:1]
    d2y = i2 - i2[0:1]
    p = i1[0:1] + w[:, None, None] * d1y + (1 - w)[:, None, None] * d2y
    pred[k] = p
    print(f"[cone] {k:8s} land delta 2015->2100 = {p[-1].mean() - p[0].mean():+8.3f}", flush=True)

out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("submissions/submission_cone_v1.parquet")
p = make_submission_parquet({k: v.astype(np.float32) for k, v in pred.items()}, out)
print("wrote", p, f"{time.time()-t0:.0f}s", flush=True)

if len(sys.argv) > 2 and sys.argv[2] == "blend":
    # blend cone with the fullres prediction given in argv[3] (default v1)
    import pyarrow.parquet as pq
    import pyarrow as pa
    from climx_common import N_SUB_ROWS
    src = sys.argv[4] if len(sys.argv) > 4 else "submissions/submission_fullres_v1.parquet"
    t = pq.read_table(src)
    cols = {k: t.column(k).to_numpy().reshape(86, 192, 288).astype(np.float64) for k in fl.IDX_NAMES}
    blend = {k: 0.5 * (cols[k] + pred[k]) for k in fl.IDX_NAMES}
    out2 = out.with_name(out.stem + "_blend50.parquet")
    make_submission_parquet({k: v.astype(np.float32) for k, v in blend.items()}, out2)
    print("wrote", out2, flush=True)
