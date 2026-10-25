"""Local sanity checks for a merged TFB submission.csv (no competition data needed)."""
import sys
import numpy as np
import pandas as pd

path = sys.argv[1]
sub = pd.read_csv(path, dtype={"submission_id": "int64", "task": str})
print("rows:", len(sub))
print("columns:", list(sub.columns))
assert list(sub.columns) == ["submission_id", "task", "speed_kmh", "flow_vph", "queue_pred", "path_flow"]
assert sub.submission_id.is_unique
assert sub.submission_id.is_monotonic_increasing
assert sub.isna().sum().sum() == 0
counts = sub.task.value_counts().to_dict()
print("task counts:", counts)
st, q, o = sub.task == "state", sub.task == "queue", sub.task == "odme"
print("state rows:", st.sum(), "queue rows:", q.sum(), "odme rows:", o.sum())
v = sub.speed_kmh[st].to_numpy(float)
f = sub.flow_vph[st].to_numpy(float)
print(f"state: speed min/median/max {v.min():.2f}/{np.median(v):.2f}/{v.max():.2f}  finite={np.isfinite(v).all()}")
print(f"state: flow  min/median/max {f.min():.2f}/{np.median(f):.2f}/{f.max():.2f}  finite={np.isfinite(f).all()}")
print("state share flow<50 vph (S_FD floor risk if >0.20 per panel/regime):", round(float((f < 50).mean()), 5))
qp = sub.queue_pred[q].to_numpy()
print(f"queue: binary={set(qp) <= {0.0,1.0}}  positives={int(qp.sum())} share={qp.mean():.5f}")
pf = sub.path_flow[o].to_numpy(float)
print(f"odme: min/median/max {pf.min():.4f}/{np.median(pf):.4f}/{pf.max():.1f}  finite={np.isfinite(pf).all()}  nonneg={(pf>=0).all()}")
zero_state = sub[(sub.task=="state") & (sub.speed_kmh==0)].shape[0]
zero_queue = sub[(sub.task=="queue") & (sub.queue_pred==0)].shape[0]
print("state rows with speed==0:", zero_state, "| queue rows with pred==0:", zero_queue)
print("sha256:", end=" ")
import hashlib
h = hashlib.sha256()
with open(path, "rb") as fh:
    for chunk in iter(lambda: fh.read(1 << 20), b""):
        h.update(chunk)
print(h.hexdigest()[:16])
print("VALIDATION OK")
