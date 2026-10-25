"""Convert old-format full_{scen}.npz (nested idx dict) to flat idx_* keys."""
import sys
from pathlib import Path
import numpy as np

scen = sys.argv[1]
p = Path(f"models/cache/full_{scen}.npz")
z = np.load(p, allow_pickle=True)
if "idx" not in z.files:
    print("already flat format, nothing to do")
    sys.exit(0)
x = z["idx"]
if isinstance(x, dict):
    d = x
elif x.ndim == 0:
    d = x[()]
else:
    d = x
if not isinstance(d, dict):
    raise SystemExit("unexpected idx container: %r" % type(d))
data = {k: z[k] for k in z.files if k != "idx"}
data.update({f"idx_{k}": v.astype(np.float32) for k, v in d.items()})
tmp = p.with_suffix(".tmp.npz")
np.savez_compressed(tmp, **data)
z.close()
tmp.rename(p)
print("converted", p)
