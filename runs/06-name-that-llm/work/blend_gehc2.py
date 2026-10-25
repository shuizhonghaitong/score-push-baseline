# Oct-25 blend: G + E2 + H + C2, canonical order via order files.
import numpy as np, pandas as pd, sys
sys.path.insert(0, "work")
from blend_common import K, CANON, to_canon, sinkhorn, make_sub
g = to_canon(np.load("work/G_out/final_lp.npy"), "work/G_out/final_lp_order.npy", "G")
e = to_canon(np.load("work/E2_out/fork_probs.npz")["logit"], "work/E2_out/probs_order.npy", "E2")
h = to_canon(np.load("work/H_out/forkH_probs.npz")["logit"], "work/H_out/probs_order.npy", "H")
lp = np.load("work/C2_out/members_lp.npy")
order = np.load("work/C2_out/members_order.npy")
c2 = np.sum(np.log(np.clip(np.exp(lp[:, :, order] - lp[:, :, order].max(2, keepdims=True)), 1e-12, 1)), axis=0)
c2 = c2 - c2.max(1, keepdims=True); c2 = np.exp(c2); c2 /= c2.sum(1, keepdims=True)
c2 = np.log(np.clip(c2, 1e-12, 1))
mix = 0.50 * g + 0.20 * e + 0.15 * h + 0.15 * c2
P = np.exp(mix - mix.max(1, keepdims=True)); P /= P.sum(1, keepdims=True)
P = sinkhorn(P)
ids = pd.read_parquet("work/G_out/submission.parquet")["id"].astype(str)
sub = make_sub(ids, P)
sub.to_parquet("work/blend_GEHC2.parquet", index=False)
print("blend GEHC2 written:", sub["model"].value_counts().to_dict())
