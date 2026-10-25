# Oct-25 extra blend: G + E2 + C2 (CPU v3 members), canonical order via order files.
import sys
import numpy as np
import pandas as pd
sys.path.insert(0, "work")
from blend_common import K, CANON, to_canon, sinkhorn, make_sub

g = to_canon(np.load("work/G_out/final_lp.npy"), "work/G_out/final_lp_order.npy", "G")
e = to_canon(np.load("work/E2_out/fork_probs.npz")["logit"], "work/E2_out/probs_order.npy", "E2")
lp = np.load("work/C2_out/members_lp.npy")  # (M, 2699, K) in C2's LABELS order
order = np.load("work/C2_out/members_order.npy")  # per CANON label: column index in C2 order
lp = lp[:, :, order]  # now (M, 2699, K) canonical
c2 = np.sum(np.log(np.clip(np.exp(lp - lp.max(2, keepdims=True)), 1e-12, 1)), axis=0)
# equal-weight log-prob blend of C2 members, renormalized
c2 = c2 - c2.max(1, keepdims=True)
c2 = np.exp(c2); c2 /= c2.sum(1, keepdims=True)
c2 = np.log(np.clip(c2, 1e-12, 1))
mix = 0.55 * g + 0.25 * e + 0.20 * c2
P = np.exp(mix - mix.max(1, keepdims=True)); P /= P.sum(1, keepdims=True)
P = sinkhorn(P)
ids = pd.read_parquet("work/G_out/submission.parquet")["id"].astype(str)
sub = make_sub(ids, P)
sub.to_parquet("work/blend_GEC2.parquet", index=False)
print("blend GEC2 written:", sub["model"].value_counts().to_dict())
