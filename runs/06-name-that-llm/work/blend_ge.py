import os
import numpy as np
import pandas as pd
from blend_common import K, CANON, to_canon, sinkhorn, make_sub

g = to_canon(np.load("work/G_out/final_lp.npy"), "work/G_out/final_lp_order.npy", "G")
e = to_canon(np.load("work/E2_out/fork_probs.npz")["logit"], "work/E2_out/probs_order.npy", "E2")
assert g.shape == (2699, K) and e.shape == (2699, K), (g.shape, e.shape)
mix = 0.70 * g + 0.30 * e
P = np.exp(mix - mix.max(1, keepdims=True)); P /= P.sum(1, keepdims=True)
P = sinkhorn(P)
ids = pd.read_parquet("work/G_out/submission.parquet")["id"].astype(str)
sub = make_sub(ids, P)
sub.to_parquet("work/blend_GE.parquet", index=False)
print("blend GE written:", sub["model"].value_counts().to_dict())
