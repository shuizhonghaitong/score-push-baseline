import os
import numpy as np
import pandas as pd
from blend_common import K, CANON, to_canon, sinkhorn, make_sub

f = to_canon(np.load("work/F_out/fork0612_probs.npz")["logit"], "work/F_out/probs_order.npy", "F")
e = to_canon(np.load("work/E2_out/fork_probs.npz")["logit"], "work/E2_out/probs_order.npy", "E2")
assert f.shape == (2699, K) and e.shape == (2699, K), (f.shape, e.shape)
mix = 0.70 * f + 0.30 * e
P = np.exp(mix - mix.max(1, keepdims=True)); P /= P.sum(1, keepdims=True)
P = sinkhorn(P)
ids = pd.read_parquet("work/F_out/submission.parquet")["id"].astype(str)
sub = make_sub(ids, P)
sub.to_parquet("work/blend_FE.parquet", index=False)
print("blend FE written:", sub["model"].value_counts().to_dict())
