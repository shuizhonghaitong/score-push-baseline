# Oct-24 extra blend: G + H (BERT-trio), canonical order via order files.
import numpy as np, pandas as pd, sys
sys.path.insert(0, "work")
from blend_common import K, CANON, to_canon, sinkhorn, make_sub
g = to_canon(np.load("work/G_out/final_lp.npy"), "work/G_out/final_lp_order.npy", "G")
h = to_canon(np.load("work/H_out/forkH_probs.npz")["logit"], "work/H_out/probs_order.npy", "H")
assert g.shape == (2699, K) and h.shape == (2699, K), (g.shape, h.shape)
mix = 0.70 * g + 0.30 * h
P = np.exp(mix - mix.max(1, keepdims=True)); P /= P.sum(1, keepdims=True)
P = sinkhorn(P)
ids = pd.read_parquet("work/G_out/submission.parquet")["id"].astype(str)
sub = make_sub(ids, P)
sub.to_parquet("work/blend_GH.parquet", index=False)
print("blend GH written:", sub["model"].value_counts().to_dict())
