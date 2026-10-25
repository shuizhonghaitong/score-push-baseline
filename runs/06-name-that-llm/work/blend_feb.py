import os
import numpy as np
import pandas as pd
from blend_common import K, CANON, to_canon, sinkhorn, make_sub

f = to_canon(np.load("work/F_out/fork0612_probs.npz")["logit"], "work/F_out/probs_order.npy", "F")
e = to_canon(np.load("work/E2_out/fork_probs.npz")["logit"], "work/E2_out/probs_order.npy", "E2")
b = pd.read_parquet("work/B_out/submission.parquet")
ids = pd.read_parquet("work/F_out/submission.parquet")["id"].astype(str)
assert list(b["id"].astype(str)) == list(ids), "B ids misaligned"
Pb = np.full((len(ids), K), 1.0 / K)
for i, lab in enumerate(b["model"].astype(str)):
    Pb[i, CANON.index(lab)] = 1.0
lb = np.log(np.clip(Pb, 1e-6, 1))
mix = 0.60 * f + 0.25 * e + 0.15 * lb
P = np.exp(mix - mix.max(1, keepdims=True)); P /= P.sum(1, keepdims=True)
P = sinkhorn(P)
sub = make_sub(ids, P)
sub.to_parquet("work/blend_FEB.parquet", index=False)
print("blend FEB written:", sub["model"].value_counts().to_dict())
