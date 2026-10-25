# Tune W_SVC/W_QWEN via prompt holdout using local Qwen features (proxy rows only).
# Note: local head trained on 240-prompt subset -> weaker than notebook head (78k rows);
# treat as directional lower bound for Qwen contribution.
import sys, os, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from common import *
T0=time.time()
prv = load_data()[2]
prv["pf"] = prv["passes_funnel"].astype(str).str.lower().isin(["true","1"])
proxy = prv[prv["pf"] & prv["model"].isin(LABELS) & (prv["split"]=="new")].reset_index(drop=True)
proxy["y"] = proxy["model"].map(LAB2I)
Z = np.load("work/proxy_z_local.npz")["Z"]; y_z = np.load("work/proxy_z_local.npz")["y"]
assert len(Z)==len(proxy) and (proxy["y"].to_numpy()==y_z).all(), "order mismatch"
P_svc = np.load("work/svc_proxy.npz")["P"]
y = proxy["y"].to_numpy()
prompts = proxy["prompt_id"].to_numpy()
uniq = np.unique(prompts)
rng = np.random.RandomState(42)
rng.shuffle(uniq)
val_prompts = set(uniq[:60].tolist())
val = np.array([p in val_prompts for p in prompts])
print("val rows", val.sum(), "train rows", (~val).sum())
from sklearn.linear_model import LogisticRegression
head = LogisticRegression(C=0.05, max_iter=3000, class_weight="balanced", solver="lbfgs").fit(Z[~val], y[~val])
P_q = head.predict_proba(Z[val])
print("qwen-holdout solo BA:", ba(y[val], P_q), f"({time.time()-T0:.0f}s)")
Ls = np.log(np.clip(P_svc[val],1e-9,1))
Lq = np.log(np.clip(P_q,1e-9,1))
print(f"{'wq':>4} {'BA':>8} {'BA+sink':>9}")
best=None
for wq in (0.0,0.25,0.5,0.75,1.0,1.5,2.0,3.0):
    P = softmax(Ls + wq*Lq)
    s1 = ba(y[val], P); s2 = ba(y[val], sinkhorn(P))
    print(f"{wq:>4} {s1:8.4f} {s2:9.4f}")
    if best is None or s2>best[0]: best=(s2,wq)
print("best (sink)", best, f"({time.time()-T0:.0f}s)")
# per-class: where does qwen help?
P = softmax(Ls + 1.0*Lq)
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y[val], P.argmax(1), labels=list(range(K)))
rec = cm.diagonal()/cm.sum(1)
for i,r in enumerate(rec): print(f"  {LABELS[i]:>22} recall {r:.3f}")
