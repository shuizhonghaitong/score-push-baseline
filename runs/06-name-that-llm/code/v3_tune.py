# v3 tuning: consume work/v3_cache/sel_Pval.npz; exponent tuning + post-processing variants.
import os, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import LABELS, LAB2I, ROOT, softmax, sinkhorn, ba
from scipy.optimize import linear_sum_assignment
T0=time.time()
def log(*a): print(f"[{time.time()-T0:6.0f}s]", *a, flush=True)
K=18
d = np.load(f"{ROOT}/work/v3_cache/sel_Pval.npz", allow_pickle=True)
y = d["y"]; sel = str(d["sel"]).split(",")
P = {n: d[n] for n in d.files if n != "y" and n != "sel"}
prv = pd.read_csv(f"{ROOT}/pool/passages.csv")
prv["pf"] = prv["passes_funnel"].astype(str).str.lower().isin(["true","1"])
val = prv[(prv["split"]=="new") & prv["pf"] & prv["model"].isin(LABELS)].reset_index(drop=True)
val["y"] = val["model"].map(LAB2I)
G = pd.factorize(val["prompt_id"])[0].astype(np.int32)
assert len(y) == len(val)
SPACED_ALLOWED = {'claude-opus-4-6','claude-opus-5','glm-5.3','glm-5.3-flash'}
forb = np.array([l not in SPACED_ALLOWED for l in LABELS])
texts = val["text"].astype(str).tolist()
def emdash(Pm):
    Q = Pm.copy(); n=0
    for i,t in enumerate(texts):
        if " \u2014 " in t and LABELS[Q[i].argmax()] not in SPACED_ALLOWED:
            Q[i, forb] = 0.0; Q[i] /= Q[i].sum(); n+=1
    return Q, n
def hungarian_dense(Pm, groups=G, lo=12, hi=18):
    Q = Pm.copy()
    sz = np.bincount(groups)
    changes = 0
    for g in np.unique(groups):
        if not (lo <= sz[g] <= hi): continue
        ii = np.where(groups == g)[0]
        r, c = linear_sum_assignment(-np.log(np.clip(Q[ii],1e-12,1.0)))
        for rr, cc in zip(r, c):
            if Q[ii[rr]].argmax() != cc: changes += 1
            Q[ii[rr]] = 0.0; Q[ii[rr], cc] = 1.0
    return Q, changes

def blend(w):
    L = np.zeros((len(y), K))
    for n, wi in zip(sel, w):
        L += wi * np.log(np.clip(P[n], 1e-9, 1.0))
    return softmax(L)

w = [1.0]*len(sel)
log("start", sel, "1:1 BA", f"{ba(y, blend(w)):.4f}", "sink", f"{ba(y, sinkhorn(blend(w))):.4f}")
# coordinate-descent exponent tuning (objective: sinkhorn BA)
for rnd in range(3):
    for i, n in enumerate(sel):
        best = (w[i], -1)
        for cand in (0.4, 0.6, 0.8, 1.0, 1.25, 1.6, 2.0):
            w2 = w[:]; w2[i] = cand
            b = ba(y, sinkhorn(blend(w2)))
            if b > best[1]: best = (cand, b)
        if best[0] != w[i]:
            log(f"tune {n}: {w[i]} -> {best[0]}  sinkBA {best[1]:.4f}")
            w[i] = best[0]
log("final weights", dict(zip(sel, w)))
Pb = blend(w)
base_ba = ba(y, Pb)
Ps = sinkhorn(Pb)
log(f"blend: BA {base_ba:.4f} / sink {ba(y,Ps):.4f}")
# post variants
Pe, ne = emdash(Pb); Pse = sinkhorn(Pe)
log(f"emdash->sink: {ba(y,Pse):.4f} (fixed {ne})")
Pesh, n2 = emdash(Ps)
log(f"emdash after sink: {ba(y,Pesh):.4f} (fixed {n2})")
Ph, ch = hungarian_dense(Pse)
log(f"+hungarian on emdash->sink: {ba(y,Ph):.4f} (changed {ch})")
Ph2, ch2 = hungarian_dense(Ps)
log(f"+hungarian on sink only: {ba(y,Ph2):.4f} (changed {ch2})")
# v2 recipe (1:1:1:1 on base4/c26/c37/w14, emdash->sink) for reference
if all(n in P for n in ("base4","c26","c37","w14")):
    Lv = 0
    for n in ("base4","c26","c37","w14"): Lv += np.log(np.clip(P[n],1e-9,1.0))
    Pv = softmax(Lv)
    pev, _ = emdash(Pv)
    log(f"v2 recipe ref: BA {ba(y,Pv):.4f} / sink {ba(y,sinkhorn(Pv)):.4f} / emdash->sink {ba(y,sinkhorn(pev)):.4f}")
np.savez(f"{ROOT}/work/v3_cache/tuned.npz", w=np.array(w), sel=",".join(sel))
log("TUNE DONE")
