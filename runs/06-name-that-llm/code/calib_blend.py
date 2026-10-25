# calibrate (w_svc, w_llm, w_twin) on the proxy once featA proxy probs are downloaded
import os, sys, glob, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from common import *
import itertools
T0=time.time()
svc = np.load('work/svc_proxy.npz')
P_svc, y = svc['P'], svc['y']
cand = glob.glob('work/featA/proxy_probs.npz') + glob.glob('downloads/**/proxy_probs.npz', recursive=True)
assert cand, "download featA outputs first"
d = np.load(cand[0])
P_plain, P_twin = d['plain'], d['twin']
# proxy true groups: all 18 per prompt (dense) -> twin rows = all rows
def score(ws, wp, wt, rows):
    L = ws*np.log(np.clip(P_svc,1e-9,1)) + wp*np.log(np.clip(P_plain,1e-9,1))
    if wt:
        L[rows] += wt*np.log(np.clip(P_twin,1e-9,1))
    P = softmax(L)
    return ba(y, P), ba(y, sinkhorn(P))
allr = slice(None)
best = None
for ws, wp in itertools.product((1.0,), (0.5, 1.0, 1.5, 2.0, 3.0)):
    for wt in (0.0, 0.5, 1.0, 1.5, 2.0):
        b, s = score(ws, wp, wt, allr)
        print(f"ws{ws} wp{wp} wt{wt}: {b:.4f} / sink {s:.4f}", flush=True)
        if best is None or s > best[0]: best = (s, ws, wp, wt)
print("BEST", best, f"({time.time()-T0:.0f}s)")
