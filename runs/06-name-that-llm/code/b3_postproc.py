import os, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from common import *
from scipy.optimize import linear_sum_assignment
T0=time.time()
train, pool, prv = load_data()
val = prv[(prv['split']=='new') & (prv['passes_funnel']==True)].copy()
val['y'] = val['model'].map(LAB2I)
val = val[val['y'].notna()].reset_index(drop=True)
y_val = val['y'].values
texts = val['text'].astype(str).tolist()
# regenerate probs: fit SVC on fit corpus (reuse cached feats? need same transform on val with fit vectorizers)
from scipy.sparse import load_npz
Xf=load_npz('work/b2_Xf_fit.npz')
Xv=load_npz('work/b2_Xv.npz')
fit = pd.concat([train[['text','model']], pool[~pool['src'].isin(('pvx','pv1'))][['text','model']],
                 prv[(prv['split']!='new') & (prv['passes_funnel']==True)][['text','model']]], ignore_index=True)
fit = fit[fit['model'].isin(LABELS)]
fit_y = fit['model'].map(LAB2I).values
from sklearn.svm import LinearSVC
est = LinearSVC(C=0.5, random_state=42, max_iter=10000, class_weight="balanced").fit(Xf, fit_y)
P0 = softmax(est.decision_function(Xv))
print("base BA", ba(y_val, P0), f"({time.time()-T0:.0f}s)", flush=True)

# em-dash guard
SPACED_ALLOWED = {'claude-opus-4-6','claude-opus-5','glm-5.3','glm-5.3-flash'}
forb = np.array([l not in SPACED_ALLOWED for l in LABELS])
P1 = P0.copy(); n=0
for i,t in enumerate(texts):
    if " \u2014 " in t and LABELS[P1[i].argmax()] not in SPACED_ALLOWED:
        P1[i, forb] = 0.0; P1[i] /= P1[i].sum(); n+=1
print(f"emdash fixed {n}: BA {ba(y_val,P1)}", flush=True)
Ps0, Ps1 = sinkhorn(P0), sinkhorn(P1)
print(f"sink: {ba(y_val,Ps0)} {ba(y_val,Ps1)}", flush=True)

# Hungarian on true prompt groups (dense: all 18 per prompt)
P2 = sinkhorn(P1)
from collections import defaultdict
grp = defaultdict(list)
for i, g in enumerate(val['prompt_id']): grp[g].append(i)
P3 = P2.copy()
changes=0
for g, idx in grp.items():
    if 12 <= len(idx) <= 18:
        cost = -np.log(np.clip(P3[idx,:], 1e-12, 1.0))
        r, c = linear_sum_assignment(cost)
        for rr, cc in zip(r, c):
            if P3[idx[rr]].argmax() != cc: changes+=1
        for rr, cc in zip(r, c):
            P3[idx[rr],:] = 0.0; P3[idx[rr],cc] = 1.0
print(f"hungarian dense changes {changes}: BA {ba(y_val,P3)}", flush=True)

# centering of log-probs within true groups
P4 = sinkhorn(P1).copy()
L4 = np.log(np.clip(P4,1e-12,1.0))
for g, idx in grp.items():
    L4[idx] -= L4[idx].mean(0, keepdims=True)
print(f"logprob centering: BA {ba(y_val, softmax(L4))}", flush=True)

# per-class recall of base
pred = P0.argmax(1); K=18
rec = np.array([(pred[y_val==k]==k).mean() for k in range(K)])
order = np.argsort(rec)
for k in order: print(f"  {LABELS[k]:24s} {rec[k]:.3f}")
np.save('work/b3_P0.npy', P0)
