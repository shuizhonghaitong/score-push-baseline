import os, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from common import *
from scipy.sparse import load_npz
from sklearn.svm import LinearSVC
T0=time.time()
train, pool, prv = load_data()
val = prv[(prv['split']=='new') & (prv['passes_funnel']==True)].copy()
val['y'] = val['model'].map(LAB2I); val = val[val['y'].notna()].reset_index(drop=True)
y_val = val['y'].to_numpy()
Xf = load_npz('work/b2_Xf_fit.npz'); Xv = load_npz('work/b2_Xv.npz')
fit = pd.concat([train[['text','model']], pool[~pool['src'].isin(('pvx','pv1'))][['text','model']],
                 prv[(prv['split']!='new') & (prv['passes_funnel']==True)][['text','model']]], ignore_index=True)
fit = fit[fit['model'].isin(LABELS)].reset_index(drop=True)
fit_y = fit['model'].map(LAB2I).to_numpy()
est = LinearSVC(C=0.25, random_state=42, max_iter=10000, class_weight="balanced").fit(Xf, fit_y)
P = softmax(est.decision_function(Xv))
print("C0.25 proxy BA", ba(y_val, P), "| sink", ba(y_val, sinkhorn(P)))
np.savez("work/svc_proxy.npz", P=P, y=y_val)
# also save trained svc + vectorizers? not needed
print(f"({time.time()-T0:.0f}s)")
