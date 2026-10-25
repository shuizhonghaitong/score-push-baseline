import os, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from common import *
from scipy.sparse import load_npz
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
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
# rebuild membership
n_tr, n_po = len(train), len(pool[~pool['src'].isin(('pvx','pv1'))])
w_full = np.ones(len(fit)); w_pool03 = w_full.copy(); w_pool05 = w_full.copy()
w_pool03[n_tr:n_tr+n_po] = 0.3; w_pool05[n_tr:n_tr+n_po] = 0.5
def rep(name, P):
    print(f"{name}: BA {ba(y_val,P):.4f} | +sink {ba(y_val,sinkhorn(P)):.4f} ({time.time()-T0:.0f}s)", flush=True)
for C in (0.25, 0.5, 1.0, 2.0):
    est = LinearSVC(C=C, random_state=42, max_iter=10000, class_weight="balanced").fit(Xf, fit_y)
    rep(f"svc_bal_C{C}", softmax(est.decision_function(Xv)))
est = LinearSVC(C=0.5, random_state=42, max_iter=10000, class_weight=None).fit(Xf, fit_y)
rep("svc_none_C0.5", softmax(est.decision_function(Xv)))
for tag, w in (("pool0.3", w_pool03), ("pool0.5", w_pool05)):
    est = LinearSVC(C=0.5, random_state=42, max_iter=10000, class_weight="balanced").fit(Xf, fit_y, sample_weight=w)
    rep(f"svc_bal_C0.5_{tag}", softmax(est.decision_function(Xv)))
est = LogisticRegression(C=2.0, max_iter=3000, class_weight="balanced", solver="liblinear").fit(Xf, fit_y)
rep("logreg_bal_C2", est.predict_proba(Xv))
est = LogisticRegression(C=2.0, max_iter=3000, class_weight="balanced", solver="lbfgs").fit(Xf, fit_y)
rep("logreg_bal_C2_lbfgs", est.predict_proba(Xv))
