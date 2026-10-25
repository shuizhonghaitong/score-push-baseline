import os, sys, time
import numpy as np, pandas as pd, re
sys.path.insert(0, os.path.dirname(__file__))
from common import *
from scipy.sparse import load_npz, hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
T0=time.time()
train, pool, prv = load_data()
val = prv[(prv['split']=='new') & (prv['passes_funnel']==True)].copy()
val['y'] = val['model'].map(LAB2I); val = val[val['y'].notna()].reset_index(drop=True)
y_val = val['y'].to_numpy()
Xv_base = load_npz('work/b2_Xv.npz')
fit = pd.concat([train[['text','model']], pool[~pool['src'].isin(('pvx','pv1'))][['text','model']],
                 prv[(prv['split']!='new') & (prv['passes_funnel']==True)][['text','model']]], ignore_index=True)
fit = fit[fit['model'].isin(LABELS)].reset_index(drop=True)
fit_y = fit['model'].map(LAB2I).to_numpy()
fit_t = fit['text'].astype(str).tolist(); val_t = val['text'].astype(str).tolist()

def mkview(tag):
    v = TfidfVectorizer(**tag); v.fit(fit_t)
    return v

views = {
 "char_2_6": dict(analyzer="char", ngram_range=(2,6), lowercase=False, sublinear_tf=True, min_df=2, max_features=250000, dtype=np.float32),
 "char_3_7": dict(analyzer="char", ngram_range=(3,7), lowercase=False, sublinear_tf=True, min_df=2, max_features=250000, dtype=np.float32),
 "word_1_4": dict(analyzer="word", ngram_range=(1,4), lowercase=False, sublinear_tf=True, min_df=2, max_features=180000, dtype=np.float32, token_pattern=r"(?u)\b\w+\b|[^\w\s]"),
}
res = {}
for name, tag in views.items():
    v = TfidfVectorizer(**tag); v.fit(fit_t)
    Xf2 = v.transform(fit_t); Xv2 = v.transform(val_t)
    est = LinearSVC(C=0.5, random_state=42, max_iter=10000, class_weight="balanced").fit(Xf2, fit_y)
    P2 = softmax(est.decision_function(Xv2))
    res[name] = P2
    print(f"{name}: solo BA {ba(y_val,P2):.4f} / sink {ba(y_val,sinkhorn(P2)):.4f} ({time.time()-T0:.0f}s)", flush=True)

# base prob (C=0.25 cached)
Pbase = np.load('work/svc_proxy.npz')['P']
print(f"base: {ba(y_val,Pbase):.4f} / {ba(y_val,sinkhorn(Pbase)):.4f}")
# blends: base + w*view
import itertools
for name, P2 in res.items():
    for w in (0.5, 1.0, 2.0):
        L = np.log(np.clip(Pbase,1e-9,1)) + w*np.log(np.clip(P2,1e-9,1))
        P = softmax(L)
        print(f"base+{w}*{name}: {ba(y_val,P):.4f} / sink {ba(y_val,sinkhorn(P)):.4f}", flush=True)
# all three
L = np.log(np.clip(Pbase,1e-9,1))
for name, P2 in res.items():
    L += 1.0*np.log(np.clip(P2,1e-9,1))
P = softmax(L)
print(f"base+all3: {ba(y_val,P):.4f} / sink {ba(y_val,sinkhorn(P)):.4f}", flush=True)
np.savez('work/b6_views.npz', **{f"P_{k}": v for k, v in res.items()})
