import sys, time; sys.path.insert(0, os.path.dirname(__file__)) if False else None
import os
import numpy as np, pandas as pd, re
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from common import *
ROOT = "/mnt/disk3/liuyutong_home/workspace/skill/score-push/runs/06-name-that-llm"
T0=time.time()
train, pool, prv = load_data()
y_train = train['model'].map(LAB2I).values
# validation: prvsiyan 'new' split, clean rows
val = prv[(prv['split']=='new') & (prv['passes_funnel']==True)].copy()
val['y'] = val['model'].map(LAB2I)
val = val[val['y'].notna()]
print("val rows:", len(val), "prompts:", val['prompt_id'].nunique(), flush=True)
# fit pool: train + pool_v4 + prvsiyan non-new clean rows
prv_fit = prv[(prv['split']!='new') & (prv['passes_funnel']==True)]
fit = pd.concat([train[['text','model']], pool[['text','model']], prv_fit[['text','model']]], ignore_index=True)
fit_y = fit['model'].map(LAB2I).values
fit = fit[fit['model'].isin(LABELS)]
fit_y = fit['model'].map(LAB2I).values
print("fit rows:", len(fit), flush=True)

FUNCTION_WORDS = set("a an the and or but if while although because as so than that which who whom whose this these those it its they them their theirs he him his she her hers we us our ours you your yours i me my mine in on at to from with without for of by over under between through into about is are was were be been being have has had do does did can could will would shall should may might must not no nor all any some each every both either neither other another such only also just very too more most less least here there when where why how what whether then now still yet even much many few little".split())
VIEWS = [("char",3,5,250000,1.0), ("word",1,3,180000,0.7), ("shape",2,6,60000,0.5), ("function",1,3,50000,0.5)]
def shape_text(t): return re.sub(r"[^\W\d_]+", lambda m: "X" if m[0].isupper() else "x", t)
def function_text(t):
    return " ".join(w.lower() if w.lower() in FUNCTION_WORDS or not w.isalnum() else "CONTENT" for w in re.findall(r"\w+|[^\w\s]", t))
def fit_feats(texts):
    A=[]
    for view,lo,hi,mf,w in VIEWS:
        prep = shape_text if view=="shape" else function_text if view=="function" else (lambda s:s)
        analyzer = "char" if view=="shape" else "word"
        opts = dict(analyzer=analyzer, ngram_range=(lo,hi), lowercase=False, sublinear_tf=True, min_df=2, max_features=mf, dtype=np.float32)
        if view=="word": opts["token_pattern"]=r"(?u)\b\w+\b|[^\w\s]"
        v=TfidfVectorizer(**opts)
        A.append(v.fit_transform([prep(t) for t in texts])*w)
    return hstack(A, format="csr")
texts_fit = fit['text'].astype(str).tolist()
texts_val = val['text'].astype(str).tolist()
Xf = fit_feats(texts_fit)
print("feats fitted", Xf.shape, f"({time.time()-T0:.0f}s)", flush=True)
Xv = None
import pickle
# need transform of val: refit is fine since val excluded from fit
def transform_feats(vecs, texts):
    B=[]
    for v,(view,lo,hi,mf,w) in zip(vecs,VIEWS):
        prep = shape_text if view=="shape" else function_text if view=="function" else (lambda s:s)
        B.append(v.transform([prep(t) for t in texts])*w)
    return hstack(B, format="csr")
vecs=[]
A=[]
for view,lo,hi,mf,w in VIEWS:
    prep = shape_text if view=="shape" else function_text if view=="function" else (lambda s:s)
    analyzer = "char" if view=="shape" else "word"
    opts = dict(analyzer=analyzer, ngram_range=(lo,hi), lowercase=False, sublinear_tf=True, min_df=2, max_features=mf, dtype=np.float32)
    if view=="word": opts["token_pattern"]=r"(?u)\b\w+\b|[^\w\s]"
    v=TfidfVectorizer(**opts); v.fit([prep(t) for t in texts_fit]); vecs.append(v)
Xv = transform_feats(vecs, texts_val)
y_val = val['y'].values

for name, est in [
    ("svc_C0.5", LinearSVC(C=0.5, random_state=42, max_iter=5000, class_weight="balanced")),
    ("svc_C1", LinearSVC(C=1.0, random_state=42, max_iter=5000, class_weight="balanced")),
    ("logreg_C4", LogisticRegression(C=4.0, max_iter=2000, class_weight="balanced", solver="liblinear")),
]:
    est.fit(Xf, fit_y)
    if hasattr(est, "decision_function"):
        P = softmax(est.decision_function(Xv))
    else:
        P = est.predict_proba(Xv)
    p_sink = sinkhorn(P)
    print(f"{name}: BA {ba(y_val,P):.4f} | +sinkhorn {ba(y_val,p_sink):.4f}  ({time.time()-T0:.0f}s)", flush=True)
    np.save(f"work/b1_{name}.npy", P, allow_pickle=False)
