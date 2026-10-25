import os, sys, time, pickle
import numpy as np, pandas as pd, re
from scipy.sparse import hstack, save_npz, load_npz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
sys.path.insert(0, os.path.dirname(__file__))
from common import *
T0=time.time()
train, pool, prv = load_data()
y_train = train['model'].map(LAB2I).values
val = prv[(prv['split']=='new') & (prv['passes_funnel']==True)].copy()
val['y'] = val['model'].map(LAB2I)
val = val[val['y'].notna()]
EXCL = ('pvx','pv1')
pool_fit = pool[~pool['src'].isin(EXCL)]
prv_fit = prv[(prv['split']!='new') & (prv['passes_funnel']==True)]
fit = pd.concat([train[['text','model']], pool_fit[['text','model']], prv_fit[['text','model']]], ignore_index=True)
fit = fit[fit['model'].isin(LABELS)].reset_index(drop=True)
fit_y = fit['model'].map(LAB2I).values
train_only = pd.concat([train[['text','model']]], ignore_index=True)
print("val:", len(val), "fit:", len(fit), "train_only:", len(train_only), flush=True)

FUNCTION_WORDS = set("a an the and or but if while although because as so than that which who whom whose this these those it its they them their theirs he him his she her hers we us our ours you your yours i me my mine in on at to from with without for of by over under between through into about is are was were be been being have has had do does did can could will would shall should may might must not no nor all any some each every both either neither other another such only also just very too more most less least here there when where why how what whether then now still yet even much many few little".split())
VIEWS = [("char",3,5,250000,1.0), ("word",1,3,180000,0.7), ("shape",2,6,60000,0.5), ("function",1,3,50000,0.5)]
def shape_text(t): return re.sub(r"[^\W\d_]+", lambda m: "X" if m[0].isupper() else "x", t)
def function_text(t):
    return " ".join(w.lower() if w.lower() in FUNCTION_WORDS or not w.isalnum() else "CONTENT" for w in re.findall(r"\w+|[^\w\s]", t))
def make_vecs():
    vs=[]
    for view,lo,hi,mf,w in VIEWS:
        opts = dict(analyzer="char" if view=="shape" else "word", ngram_range=(lo,hi), lowercase=False,
                    sublinear_tf=True, min_df=2, max_features=mf, dtype=np.float32)
        if view=="word": opts["token_pattern"]=r"(?u)\b\w+\b|[^\w\s]"
        vs.append((view, TfidfVectorizer(**opts), w))
    return vs
def prep(view,t):
    return shape_text(t) if view=="shape" else function_text(t) if view=="function" else t
def fitX(vs, texts):
    A=[v.fit_transform([prep(view,t) for t in texts])*w for view,v,w in vs]
    return hstack(A, format="csr")
def trX(vs, texts):
    A=[v.transform([prep(view,t) for t in texts])*w for view,v,w in vs]
    return hstack(A, format="csr")

def load_all():
    import pickle
    d={}
    for k in ("Xf_fit","Xf_train","Xv"):
        d[k]=load_npz(f"work/b2_{k}.npz")[f"X"]
    return d
if all(os.path.exists(f"work/b2_{k}.npz") for k in ("Xf_fit","Xf_train","Xv")):
    Xf_fit, Xf_train, Xv = load_all()["Xf_fit"], load_all()["Xf_train"], load_all()["Xv"]
    print("loaded cache", f"({time.time()-T0:.0f}s)", flush=True)
else:
    vs = make_vecs()
    Xf_fit = fitX(vs, fit['text'].astype(str).tolist())
    save_npz("work/b2_Xf_fit.npz", Xf_fit)
    print("fitX fit done", Xf_fit.shape, f"({time.time()-T0:.0f}s)", flush=True)
    vs2 = make_vecs(); Xf_train = fitX(vs2, train_only['text'].astype(str).tolist())
    save_npz("work/b2_Xf_train.npz", Xf_train)
    print("fitX train done", f"({time.time()-T0:.0f}s)", flush=True)
    Xv = trX(vs, val["text"].astype(str).tolist())
    save_npz("work/b2_Xv.npz", Xv)
    print("Xv done, cached", f"({time.time()-T0:.0f}s)", flush=True)

y_val = val['y'].values
results=[]
def eval_name(name, P):
    b = ba(y_val, P); s = ba(y_val, sinkhorn(P))
    print(f"{name}: BA {b:.4f} | +sinkhorn {s:.4f} ({time.time()-T0:.0f}s)", flush=True)
    results.append((name,b,s))
    return P

for C in (0.25, 0.5, 1.0):
    est = LinearSVC(C=C, random_state=42, max_iter=10000, class_weight="balanced")
    est.fit(Xf_fit, fit_y)
    P = eval_name(f"svc{C}_pool", softmax(est.decision_function(Xv)))
for C in (1.0,):
    est = LinearSVC(C=C, random_state=42, max_iter=10000, class_weight="balanced")
    est.fit(Xf_train, y_train)
    P = eval_name(f"svc{C}_trainonly", softmax(est.decision_function(Xv)))
import pickle as pk
pk.dump(results, open('work/b2_results.pkl','wb'))
