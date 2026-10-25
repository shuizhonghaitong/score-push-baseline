# Prefit the 4 SVC members locally -> small dataset (vocabs, idf, coefs) for the submission notebook.
# Faithful to notebookB/blend-v3 fit: train + pool(all src) + prv_clean + prv_kimi, model in LABELS.
# Sanity: clean-fit variant (pool w/o pvx,pv1; prv w/o split 'new') vs the ~0.4594 proxy target.
# Memory: one member at a time; view matrices freed after each member.
import faulthandler, gc, os, sys, time, pickle
faulthandler.enable()
import numpy as np, pandas as pd, re
sys.path.insert(0, os.path.dirname(__file__))
from common import *
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
T0=time.time()
def log(*a, **kw):
    print(*a, flush=True)

train, pool, prv = load_data()
prv["pf"] = prv["passes_funnel"].astype(str).str.lower().isin(["true","1"])
prv_clean = prv[prv["pf"] & prv["model"].isin(LABELS)]
prv_kimi = prv[(prv["split"]=="new_kimi_nothink") & (prv["model"].isin(LABELS))]
val = prv[(prv["split"]=="new") & (prv["pf"])].copy()
val["y"] = val["model"].map(LAB2I); val = val[val["y"].notna()].reset_index(drop=True)
y_val = val["y"].to_numpy(); val_t = val["text"].astype(str).tolist()

def make_fit(excl_pool=False, excl_new=False):
    pf = pool[~pool["src"].isin(("pvx","pv1"))] if excl_pool else pool
    pc = prv_clean[prv_clean["split"] != "new"] if excl_new else prv_clean
    fit = pd.concat([train[["text","model"]], pf[["text","model"]], pc[["text","model"]], prv_kimi[["text","model"]]], ignore_index=True)
    fit = fit[fit["model"].isin(LABELS)].reset_index(drop=True)
    return fit["text"].astype(str).tolist(), fit["model"].map(LAB2I).to_numpy()

FUNCTION_WORDS = set("a an the and or but if while although because as so than that which who whom whose this these those it its they them their theirs he him his she her hers we us our ours you your yours i me my mine in on at to from with without for of by over under between through into about is are was were be been being have has had do does did can could will would shall should may might must not no nor all any some each every both either neither other another such only also just very too more most less least here there when where why how what whether then now still yet even much many few little".split())
def shape_text(t): return re.sub(r"[^\W\d_]+", lambda m: "X" if m[0].isupper() else "x", t)
def function_text(t): return " ".join(w.lower() if w.lower() in FUNCTION_WORDS or not w.isalnum() else "CONTENT" for w in re.findall(r"\w+|[^\w\s]", t))

CHAR = dict(lowercase=False, sublinear_tf=True, min_df=2, dtype=np.float32)
WORD = dict(lowercase=False, sublinear_tf=True, min_df=2, dtype=np.float32, token_pattern=r"(?u)\b\w+\b|[^\w\s]")
# name: (prep_fn, vtag, weight)
VIEWS = {
 "c35":  (None,          dict(analyzer="char", ngram_range=(3,5), max_features=250000, **CHAR), 1.0),
 "w13":  (None,          dict(analyzer="word", ngram_range=(1,3), max_features=180000, **WORD), 0.7),
 "sh26": (shape_text,    dict(analyzer="char", ngram_range=(2,6), max_features=60000, **CHAR),  0.5),
 "fn13": (function_text, dict(analyzer="word", ngram_range=(1,3), max_features=50000, **WORD),  0.5),
 "c26":  (None,          dict(analyzer="char", ngram_range=(2,6), max_features=250000, **CHAR), 1.0),
 "c37":  (None,          dict(analyzer="char", ngram_range=(3,7), max_features=250000, **CHAR), 1.0),
 "w14":  (None,          dict(analyzer="word", ngram_range=(1,4), max_features=180000, **WORD), 1.0),
}
MEMBERS = [("c35","w13","sh26","fn13"), ("c26",), ("c37",), ("w14",)]

fit_t, fit_y = make_fit(excl_pool=False, excl_new=False)
log("fit rows (nb)", len(fit_t), f"({time.time()-T0:.0f}s)")
fit_t_clean, fit_y_clean = make_fit(excl_pool=True, excl_new=True)
log("fit rows (clean)", len(fit_t_clean), f"({time.time()-T0:.0f}s)")

_out = f"{ROOT}/work/svc4"
os.makedirs(f"{_out}/views", exist_ok=True)
os.makedirs(f"{_out}/members", exist_ok=True)
meta = {"labels": LABELS, "members": [list(m) for m in MEMBERS], "views": {}}

def fit_member(member, texts_fit, texts_val, y_fit, tag, save_artifacts=False):
    Xf_parts, Xv_parts, vecs = [], [], {}
    for name in member:
        prep, vtag, w = VIEWS[name]
        tf = [prep(t) for t in texts_fit] if prep else texts_fit
        tv = [prep(t) for t in texts_val] if prep else texts_val
        v = TfidfVectorizer(**vtag)
        Xf = v.fit_transform(tf)
        Xv = v.transform(tv)
        log(f"  [{tag}] view {name}: {Xf.shape} nnz={Xf.nnz} ({time.time()-T0:.0f}s)", flush=True)
        if save_artifacts:
            with open(f"{_out}/views/{name}_vocab.pkl","wb") as f: pickle.dump(v.vocabulary_, f)
            np.save(f"{_out}/views/{name}_idf.npy", v.idf_.astype(np.float64))
            meta["views"][name] = {"w": w, "n_features": len(v.vocabulary_)}
            del v
        Xf_parts.append(Xf * w); Xv_parts.append(Xv * w)
        del Xf, Xv
        gc.collect()
    Xf = hstack(Xf_parts, format="csr"); Xv = hstack(Xv_parts, format="csr")
    est = LinearSVC(C=0.25, random_state=42, max_iter=10000, class_weight="balanced").fit(Xf, y_fit)
    P = softmax(est.decision_function(Xv))
    ba_solo = ba(y_val, P); ba_sink = ba(y_val, sinkhorn(P))
    log(f"[{tag}] member {list(member)}: BA {ba_solo:.4f} / sink {ba_sink:.4f} ({time.time()-T0:.0f}s)", flush=True)
    if save_artifacts:
        np.save(f"{_out}/members/m{MEMBERS.index(list(member))}_coef.npy", est.coef_.astype(np.float32))
        np.save(f"{_out}/members/m{MEMBERS.index(list(member))}_intercept.npy", est.intercept_.astype(np.float32))
    del Xf, Xv, est
    gc.collect()
    return P

# ---- phase 1: clean-fit sanity (proxy BAs + blend) ----
log("=== clean-fit sanity ===", flush=True)
P_clean = []
for mi, m in enumerate(MEMBERS):
    P_clean.append(fit_member(m, fit_t_clean, val_t, fit_y_clean, "clean"))
Lc = sum(np.log(np.clip(p, 1e-9, 1.0)) for p in P_clean)
Pb = softmax(Lc)
log(f"[clean] 4-member blend: BA {ba(y_val,Pb):.4f} / sink {ba(y_val,sinkhorn(Pb)):.4f} (target ~0.435/0.459) ({time.time()-T0:.0f}s)", flush=True)
del P_clean, Pb, Lc, fit_t_clean, fit_y_clean
gc.collect()

# ---- phase 2: notebook fit (real artifacts) ----
log("=== notebook-fit artifacts ===", flush=True)
for mi, m in enumerate(MEMBERS):
    fit_member(m, fit_t, val_t, fit_y, "nb", save_artifacts=True)
with open(f"{_out}/meta.pkl","wb") as f: pickle.dump(meta, f)
log("DONE", f"({time.time()-T0:.0f}s)", flush=True)
