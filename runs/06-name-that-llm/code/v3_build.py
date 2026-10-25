# v3 final artifacts: refit selected members on fit_full, save prefit bundle work/svc5/.
import gc, json, os, pickle, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import LABELS, LAB2I, ROOT, softmax
from scipy.sparse import hstack, csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
T0=time.time()
def log(*a): print(f"[{time.time()-T0:7.0f}s]", *a, flush=True)

cfg = json.load(open(f"{ROOT}/work/v3_config.json"))
SEL = cfg["members"]; EXP = cfg["exponents"]; POST = cfg["post"]
train, pool, prv = pd.read_csv(f"{ROOT}/data/train.csv"), pd.read_parquet(f"{ROOT}/pool/v4/ntl_pool_v4.parquet"), pd.read_csv(f"{ROOT}/pool/passages.csv")
prv["pf"] = prv["passes_funnel"].astype(str).str.lower().isin(["true","1"])
prv_clean = prv[prv["pf"] & prv["model"].isin(LABELS)]
prv_kimi = prv[(prv["split"] == "new_kimi_nothink") & prv["model"].isin(LABELS)]
fit_full = pd.concat([train[["text","model"]], pool[["text","model"]], prv_clean[["text","model"]], prv_kimi[["text","model"]]], ignore_index=True)
fit_full = fit_full[fit_full["model"].isin(LABELS)].reset_index(drop=True)
fit_t = fit_full["text"].astype(str).tolist(); fit_y = fit_full["model"].map(LAB2I).to_numpy()
log("fit_full rows", len(fit_t))

FUNCTION_WORDS = set("a an the and or but if while although because as so than that which who whom whose this these those it its they them their theirs he him his she her hers we us our ours you your yours i me my mine in on at to from with without for of by over under between through into about is are was were be been being have has had do does did can could will would shall should may might must not no nor all any some each every both either neither other another such only also just very too more most less least here there when where why how what whether then now still yet even much many few little".split())
def shape_text(t): return re.sub(r"[^\W\d_]+", lambda m: "X" if m[0].isupper() else "x", t)
def function_text(t): return " ".join(w.lower() if w.lower() in FUNCTION_WORDS or not w.isalnum() else "CONTENT" for w in re.findall(r"\w+|[^\w\s]", t))
import re
PREP = {"raw": None, "shape": shape_text, "function": function_text}
CHAR = dict(lowercase=False, sublinear_tf=True, min_df=2, dtype=np.float32)
WORD = dict(lowercase=False, sublinear_tf=True, min_df=2, dtype=np.float32, token_pattern=r"(?u)\b\w+\b|[^\w\s]")
VIEWS = {
 "c24":  ("raw", dict(analyzer="char", ngram_range=(2,4),  max_features=250000, **CHAR)),
 "c26":  ("raw", dict(analyzer="char", ngram_range=(2,6),  max_features=250000, **CHAR)),
 "c35":  ("raw", dict(analyzer="char", ngram_range=(3,5),  max_features=250000, **CHAR)),
 "c37":  ("raw", dict(analyzer="char", ngram_range=(3,7),  max_features=250000, **CHAR)),
 "c48":  ("raw", dict(analyzer="char", ngram_range=(4,8),  max_features=250000, **CHAR)),
 "c26_400": ("raw", dict(analyzer="char", ngram_range=(2,6), max_features=400000, **CHAR)),
 "c35_100": ("raw", dict(analyzer="char", ngram_range=(3,5), max_features=100000, **CHAR)),
 "sh26": ("shape", dict(analyzer="char", ngram_range=(2,6), max_features=60000, **CHAR)),
 "sh37": ("shape", dict(analyzer="char", ngram_range=(3,7), max_features=60000, **CHAR)),
 "fn13": ("function", dict(analyzer="word", ngram_range=(1,3), max_features=50000, **WORD)),
 "w12":  ("raw", dict(analyzer="word", ngram_range=(1,2),  max_features=180000, **WORD)),
 "w13":  ("raw", dict(analyzer="word", ngram_range=(1,3),  max_features=180000, **WORD)),
 "w14":  ("raw", dict(analyzer="word", ngram_range=(1,4),  max_features=180000, **WORD)),
 "w15":  ("raw", dict(analyzer="word", ngram_range=(1,5),  max_features=180000, **WORD)),
 "w25":  ("raw", dict(analyzer="word", ngram_range=(2,5),  max_features=180000, **WORD)),
 "w13_400": ("raw", dict(analyzer="word", ngram_range=(1,3), max_features=400000, **WORD)),
}
COMBOS = {"base4": [("c35",1.0),("w13",0.7),("sh26",0.5),("fn13",0.5)]}

OUT = f"{ROOT}/work/svc5"
os.makedirs(f"{OUT}/views", exist_ok=True); os.makedirs(f"{OUT}/members", exist_ok=True)
meta = {"labels": LABELS, "post": POST, "exponents": dict(zip(SEL, EXP)), "members": [], "views": {}}
view_cache = {}
def get_view(name):
    if name in view_cache: return view_cache[name]
    prep, vtag = VIEWS[name]
    f = PREP.get(prep)
    v = TfidfVectorizer(**vtag)
    v.fit([f(t) for t in fit_t] if f else fit_t)
    with open(f"{OUT}/views/{name}_vocab.pkl","wb") as fh: pickle.dump(v.vocabulary_, fh)
    np.save(f"{OUT}/views/{name}_idf.npy", v.idf_.astype(np.float64))
    meta["views"][name] = {"vtag": vtag, "prep": prep, "n_features": len(v.vocabulary_)}
    view_cache[name] = v
    log(f"view {name} fit_full: {len(v.vocabulary_)} feats ({time.time()-T0:.0f}s)")
    return v

mi = 0
for member in SEL:
    if member in COMBOS:
        parts = []
        for vn, w in COMBOS[member]:
            v = get_view(vn)
            f = PREP.get(VIEWS[vn][0])
            X = v.transform([f(t) for t in fit_t] if f else fit_t) * w
            parts.append(X)
        Xf = hstack(parts, format="csr")
        meta["members"].append({"name": member, "combo": [[vn, w] for vn, w in COMBOS[member]]})
    else:
        v = get_view(member)
        f = PREP.get(VIEWS[member][0])
        Xf = v.transform([f(t) for t in fit_t] if f else fit_t)
        meta["members"].append({"name": member, "combo": [[member, 1.0]]})
    est = LinearSVC(C=0.25, random_state=42, max_iter=20000, class_weight="balanced").fit(Xf, fit_y)
    np.save(f"{OUT}/members/m{mi}_coef.npy", est.coef_.astype(np.float32))
    np.save(f"{OUT}/members/m{mi}_intercept.npy", est.intercept_.astype(np.float32))
    log(f"member {member} fit_full done ({time.time()-T0:.0f}s)")
    del Xf, est; gc.collect()
    mi += 1
with open(f"{OUT}/meta.pkl","wb") as fh: pickle.dump(meta, fh)
import shutil; shutil.copyfile(f"{OUT}/meta.pkl", f"{OUT}/svc5_meta.pkl")
log("BUILD DONE", mi, "members")
