# Rebuild the 4 prefit SVC members' val-proxy probabilities from attached artifacts,
# then tune member blend exponents (+ later Qwen weight) on the clean val split.
import gc, os, pickle, re, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from common import *
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
T0 = time.time()
def log(*a): print(*a)

SVCD = "work/svc4"
meta = pickle.load(open(SVCD + "/meta.pkl", "rb"))
LABELS = meta["labels"]
LAB2I = {l: i for i, l in enumerate(LABELS)}
K = len(LABELS)

train, pool, prv = load_data()
prv["pf"] = prv["passes_funnel"].astype(str).str.lower().isin(["true", "1"])
_pvx = set(pool[pool["src"] == "pvx"]["text"])
val = prv[(prv["split"] == "new") & (prv["pf"]) & val_text_in_pvx].copy() if False else prv[(prv["split"] == "new") & (prv["pf"]) & (prv["text"].isin(_pvx))].copy()
log("NOTE: pvx rows are in fit data of prefit artifacts -> absolute BA optimistic; use for RELATIVE weight tuning only")
val["y"] = val["model"].map(LAB2I); val = val[val["y"].notna()].reset_index(drop=True)
kimi_val = prv[(prv["split"] == "new_kimi_nothink") & (prv["pf"]) & (prv["model"] == "kimi-k3")].copy()
kimi_val["y"] = kimi_val["model"].map(LAB2I)
y = val["y"].to_numpy(); texts = val["text"].astype(str).tolist()
log("val rows", len(val))

CHAR = dict(lowercase=False, sublinear_tf=True, min_df=2, dtype=np.float32)
WORD = dict(lowercase=False, sublinear_tf=True, min_df=2, dtype=np.float32, token_pattern=r"(?u)\b\w+\b|[^\w\s]")
VTAG = {
 "c35":  dict(analyzer="char", ngram_range=(3,5), max_features=250000, **CHAR),
 "w13":  dict(analyzer="word", ngram_range=(1,3), max_features=180000, **WORD),
 "sh26": dict(analyzer="char", ngram_range=(2,6), max_features=60000, **CHAR),
 "fn13": dict(analyzer="word", ngram_range=(1,3), max_features=50000, **WORD),
 "c26":  dict(analyzer="char", ngram_range=(2,6), max_features=250000, **CHAR),
 "c37":  dict(analyzer="char", ngram_range=(3,7), max_features=250000, **CHAR),
 "w14":  dict(analyzer="word", ngram_range=(1,4), max_features=180000, **WORD),
}
FUNCTION_WORDS = set("a an the and or but if while although because as so than that which who whom whose this these those it its they them their theirs he him his she her hers we us our ours you your yours i me my mine in on at to from with without for of by over under between through into about is are was were be been being have has had do does did can could will would shall should may might must not no nor all any some each every both either neither other another such only also just very too more most less least here there when where why how what whether then now still yet even much many few little".split())
def shape_text(t): return re.sub(r"[^\W\d_]+", lambda m: "X" if m[0].isupper() else "x", t)
def function_text(t): return " ".join(w.lower() if w.lower() in FUNCTION_WORDS or not w.isalnum() else "CONTENT" for w in re.findall(r"\w+|[^\w\s]", t))
PREP = {"char": None, "word": None, "raw": None, "shape": shape_text, "function": function_text}

views_cache = {}
def view_matrix(name, ts):
    if name not in views_cache:
        v = TfidfVectorizer(**VTAG[name])
        v.vocabulary_ = pickle.load(open(SVCD + f"/views/{name}_vocab.pkl", "rb"))
        v._tfidf = TfidfTransformer(norm="l2", use_idf=True, smooth_idf=True, sublinear_tf=True)
        v._tfidf.idf_ = np.load(SVCD + f"/views/{name}_idf.npy")
        views_cache[name] = v
    v = views_cache[name]
    p = PREP[meta["views"][name]["prep"]]
    t = [p(x) for x in ts] if p else ts
    return v.transform(t) * meta["views"][name]["w"]

PM = []
for mi, member in enumerate(meta["members"]):
    Xs = [view_matrix(n, texts) for n in member]
    X = hstack(Xs, format="csr")
    coef = np.load(SVCD + f"/members/m{mi}_coef.npy")
    inter = np.load(SVCD + f"/members/m{mi}_intercept.npy")
    d = X @ coef.T + inter
    P = softmax(d)
    PM.append(P)
    log(f"member {mi} {list(member)}: BA {ba(y, P):.4f} / sink {ba(y, sinkhorn(P)):.4f} ({time.time()-T0:.0f}s)")
    del X, Xs
    gc.collect()
np.savez_compressed("work/svc4_val.npz", y=y, **{f"P{mi}": P for mi, P in enumerate(PM)})

# ---- tune member exponents (base 1:1:1:1 = current) ----
Lbase = sum(np.log(np.clip(p, 1e-9, 1.0)) for p in PM)
log("blend 1:1:1:1:", f"BA {ba(y, softmax(Lbase)):.4f} / sink {ba(y, sinkhorn(softmax(Lbase))):.4f}")
import itertools
best = (-1.0, None)
grid = (0.5, 0.75, 1.0, 1.25, 1.5, 2.0)
for w0 in (0.5, 1.0, 1.5, 2.0):
    for w1 in (0.5, 1.0, 1.5):
        for w2 in (0.5, 1.0, 1.5):
            for w3 in (0.5, 1.0, 1.5):
                L = w0*np.log(np.clip(PM[0],1e-9,1)) + w1*np.log(np.clip(PM[1],1e-9,1)) + w2*np.log(np.clip(PM[2],1e-9,1)) + w3*np.log(np.clip(PM[3],1e-9,1))
                s = ba(y, sinkhorn(softmax(L)))
                if s > best[0]:
                    best = (s, (w0, w1, w2, w3))
log("best member exponents:", best, f"({time.time()-T0:.0f}s)")

# ---- regularized log-linear meta-learner (alternative to grid) ----
from sklearn.linear_model import LogisticRegression
M = np.stack([np.log(np.clip(p, 1e-9, 1.0)) for p in PM], axis=1)  # (N,4,18)
Xmeta = np.concatenate([M.reshape(len(y), -1)], axis=1)
clf = LogisticRegression(C=0.5, max_iter=2000, solver="lbfgs")
clf.fit(Xmeta, y)
Pm = np.zeros((len(y), K))
probs = clf.predict_proba(Xmeta)
for j, cl in enumerate(clf.classes_):
    Pm[:, int(cl)] = probs[:, j]
log("meta-learner C0.5:", f"BA {ba(y, Pm):.4f} / sink {ba(y, sinkhorn(Pm)):.4f}")
Pm2 = softmax(clf.decision_function(Xmeta))
log("meta-learner softmax:", f"BA {ba(y, Pm2):.4f} / sink {ba(y, sinkhorn(Pm2)):.4f}")

# ---- kimi rows (in fit data - sanity only) ----
if len(kimi_val):
    ky = kimi_val["y"].to_numpy(); kts = kimi_val["text"].astype(str).tolist()
    KL = 0
    for mi in range(4):
        Xs = [view_matrix(n, kts) for n in meta["members"][mi]]
        X = hstack(Xs, format="csr")
        d = X @ np.load(SVCD + f"/members/m{mi}_coef.npy").T + np.load(SVCD + f"/members/m{mi}_intercept.npy")
        KL += np.log(np.clip(softmax(d), 1e-9, 1))
    Pk = softmax(KL)
    log("kimi rows (leaky, sanity):", f"BA {ba(ky, Pk):.4f}")
print("TUNE DONE", f"({time.time()-T0:.0f}s)")
