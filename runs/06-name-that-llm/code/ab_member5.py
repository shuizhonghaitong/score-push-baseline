# A/B: does a 5th SVC member (w14, C=1.0) improve the clean-proxy blend?
# Members m0-m3 loaded from prefit artifacts; m4 fit fresh on nb fit set (long; CPU).
import os, sys, time, pickle, re, gc
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from common import *
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
T0 = time.time()


train, pool, prv = load_data()
prv["pf"] = prv["passes_funnel"].astype(str).str.lower().isin(["true", "1"])
prv_clean = prv[prv["pf"] & prv["model"].isin(LABELS)]
prv_kimi = prv[(prv["split"] == "new_kimi_nothink") & (prv["model"].isin(LABELS))]
fit = pd.concat([train[["text","model"]], pool[["text","model"]], prv_clean[["text","model"]], prv_kimi[["text","model"]]], ignore_index=True)
fit = fit[fit["model"].isin(LABELS)].reset_index(drop=True)
fit_t = fit["text"].astype(str).tolist(); fit_y = fit["model"].map(LAB2I).to_numpy()
proxy = prv[(prv["pf"]) & (prv["model"].isin(LABELS)) & (prv["split"] == "new")].reset_index(drop=True)
proxy["y"] = proxy["model"].map(LAB2I)
pvx = set(pool[pool["src"] == "pvx"]["text"])
cm = (~proxy["text"].isin(pvx)).to_numpy()
yc = proxy["y"][cm].to_numpy(); vt = proxy["text"].astype(str).tolist()
print("fit", len(fit_t), "val clean", int(cm.sum()))

SVCD = "work/svc4"
meta = pickle.load(open(SVCD + "/meta.pkl", "rb"))
CHAR = dict(lowercase=False, sublinear_tf=True, min_df=2, dtype=np.float32)
WORD = dict(lowercase=False, sublinear_tf=True, min_df=2, dtype=np.float32, token_pattern=r"(?u)\b\w+\b|[^\w\s]")
VTAG = {"w14": dict(analyzer="word", ngram_range=(1,4), max_features=180000, **WORD)}
# m0-m3 from artifacts
from sklearn.feature_extraction.text import TfidfTransformer
VTAG_ALL = {
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
PREP = {"raw": None, "shape": shape_text, "function": function_text}
def member_probs(member, texts, coef, inter):
    Xs = []
    for n in member:
        p = PREP.get(meta["views"][n]["prep"])
        t = [p(x) for x in texts] if p else texts
        v = TfidfVectorizer(**VTAG_ALL[n]); v.vocabulary_ = pickle.load(open(SVCD + f"/views/{n}_vocab.pkl", "rb"))
        v._tfidf = TfidfTransformer(norm="l2", use_idf=True, smooth_idf=True, sublinear_tf=True)
        v._tfidf.idf_ = np.load(SVCD + f"/views/{n}_idf.npy")
        Xs.append(v.transform(t) * meta["views"][n]["w"])
    X = hstack(Xs, format="csr")
    return softmax(X @ coef.T + inter)
PM = []
for mi, member in enumerate(meta["members"]):
    PM.append(member_probs(member, vt, np.load(SVCD + f"/members/m{mi}_coef.npy"), np.load(SVCD + f"/members/m{mi}_intercept.npy")))
    print(f"artifact member {mi} ok ({time.time()-T0:.0f}s)", flush=True)

# m4: w14 C=1.0 fit fresh
print("fitting m4 (w14 C=1.0) on fit...", flush=True)
v = TfidfVectorizer(**VTAG["w14"])
Xf = v.fit_transform(fit_t)
Xv = v.transform(vt)
est = LinearSVC(C=1.0, random_state=42, max_iter=10000, class_weight="balanced").fit(Xf, fit_y)
PM4 = softmax(est.decision_function(Xv))
print("m4 fit done", f"({time.time()-T0:.0f}s)", flush=True)

def blend(Ps, ws):
    L = sum(w * np.log(np.clip(p, 1e-9, 1)) for w, p in zip(ws, Ps))
    P = softmax(L)
    return ba(yc, P), ba(yc, sinkhorn(P))
b4, s4 = blend(PM, [1, 1, 1, 1])
print(f"4-member 1:1:1:1 clean: {b4:.4f} / sink {s4:.4f}")
for w5 in (0.25, 0.5, 0.75, 1.0, 1.5):
    b, s = blend(PM + [PM4], [1, 1, 1, 1, w5])
    print(f"5-member w5={w5}: {b:.4f} / sink {s:.4f}")
np.savez_compressed("work/ab_member5.npz", **{f"P{i}": p for i, p in enumerate(PM + [PM4])}, y=proxy["y"].to_numpy(), clean=cm)
print("DONE", f"({time.time()-T0:.0f}s)")
