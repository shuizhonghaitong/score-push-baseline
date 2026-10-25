# v3 CPU ensemble: candidate member search under the CLEAN protocol.
# fit_clean = train + pool[-pvx,-pv1] + prv[funnel, split!='new'] + kimi(nothink)
# val       = prv[split=='new', funnel]  (5326 rows; model has never seen them)
# fit_full  = train + pool(all) + prv[funnel, all splits] + kimi  (deploy corpus, same as v2)
# Cache TF-IDF matrices under work/v3_cache/. Single-process per view (parallel via pool).
import gc, os, pickle, sys, time, re
from concurrent.futures import ProcessPoolExecutor
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import LABELS, LAB2I, ROOT, softmax, sinkhorn, ba
from scipy.sparse import save_npz, load_npz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
T0 = time.time()
def log(*a, **kw): print(f"[{time.time()-T0:7.0f}s]", *a, flush=True)

CACHE = f"{ROOT}/work/v3_cache"
os.makedirs(CACHE, exist_ok=True)

train, pool, prv = pd.read_csv(f"{ROOT}/data/train.csv"), pd.read_parquet(f"{ROOT}/pool/v4/ntl_pool_v4.parquet"), pd.read_csv(f"{ROOT}/pool/passages.csv")
prv["pf"] = prv["passes_funnel"].astype(str).str.lower().isin(["true", "1"])
prv_clean = prv[prv["pf"] & prv["model"].isin(LABELS)]
prv_kimi = prv[(prv["split"] == "new_kimi_nothink") & prv["model"].isin(LABELS)]
val = prv[(prv["split"] == "new") & prv["pf"] & prv["model"].isin(LABELS)].copy()
val["y"] = val["model"].map(LAB2I); val = val.reset_index(drop=True)
y_val = val["y"].to_numpy(); val_t = val["text"].astype(str).tolist()

fit_clean = pd.concat([train[["text","model"]], pool[~pool["src"].isin(("pvx","pv1"))][["text","model"]],
                       prv_clean[prv_clean["split"] != "new"][["text","model"]], prv_kimi[["text","model"]]], ignore_index=True)
fit_full = pd.concat([train[["text","model"]], pool[["text","model"]], prv_clean[["text","model"]], prv_kimi[["text","model"]]], ignore_index=True)
fit_clean = fit_clean[fit_clean["model"].isin(LABELS)].reset_index(drop=True)
fit_full = fit_full[fit_full["model"].isin(LABELS)].reset_index(drop=True)
log("rows clean/full/val:", len(fit_clean), len(fit_full), len(val))
fit_clean_t = fit_clean["text"].astype(str).tolist(); fit_clean_y = fit_clean["model"].map(LAB2I).to_numpy()
fit_full_t = fit_full["text"].astype(str).tolist(); fit_full_y = fit_full["model"].map(LAB2I).to_numpy()

FUNCTION_WORDS = set("a an the and or but if while although because as so than that which who whom whose this these those it its they them their theirs he him his she her hers we us our ours you your yours i me my mine in on at to from with without for of by over under between through into about is are was were be been being have has had do does did can could will would shall should may might must not no nor all any some each every both either neither other another such only also just very too more most less least here there when where why how what whether then now still yet even much many few little".split())
def shape_text(t): return re.sub(r"[^\W\d_]+", lambda m: "X" if m[0].isupper() else "x", t)
def function_text(t): return " ".join(w.lower() if w.lower() in FUNCTION_WORDS or not w.isalnum() else "CONTENT" for w in re.findall(r"\w+|[^\w\s]", t))
PREP = {"raw": None, "shape": shape_text, "function": function_text}

CHAR = dict(lowercase=False, sublinear_tf=True, min_df=2, dtype=np.float32)
WORD = dict(lowercase=False, sublinear_tf=True, min_df=2, dtype=np.float32, token_pattern=r"(?u)\b\w+\b|[^\w\s]")
# name: (prep, vtag, w)
VIEWS = {
 "c24":  ("raw", dict(analyzer="char", ngram_range=(2,4),  max_features=250000, **CHAR), 1.0),
 "c26":  ("raw", dict(analyzer="char", ngram_range=(2,6),  max_features=250000, **CHAR), 1.0),
 "c35":  ("raw", dict(analyzer="char", ngram_range=(3,5),  max_features=250000, **CHAR), 1.0),
 "c37":  ("raw", dict(analyzer="char", ngram_range=(3,7),  max_features=250000, **CHAR), 1.0),
 "c48":  ("raw", dict(analyzer="char", ngram_range=(4,8),  max_features=250000, **CHAR), 1.0),
 "c26_400": ("raw", dict(analyzer="char", ngram_range=(2,6), max_features=400000, **CHAR), 1.0),
 "c35_100": ("raw", dict(analyzer="char", ngram_range=(3,5), max_features=100000, **CHAR), 1.0),
 "sh26": ("shape", dict(analyzer="char", ngram_range=(2,6), max_features=60000, **CHAR), 1.0),
 "sh37": ("shape", dict(analyzer="char", ngram_range=(3,7), max_features=60000, **CHAR), 1.0),
 "fn13": ("function", dict(analyzer="word", ngram_range=(1,3), max_features=50000, **WORD), 1.0),
 "w12":  ("raw", dict(analyzer="word", ngram_range=(1,2),  max_features=180000, **WORD), 1.0),
 "w13":  ("raw", dict(analyzer="word", ngram_range=(1,3),  max_features=180000, **WORD), 1.0),
 "w14":  ("raw", dict(analyzer="word", ngram_range=(1,4),  max_features=180000, **WORD), 1.0),
 "w15":  ("raw", dict(analyzer="word", ngram_range=(1,5),  max_features=180000, **WORD), 1.0),
 "w25":  ("raw", dict(analyzer="word", ngram_range=(2,5),  max_features=180000, **WORD), 1.0),
 "w13_400": ("raw", dict(analyzer="word", ngram_range=(1,3), max_features=400000, **WORD), 1.0),
}

def build_view(name):
    t0 = time.time()
    prep, vtag, w = VIEWS[name]
    f = PREP.get(prep)
    Xc = None
    cf = f"{CACHE}/{name}_clean_fit.npz"
    cv = f"{CACHE}/{name}_clean_val.npz"
    if os.path.exists(cf) and os.path.exists(cv):
        Xc, Xv = load_npz(cf), load_npz(cv)
    else:
        v = TfidfVectorizer(**vtag)
        Xc = v.fit_transform([f(t) for t in fit_clean_t] if f else fit_clean_t)
        Xv = v.transform([f(t) for t in val_t] if f else val_t)
        with open(f"{CACHE}/{name}_vec.pkl","wb") as fh: pickle.dump(v, fh)
        save_npz(cf, Xc); save_npz(cv, Xv)
    log(f"view {name}: {Xc.shape} nnz={Xc.nnz} ({time.time()-t0:.0f}s cached)", flush=True)
    return Xc, Xv

def fit_svc(name, C=0.25):
    Xc, Xv = build_view(name)
    tag = C if C == 0.25 else f"C{C}"
    cp = f"{CACHE}/{name}_{tag}_Pval.npy"
    if os.path.exists(cp):
        P = np.load(cp); log(f"svc {name}/{tag} cached"); return P
    est = LinearSVC(C=C, random_state=42, max_iter=20000, class_weight="balanced").fit(Xc, fit_clean_y)
    P = softmax(est.decision_function(Xv))
    np.save(cp, P)
    with open(f"{CACHE}/{name}_{tag}_coef.npy","wb") as fh: np.save(fh, est.coef_.astype(np.float32))
    np.save(f"{CACHE}/{name}_{tag}_intercept.npy", est.intercept_.astype(np.float32))
    log(f"solo {name}/{tag}: BA {ba(y_val,P):.4f} / sink {ba(y_val,sinkhorn(P)):.4f}", flush=True)
    return P

if __name__ == "__main__":
    names = list(VIEWS)
    P = {}
    with ProcessPoolExecutor(max_workers=8) as ex:
        futs = {n: ex.submit(fit_svc, n) for n in names}
        for n in names:
            P[n] = futs[n].result()
    # combo member base4 = c35 + w13 + sh26 + fn13 (v2's member0): build from view mats
    prep_w = {n: VIEWS[n][2] for n in ("c35","w13","sh26","fn13")}
    from scipy.sparse import hstack
    Xc = hstack([load_npz(f"{CACHE}/{n}_clean_fit.npz") * w for n, w in prep_w.items()], format="csr")
    Xv = hstack([load_npz(f"{CACHE}/{n}_clean_val.npz") * w for n, w in prep_w.items()], format="csr")
    est = LinearSVC(C=0.25, random_state=42, max_iter=20000, class_weight="balanced").fit(Xc, fit_clean_y)
    P["base4"] = softmax(est.decision_function(Xv))
    np.save(f"{CACHE}/base4_Pval.npy", P["base4"])
    np.save(f"{CACHE}/base4_coef.npy", est.coef_.astype(np.float32))
    np.save(f"{CACHE}/base4_intercept.npy", est.intercept_.astype(np.float32))
    log(f"solo base4: BA {ba(y_val,P['base4']):.4f} / sink {ba(y_val,sinkhorn(P['base4'])):.4f}")

    # greedy forward selection on equal-weight log-prob blend + sinkhorn
    def blend(names_sel, weights=None):
        L = np.zeros((len(y_val), 18))
        for i, n in enumerate(names_sel):
            L += (1.0 if weights is None else weights[i]) * np.log(np.clip(P[n], 1e-9, 1.0))
        return softmax(L)
    sel = []
    for _ in range(8):
        best_n, best_ba = None, -1
        for n in names:
            if n in sel: continue
            Pb = blend(sel + [n])
            b = ba(y_val, sinkhorn(Pb))
            if b > best_ba: best_n, best_ba = n, b
        if best_n is None: break
        sel.append(best_n)
        log(f"greedy +{best_n} -> {sel}: BA {ba(y_val,sinkhorn(blend(sel))):.4f}")
        if len(sel) >= 6: break
    np.savez(f"{CACHE}/sel_Pval.npz", **{n: P[n] for n in names}, y=y_val, sel=",".join(sel))
    log("SEL DONE:", sel)
