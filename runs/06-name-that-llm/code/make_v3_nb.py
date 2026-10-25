# Build notebookC2 (CPU, no internet) from work/svc5 meta + work/v3_config.json.
import json, os, pickle
ROOT = "/mnt/disk3/liuyutong_home/workspace/skill/score-push/runs/06-name-that-llm"
meta = pickle.load(open(f"{ROOT}/work/svc5/meta.pkl", "rb"))
LABELS = meta["labels"]
CANON = ['gpt-5.5','gpt-5.6-luna','gpt-5.6-sol','gpt-5.6-terra','gpt-6-astra',
 'gemini-3.1-pro-preview','gemini-3.6-flash','gemini-3.7-flash','gemini-3.8-flash',
 'claude-opus-4-6','claude-opus-4-8','claude-opus-5','claude-sonnet-5','claude-fable-5',
 'qwen3.8-max','glm-5.3','glm-5.3-flash','kimi-k3']
ORDER = ",".join(str(LABELS.index(l)) for l in CANON)

SETUP = '''# run06 v3: prefit multi-member TF-IDF SVC ensemble (CPU, no internet)
import gc, glob, os, re, time, pickle, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
T0 = time.time()
def log(*a): print(*a, flush=True)

ROOT = "/kaggle/input"
def find_file(name, root=ROOT):
    hits = sorted(glob.glob(os.path.join(root, "**", name), recursive=True))
    if not hits: raise FileNotFoundError(name)
    return hits[0]
comp = next(p for p in ["/kaggle/input/competitions/name-that-llm", "/kaggle/input/name-that-llm"] if os.path.exists(p + "/train.csv"))
os.makedirs("/kaggle/working/v3", exist_ok=True)
with open(comp + "/test.csv", "rb") as fsrc, open("/kaggle/working/v3/test.csv", "wb") as fdst:
    while True:
        buf = fsrc.read(1 << 24)
        if not buf: break
        fdst.write(buf)
SVCD = None
for dp, dn, fn in os.walk(ROOT):
    if "svc5_meta.pkl" in fn: SVCD = dp; break
assert SVCD, "svc5 dataset not attached"
meta = pickle.load(open(os.path.join(SVCD, "svc5_meta.pkl"), "rb"))
LABELS = meta["labels"]; K = len(LABELS)
test = pd.read_csv("/kaggle/working/v3/test.csv")
test["text"] = test["text"].fillna("").astype(str)
log("test", test.shape, f"({time.time()-T0:.0f}s)")

FUNCTION_WORDS = set("a an the and or but if while although because as so than that which who whom whose this these those it its they them their theirs he him his she her hers we us our ours you your yours i me my mine in on at to from with without for of by over under between through into about is are was were be been being have has had do does did can could will would shall should may might must not no nor all any some each every both either neither other another such only also just very too more most less least here there when where why how what whether then now still yet even much many few little".split())
def shape_text(t): return re.sub(r"[^\\W\\d_]+", lambda m: "X" if m[0].isupper() else "x", t)
def function_text(t): return " ".join(w.lower() if w.lower() in FUNCTION_WORDS or not w.isalnum() else "CONTENT" for w in re.findall(r"\\w+|[^\\w\\s]", t))
PREP = {"raw": None, "shape": shape_text, "function": function_text}
def softmax(x):
    e = np.exp(x - x.max(1, keepdims=True)); return e / e.sum(1, keepdims=True)

VTAGS = {}
def vec_for(vn):
    if vn in VTAGS: return VTAGS[vn]
    vtag = meta["views"][vn]["vtag"]; prep = meta["views"][vn]["prep"]
    v = TfidfVectorizer(**vtag)
    v.vocabulary_ = pickle.load(open(f"{SVCD}/views/{vn}_vocab.pkl", "rb"))
    v._tfidf = TfidfTransformer(norm="l2", use_idf=True, smooth_idf=True, sublinear_tf=True)
    v._tfidf.idf_ = np.load(f"{SVCD}/views/{vn}_idf.npy")
    VTAGS[vn] = (v, prep)
    return VTAGS[vn]

texts = test["text"].tolist()
LP = np.zeros((len(meta["members"]), len(test), K), np.float32)
for mi, m in enumerate(meta["members"]):
    parts = []
    for vn, w in m["combo"]:
        v, prep = vec_for(vn)
        f = PREP[prep]
        X = v.transform([f(t) if f else t for t in texts]) * w
        parts.append(X)
    X = hstack(parts, format="csr") if len(parts) > 1 else parts[0]
    d = X @ np.load(f"{SVCD}/members/m{mi}_coef.npy").T + np.load(f"{SVCD}/members/m{mi}_intercept.npy")
    P = softmax(d.astype(np.float64))
    LP[mi] = np.log(np.clip(P, 1e-9, 1.0))
    log(f"member {m['name']} done ({time.time()-T0:.0f}s)")
    del X, parts; gc.collect()
np.save("/kaggle/working/v3/members_lp.npy", LP)
np.save("/kaggle/working/members_lp.npy", LP)
# column permutation from this ensemble's LABELS order to canonical order
np.save("/kaggle/working/members_order.npy", np.array("__ORDER__".split(","), dtype=np.int32))
'''

MAIN = '''
EXP = list(meta["exponents"].values())
L = np.sum([wi * LP[i] for i, wi in enumerate(EXP)], axis=0)
P = softmax(L)

SPACED_ALLOWED = {"claude-opus-4-6", "claude-opus-5", "glm-5.3", "glm-5.3-flash"}
forb = np.array([l not in SPACED_ALLOWED for l in LABELS])
def emdash_guard(Pm):
    n = 0
    for i, t in enumerate(texts):
        if " \\u2014 " in t and LABELS[Pm[i].argmax()] not in SPACED_ALLOWED:
            Pm[i, forb] = 0.0; Pm[i] /= Pm[i].sum(); n += 1
    return Pm, n
def sinkhorn_bal(Pm, iters=100):
    if len(Pm) < 50: return Pm
    target = np.full(K, len(Pm) / K)
    for _ in range(iters):
        Pm *= target / Pm.sum(0); Pm /= Pm.sum(1, keepdims=True)
    return Pm
def cluster_prompts(X, thr=0.90):
    # average-linkage on cosine distance; validated ARI 0.996 on 55-prompt val subsample.
    # Oversized clusters (>18) are simply skipped by the downstream 12-18 gate (no harm).
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform
    n = len(X)
    if n < 500:
        return np.arange(n)
    D = 1.0 - (X @ X.T).toarray(); np.fill_diagonal(D, 0.0)
    D = np.clip((D + D.T) / 2.0, 0.0, 2.0)
    Z = linkage(squareform(D, checks=False), method="average")
    return np.asarray(fcluster(Z, thr, criterion="distance"), dtype=np.int32)
def hungarian_dense(Pm, groups, lo=12, hi=18):
    from scipy.optimize import linear_sum_assignment
    szg = np.bincount(groups); ch = 0
    for g in np.unique(groups):
        if not (lo <= szg[g] <= hi): continue
        ii = np.where(groups == g)[0]
        r, c = linear_sum_assignment(-np.log(np.clip(Pm[ii], 1e-12, 1.0)))
        for rr, cc in zip(r, c):
            if Pm[ii[rr]].argmax() != cc: ch += 1
            Pm[ii[rr]] = 0.0; Pm[ii[rr], cc] = 1.0
    return Pm, ch

Psafe = sinkhorn_bal(P.copy())
Psafe, n1 = emdash_guard(Psafe)
sub = pd.DataFrame({"id": test["id"].astype(str), "model": np.array(LABELS)[Psafe.argmax(1)]})
sub.to_parquet("/kaggle/working/submission.parquet", index=False)
sub.to_csv("/kaggle/working/submission.csv", index=False)
log(f"safe variant: emdash fixed {n1}; labels {sub['model'].value_counts().to_dict()}")

Ph = sinkhorn_bal(P.copy())
Ph, n2 = emdash_guard(Ph)
try:
    v = TfidfVectorizer(sublinear_tf=True, stop_words="english", min_df=2, max_df=0.05)
    Xc = v.fit_transform(texts)
    G = cluster_prompts(Xc)
except ValueError as e:
    log("clustering fallback (tiny test):", e); G = np.arange(len(texts))
sizes = np.bincount(G)
hist = dict(zip(*np.unique(sizes[sizes > 0], return_counts=True)))
ndense = int((sizes >= 12).sum())
log(f"clusters: {len(np.unique(G))} dense(12-18)={ndense} size_hist={hist}")
if ndense >= 5 and len(test) >= 500:
    for tag, thr in (("hung", 0.90), ("hung85", 0.85)):
        Gt = cluster_prompts(Xc, thr)
        Ph2 = sinkhorn_bal(P.copy())
        Ph2, _ = emdash_guard(Ph2)
        Ph2, ch = hungarian_dense(Ph2, Gt)
        log(f"{tag}: dense={int((np.bincount(Gt) >= 12).sum())} changed {ch}")
        subh = pd.DataFrame({"id": test["id"].astype(str), "model": np.array(LABELS)[Ph2.argmax(1)]})
        subh.to_parquet(f"/kaggle/working/submission_{tag}.parquet", index=False)
        subh.to_csv(f"/kaggle/working/submission_{tag}.csv", index=False)
        log(f"{tag} variant written; diff vs safe: {int((subh['model'].values != sub['model'].values).sum())}")
else:
    log("hung variants skipped (no dense prompt structure detected)")
log(f"v3 done in {time.time()-T0:.0f}s")
'''

import nbformat
nb = nbformat.v4.new_notebook()
nb.cells = [
    nbformat.v4.new_markdown_cell("# run06 v3: prefit TF-IDF SVC ensemble (CPU)\n\nNo GPU, no internet. Attach `lilac1996/ntl-run06-svc5` (6 prefit members). Outputs: submission.parquet (safe), submission_hung.parquet (Hungarian on dense prompt clusters, if structure detected), members_lp.npy (member log-probs, this ensemble's LABELS order), members_order.npy (permutation to canonical order)."),
    nbformat.v4.new_code_cell(SETUP.replace("__ORDER__", ORDER)),
    nbformat.v4.new_code_cell(MAIN),
]
os.makedirs(f"{ROOT}/notebookC2", exist_ok=True)
nbformat.write(nb, f"{ROOT}/notebookC2/ntl_run06_v3_cpu.ipynb")
json.dump({
  "id": "lilac1996/ntl-run06-v3-cpu",
  "title": "ntl-run06-v3-cpu",
  "code_file": "ntl_run06_v3_cpu.ipynb",
  "language": "python",
  "kernel_type": "notebook",
  "is_private": "true",
  "enable_gpu": "false",
  "enable_tpu": "false",
  "enable_internet": "false",
  "machine_shape": "",
  "dataset_sources": ["lilac1996/ntl-run06-svc5"],
}, open(f"{ROOT}/notebookC2/kernel-metadata.json", "w"), indent=1)
print("notebookC2 written; members:", [m["name"] for m in meta["members"]], "order:", ORDER)
