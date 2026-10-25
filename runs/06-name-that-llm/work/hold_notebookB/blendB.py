# Name That LLM - final pipeline (run 06): multi-view SVC + Qwen3-4B early-layer heads
# + prompt-group twin + Sinkhorn + em-dash guard. CPU+T4, internet on, pools attached.
import gc, glob, json, os, re, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import torch
T0 = time.time()
ROOT = "/kaggle/input"

LABELS = ['gpt-5.5','gpt-5.6-luna','gpt-5.6-sol','gpt-5.6-terra','gpt-6-astra',
          'gemini-3.1-pro-preview','gemini-3.6-flash','gemini-3.7-flash','gemini-3.8-flash',
          'claude-fable-5','claude-opus-4-6','claude-opus-4-8','claude-opus-5','claude-sonnet-5',
          'qwen3.8-max','glm-5.3','glm-5.3-flash','kimi-k3']
K = len(LABELS)
LAB2I = {l: i for i, l in enumerate(LABELS)}

def softmax(x):
    e = np.exp(x - x.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)

def find_file(name, root=ROOT):
    hits = sorted(glob.glob(os.path.join(root, "**", name), recursive=True))
    if not hits:
        raise FileNotFoundError(name)
    return hits[0]

comp = next(p for p in ["/kaggle/input/competitions/name-that-llm", "/kaggle/input/name-that-llm"] if os.path.exists(p + "/train.csv"))
# Kaggle dataset-cache hiccup mitigation: bulk-copy attached inputs to /tmp once (sequential reads stay fast)
os.makedirs("/kaggle/working/ntli", exist_ok=True)
for fn in ("train.csv", "test.csv"):
    dst = f"/kaggle/working/ntli/{fn}"
    if not os.path.exists(dst):
        with open(comp + "/" + fn, "rb") as fsrc, open(dst, "wb") as fdst:
            while True:
                buf = fsrc.read(1 << 24)
                if not buf: break
                fdst.write(buf)
for src, dst in [(find_file("ntl_pool_v4.parquet"), "/kaggle/working/ntli/pool.parquet"), (find_file("passages.csv"), "/kaggle/working/ntli/passages.csv")]:
    if not os.path.exists(dst):
        with open(src, "rb") as fsrc, open(dst, "wb") as fdst:
            while True:
                buf = fsrc.read(1 << 24)
                if not buf: break
                fdst.write(buf)
print("inputs copied", f"({time.time() - T0:.0f}s)", flush=True)
train = pd.read_csv("/kaggle/working/ntli/train.csv")
test = pd.read_csv("/kaggle/working/ntli/test.csv")
test["text"] = test["text"].fillna("").astype(str)
y_train = train["model"].map(LAB2I).to_numpy()
print("train", train.shape, "test", test.shape, flush=True)

# ================= 1) SVC member =================
pool = pd.read_parquet("/kaggle/working/ntli/pool.parquet")
prv = pd.read_csv("/kaggle/working/ntli/passages.csv")
prv["passes_funnel"] = prv["passes_funnel"].astype(str).str.lower().isin(["true", "1"])
prv_clean = prv[prv["passes_funnel"] & prv["model"].isin(LABELS)].copy()
# final fit: everything (incl. 'new' clean rows - host test prompts are unseen)
prv_all = prv_clean.copy()
prv_all_k = prv[(prv["split"] == "new_kimi_nothink") & (prv["model"].isin(LABELS))]  # extra kimi rows
fit = pd.concat([train[["text", "model"]], pool[["text", "model"]],
                 prv_all[["text", "model"]], prv_all_k[["text", "model"]]], ignore_index=True)
fit = fit[fit["model"].isin(LABELS)].reset_index(drop=True)
fit_y = fit["model"].map(LAB2I).to_numpy()
print("fit rows", len(fit), flush=True)

FUNCTION_WORDS = set("a an the and or but if while although because as so than that which who whom whose this these those it its they them their theirs he him his she her hers we us our ours you your yours i me my mine in on at to from with without for of by over under between through into about is are was were be been being have has had do does did can could will would shall should may might must not no nor all any some each every both either neither other another such only also just very too more most less least here there when where why how what whether then now still yet even much many few little".split())
VIEWS = [("char", 3, 5, 250000, 1.0), ("word", 1, 3, 180000, 0.7), ("shape", 2, 6, 60000, 0.5), ("function", 1, 3, 50000, 0.5)]
def shape_text(t):
    return re.sub(r"[^\W\d_]+", lambda m: "X" if m[0].isupper() else "x", t)
def function_text(t):
    return " ".join(w.lower() if w.lower() in FUNCTION_WORDS or not w.isalnum() else "CONTENT"
                    for w in re.findall(r"\w+|[^\w\s]", t))
def prep(view, t):
    return shape_text(t) if view == "shape" else function_text(t) if view == "function" else t

from scipy.sparse import hstack
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

vecs = []
for view, lo, hi, mf, w in VIEWS:
    opts = dict(analyzer="char" if view == "shape" else "word", ngram_range=(lo, hi), lowercase=False,
                sublinear_tf=True, min_df=2, max_features=mf, dtype=np.float32)
    if view == "word":
        opts["token_pattern"] = r"(?u)\b\w+\b|[^\w\s]"
    v = TfidfVectorizer(**opts)
    v.fit([prep(view, t) for t in fit["text"].astype(str)])
    vecs.append((view, v, w))
def transform(texts):
    return hstack([v.transform([prep(view, t) for t in texts]) * w for view, v, w in vecs], format="csr")

Xf = transform(fit["text"].astype(str).tolist())
svc = LinearSVC(C=0.25, random_state=42, max_iter=10000, class_weight="balanced").fit(Xf, fit_y)
Xt = transform(test["text"].tolist())
P_svc = softmax(svc.decision_function(Xt))
del Xf, Xt
gc.collect()
print("svc done", f"({time.time() - T0:.0f}s)", flush=True)

# test prompt clusters (content TF-IDF, average linkage, cut 0.85)
Xc = TfidfVectorizer(stop_words="english", sublinear_tf=True, min_df=1).fit_transform(test["text"].tolist())
D = np.clip(1 - (Xc @ Xc.T).toarray(), 0, None)
np.fill_diagonal(D, 0)
if len(test) >= 2:
    groups = fcluster(linkage(squareform(D, checks=False), method="average"), 0.85, criterion="distance")
else:
    groups = np.ones(len(test), dtype=int)
size = np.bincount(groups)[groups]
twin_rows = (size >= 2) & (size <= 18)
print(f"clusters: {int((size == 1).sum())} singletons, {int(twin_rows.sum())} rows in 2-18 clusters", flush=True)

# ================= 2) Qwen3-4B heads (attached dataset natmin322/qwen3-4b) =================
from transformers import AutoModel, AutoTokenizer
from sklearn.linear_model import LogisticRegression

LAYERS = (1, 2, 3, 5, 8)
cands = [d for d in (os.path.dirname(p) for p in glob.glob(ROOT + "/**/model.safetensors.index.json", recursive=True))
         if os.path.exists(os.path.join(d, "config.json"))]
assert cands, "attach natmin322/qwen3-4b"
src_dir = sorted(cands)[0]
model_dir = "/kaggle/working/ntli/model"
if not os.path.exists(os.path.join(model_dir, "config.json")):
    os.makedirs(model_dir, exist_ok=True)
    for fn in os.listdir(src_dir):
        sp = os.path.join(src_dir, fn)
        dp = os.path.join(model_dir, fn)
        if os.path.isfile(sp) and not os.path.exists(dp):
            with open(sp, "rb") as fsrc, open(dp, "wb") as fdst:
                while True:
                    buf = fsrc.read(1 << 24)
                    if not buf: break
                    fdst.write(buf)
print("model dir:", model_dir, f"({time.time() - T0:.0f}s)", flush=True)
tok = AutoTokenizer.from_pretrained(model_dir)
tok.padding_side = "right"
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModel.from_pretrained(model_dir, torch_dtype=torch.float16).to("cuda").eval()
H = model.config.hidden_size
MAXLEN = 320

@torch.no_grad()
def feats(texts, bs=16):
    ids_all = [tok(t, add_special_tokens=False, truncation=True, max_length=MAXLEN)["input_ids"] + [tok.eos_token_id] for t in texts]
    Z = np.zeros((len(ids_all), H * len(LAYERS)), np.float32)
    order = np.argsort([len(x) for x in ids_all], kind="stable")
    t0 = time.time()
    for s in range(0, len(order), bs):
        idx = order[s:s + bs]
        enc = tok.pad({"input_ids": [ids_all[i] for i in idx]}, return_tensors="pt").to("cuda")
        hs = model(**enc, output_hidden_states=True, use_cache=False).hidden_states
        m = enc["attention_mask"].unsqueeze(-1).float().cuda(); m[:, 0] = 0
        den = m.sum(1).clamp(min=1)
        parts = [( (hs[l].float().cuda() * m).sum(1) / den ).half().cpu().numpy().astype(np.float32) for l in LAYERS]
        Z[idx] = np.concatenate(parts, 1)
        if s % (bs * 50) == 0:
            print(f"  feats {s + bs}/{len(order)} ({(s + bs) / max(time.time() - t0, 1e-9):.0f} rows/s)", flush=True)
        del hs, enc
        torch.cuda.empty_cache()
    return Z

Zfit = feats(fit["text"].astype(str).tolist())
Ztest = feats(test["text"].tolist())
print("llm features done", f"({time.time() - T0:.0f}s)", flush=True)

# plain head
plain = LogisticRegression(C=0.05, max_iter=2000, class_weight="balanced", solver="lbfgs").fit(Zfit, fit_y)
P_plain = plain.predict_proba(Ztest)
# twin head (centered features of prompt groups 2..18)
n_tr, n_po = len(train), len(pool)
grp = np.array(["tr%d" % i for i in range(n_tr)] + [f"pool_{g}" for g in pool["group"].astype(str)]
               + ["prv%s" % g for g in prv_all["prompt_id"].astype(str)] + ["k%s" % g for g in prv_all_k["prompt_id"].astype(str)])
cnt = pd.Series(grp).value_counts()
good = np.fromiter((g for g, c in cnt.items() if 2 <= c <= 18), dtype=grp.dtype)
twin_mask = np.isin(grp, good)
Zt = Zfit[twin_mask].copy()
ga = grp[twin_mask]
for g in np.unique(ga):
    Zt[ga == g] -= Zt[ga == g].mean(0)
twin = LogisticRegression(C=0.05, max_iter=2000, class_weight="balanced", solver="lbfgs").fit(Zt, fit_y[twin_mask])
Zt2 = Ztest[twin_rows].copy()
gt = groups[twin_rows]
for g in np.unique(gt):
    Zt2[gt == g] -= Zt2[gt == g].mean(0)
P_twin = twin.predict_proba(Zt2)

# ================= 3) blend =================
W_SVC = 1.0      # placeholder until calibrated locally
W_LLMP = 1.0
W_TWIN = 1.0
LOGIT = W_SVC * np.log(np.clip(P_svc, 1e-9, 1.0)) + W_LLMP * np.log(np.clip(P_plain, 1e-9, 1.0))
LOGIT[twin_rows] += W_TWIN * np.log(np.clip(P_twin, 1e-9, 1.0))
P = softmax(LOGIT)

# em-dash guard
SPACED_ALLOWED = {'claude-opus-4-6', 'claude-opus-5', 'glm-5.3', 'glm-5.3-flash'}
forB = np.array([l not in SPACED_ALLOWED for l in LABELS])
for i, t in enumerate(test["text"]):
    if " \u2014 " in t and LABELS[P[i].argmax()] not in SPACED_ALLOWED:
        P[i, forB] = 0.0; P[i] /= P[i].sum()

if len(P) >= 50:
    target = np.full(K, len(P) / K)
    for _ in range(100):
        P *= target / P.sum(0); P /= P.sum(1, keepdims=True)

sub = pd.DataFrame({"id": test["id"].astype(str), "model": np.array(LABELS)[P.argmax(1)]})
sub.to_parquet("submission.parquet", index=False)
print(sub["model"].value_counts().to_string())
print(f"done {time.time() - T0:.0f}s")
