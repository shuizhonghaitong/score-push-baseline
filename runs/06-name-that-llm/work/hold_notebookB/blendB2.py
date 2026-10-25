# Name That LLM - final blend (run 06, v3): 4-member TF-IDF SVC ensemble + Qwen3-4B
# early-layer logistic head (frozen features, layers 1,2,3,5,8 mean-pooled), log-prob
# blend 1:1, Sinkhorn balancing, spaced em-dash guard. T4x1, internet OFF, pools+model attached.
import gc, glob, os, re, time, warnings
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
    if not hits: raise FileNotFoundError(name)
    return hits[0]

# ---- inputs (bulk-copy to /kaggle/working to dodge flaky dataset cache) ----
comp = next(p for p in ["/kaggle/input/competitions/name-that-llm", "/kaggle/input/name-that-llm"] if os.path.exists(p + "/train.csv"))
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
cands = [d for d in (os.path.dirname(p) for p in glob.glob(ROOT + "/**/model.safetensors.index.json", recursive=True))
         if os.path.exists(os.path.join(d, "config.json"))]
assert cands, "attach natmin322/qwen3-4b"
src_dir = sorted(cands)[0]
model_dir = "/kaggle/working/ntli/model"
if not os.path.exists(os.path.join(model_dir, "config.json")):
    os.makedirs(model_dir, exist_ok=True)
    for fn in os.listdir(src_dir):
        sp, dp = os.path.join(src_dir, fn), os.path.join(model_dir, fn)
        if os.path.isfile(sp) and not os.path.exists(dp):
            with open(sp, "rb") as fsrc, open(dp, "wb") as fdst:
                while True:
                    buf = fsrc.read(1 << 24)
                    if not buf: break
                    fdst.write(buf)
print("inputs copied", f"({time.time() - T0:.0f}s)", flush=True)

train = pd.read_csv("/kaggle/working/ntli/train.csv")
test = pd.read_csv("/kaggle/working/ntli/test.csv")
test["text"] = test["text"].fillna("").astype(str)
print("train", train.shape, "test", test.shape, flush=True)

pool = pd.read_parquet("/kaggle/working/ntli/pool.parquet")
prv = pd.read_csv("/kaggle/working/ntli/passages.csv")
prv["passes_funnel"] = prv["passes_funnel"].astype(str).str.lower().isin(["true", "1"])
prv_clean = prv[prv["passes_funnel"] & prv["model"].isin(LABELS)].copy()
prv_kimi = prv[(prv["split"] == "new_kimi_nothink") & (prv["model"].isin(LABELS))]
fit = pd.concat([train[["text", "model"]], pool[["text", "model"]],
                 prv_clean[["text", "model"]], prv_kimi[["text", "model"]]], ignore_index=True)
fit = fit[fit["model"].isin(LABELS)].reset_index(drop=True)
fit_y = fit["model"].map(LAB2I).to_numpy()
fit_t = fit["text"].astype(str).tolist()
test_t = test["text"].tolist()
print("fit rows", len(fit), flush=True)

# ================= 1) 4-member TF-IDF SVC ensemble =================
FUNCTION_WORDS = set("a an the and or but if while although because as so than that which who whom whose this these those it its they them their theirs he him his she her hers we us our ours you your yours i me my mine in on at to from with without for of by over under between through into about is are was were be been being have has had do does did can could will would shall should may might must not no nor all any some each every both either neither other another such only also just very too more most less least here there when where why how what whether then now still yet even much many few little".split())

def shape_text(t):
    return re.sub(r"[^\W\d_]+", lambda m: "X" if m[0].isupper() else "x", t)

def function_text(t):
    return " ".join(w.lower() if w.lower() in FUNCTION_WORDS or not w.isalnum() else "CONTENT"
                    for w in re.findall(r"\w+|[^\w\s]", t))

from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

def fit_svc(texts_fit, texts_test, views):
    mats_f, mats_t = [], []
    for view, tag, w in views:
        prep = shape_text if view == "shape" else function_text if view == "function" else (lambda s: s)
        v = TfidfVectorizer(**tag)
        v.fit([prep(t) for t in texts_fit])
        mats_f.append(v.transform([prep(t) for t in texts_fit]) * w)
        mats_t.append(v.transform([prep(t) for t in texts_test]) * w)
    Xf = hstack(mats_f, format="csr")
    est = LinearSVC(C=0.25, random_state=42, max_iter=10000, class_weight="balanced").fit(Xf, fit_y)
    del Xf, mats_f
    gc.collect()
    return softmax(est.decision_function(hstack(mats_t, format="csr")))

CHAR = dict(lowercase=False, sublinear_tf=True, min_df=2, dtype=np.float32)
WORD = dict(lowercase=False, sublinear_tf=True, min_df=2, dtype=np.float32, token_pattern=r"(?u)\b\w+\b|[^\w\s]")
MEMBERS = [
    [("char", dict(analyzer="char", ngram_range=(3, 5), max_features=250000, **CHAR), 1.0),
     ("word", dict(analyzer="word", ngram_range=(1, 3), max_features=180000, **WORD), 0.7),
     ("shape", dict(analyzer="char", ngram_range=(2, 6), max_features=60000, **CHAR), 0.5),
     ("function", dict(analyzer="word", ngram_range=(1, 3), max_features=50000, **WORD), 0.5)],
    [("raw", dict(analyzer="char", ngram_range=(2, 6), max_features=250000, **CHAR), 1.0)],
    [("raw", dict(analyzer="char", ngram_range=(3, 7), max_features=250000, **CHAR), 1.0)],
    [("raw", dict(analyzer="word", ngram_range=(1, 4), max_features=180000, **WORD), 1.0)],
]
LOGIT_SVC = np.zeros((len(test_t), K))
for i, views in enumerate(MEMBERS):
    P = fit_svc(fit_t, test_t, views)
    LOGIT_SVC += np.log(np.clip(P, 1e-9, 1.0))
    del P
    print(f"svc member {i} done ({time.time() - T0:.0f}s)", flush=True)

# ================= 2) Qwen3-4B early-layer head =================
from transformers import AutoModel, AutoTokenizer
from sklearn.linear_model import LogisticRegression

LAYERS = (1, 2, 3, 5, 8)
tok = AutoTokenizer.from_pretrained(model_dir)
tok.padding_side = "right"
if tok.pad_token is None: tok.pad_token = tok.eos_token
model = AutoModel.from_pretrained(model_dir, torch_dtype=torch.float16).to("cuda").eval()
H = model.config.hidden_size
MAXLEN = 320

@torch.no_grad()
def feats(texts, bs=16, tag=""):
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
        parts = [((hs[l].float().cuda() * m).sum(1) / den).half().cpu().numpy().astype(np.float32) for l in LAYERS]
        Z[idx] = np.concatenate(parts, 1)
        if s % (bs * 100) == 0:
            print(f"  {tag} {s + bs}/{len(order)} ({(s + bs) / max(time.time() - t0, 1e-9):.0f} rows/s)", flush=True)
        del hs, enc
        torch.cuda.empty_cache()
    return Z

Zfit = feats(fit_t, tag="fit")
print("fit features done", f"({time.time() - T0:.0f}s)", flush=True)
Ztest = feats(test_t, tag="test")
del model
gc.collect(); torch.cuda.empty_cache()

head = LogisticRegression(C=0.05, max_iter=2000, class_weight="balanced", solver="lbfgs").fit(Zfit, fit_y)
P_qwen = head.predict_proba(Ztest)
del Zfit
gc.collect()
print("qwen head done", f"({time.time() - T0:.0f}s)", flush=True)

# ================= 3) blend =================
W_SVC, W_QWEN = 1.0, 1.0
P = softmax(W_SVC * LOGIT_SVC + W_QWEN * np.log(np.clip(P_qwen, 1e-9, 1.0)))

SPACED_ALLOWED = {'claude-opus-4-6', 'claude-opus-5', 'glm-5.3', 'glm-5.3-flash'}
forB = np.array([l not in SPACED_ALLOWED for l in LABELS])
for i, t in enumerate(test_t):
    if " \u2014 " in t and LABELS[P[i].argmax()] not in SPACED_ALLOWED:
        P[i, forB] = 0.0
        P[i] /= P[i].sum()

if len(P) >= 50:
    target = np.full(K, len(P) / K)
    for _ in range(100):
        P *= target / P.sum(0)
        P /= P.sum(1, keepdims=True)

sub = pd.DataFrame({"id": test["id"].astype(str), "model": np.array(LABELS)[P.argmax(1)]})
sub.to_parquet("submission.parquet", index=False)
sub.to_csv("submission.csv", index=False)
np.savez_compressed("blend_probs.npz", svc=np.exp(LOGIT_SVC), qwen=P_qwen)
print(sub["model"].value_counts().to_string())
print(f"done {time.time() - T0:.0f}s")
