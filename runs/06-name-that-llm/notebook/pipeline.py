# Name That LLM - submission pipeline (CPU, internet on)
# 4-view TF-IDF + LinearSVC trained on train + public pools; em-dash guard; Sinkhorn balance.
import glob, json, os, re, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

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
train = pd.read_csv(comp + "/train.csv")
test = pd.read_csv(comp + "/test.csv")
test["text"] = test["text"].fillna("").astype(str)
y = train["model"].map(LAB2I).values
print("train", train.shape, "test", test.shape, flush=True)

# ---------- extra public pools (downloaded in cell 1 into ./extra) ----------
EXTRA = "./extra"
fit_parts, fit_names = [train], ["train"]
pool = pd.read_parquet(os.path.join(EXTRA, "ntl_pool_v4.parquet") if os.path.exists(os.path.join(EXTRA, "ntl_pool_v4.parquet")) else find_file("ntl_pool_v4.parquet"))
fit_parts.append(pool[["text", "model"]]); fit_names.append(f"pool_v4({len(pool)})")
prv = pd.read_csv(os.path.join(EXTRA, "passages.csv") if os.path.exists(os.path.join(EXTRA, "passages.csv")) else find_file("passages.csv"))
prv_c = prv[(prv["passes_funnel"] == True) & (prv["model"].isin(LABELS))]
fit_parts.append(prv_c[["text", "model"]]); fit_names.append(f"prvsiyan({len(prv_c)})")
fit = pd.concat(fit_parts, ignore_index=True)
fit = fit[fit["model"].isin(LABELS)]
fit_y = fit["model"].map(LAB2I).values
print("fit rows:", {n: 0 for n in fit_names}, len(fit), flush=True)

# ---------- features ----------
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
print("vectorizers fit", f"({time.time() - T0:.0f}s)", flush=True)

def transform(texts):
    A = [v.transform([prep(view, t) for t in texts]) * w for view, v, w in vecs]
    return hstack(A, format="csr")

Xf = transform(fit["text"].astype(str).tolist())
svc = LinearSVC(C=0.25, random_state=42, max_iter=10000, class_weight="balanced").fit(Xf, fit_y)
del Xf
import gc; gc.collect()
print("svc fit", f"({time.time() - T0:.0f}s)", flush=True)

P = softmax(svc.decision_function(transform(test["text"].tolist())))

# ---------- em-dash guard (train.csv: spaced em-dash only in 4 models) ----------
SPACED_ALLOWED = {'claude-opus-4-6', 'claude-opus-5', 'glm-5.3', 'glm-5.3-flash'}
forBanned = np.array([l not in SPACED_ALLOWED for l in LABELS])
fixed = 0
for i, t in enumerate(test["text"]):
    if " \u2014 " in t and LABELS[P[i].argmax()] not in SPACED_ALLOWED:
        P[i, forBanned] = 0.0
        P[i] /= P[i].sum()
        fixed += 1
print("em-dash guard fixed", fixed, flush=True)

# ---------- class balance (uniform target; BA rewards per-class recall) ----------
if len(P) >= 50:
    target = np.full(K, len(P) / K)
    for _ in range(100):
        P *= target / P.sum(0)
        P /= P.sum(1, keepdims=True)

sub = pd.DataFrame({"id": test["id"].astype(str), "model": np.array(LABELS)[P.argmax(1)]})
sub.to_parquet("submission.parquet", index=False)
sub.to_csv("submission.csv", index=False)
print(sub["model"].value_counts().to_string())
print(f"done in {time.time() - T0:.0f}s")
