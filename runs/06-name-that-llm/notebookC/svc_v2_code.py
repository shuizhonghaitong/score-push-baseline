# Name That LLM - multi-view TF-IDF SVC ensemble (run 06, v2)
# 4 SVC members (base 4-view, char 2-6, char 3-7, word 1-4), log-prob blend 1:1:1:1,
# Sinkhorn balancing, spaced em-dash guard. CPU, internet OFF, pools attached.
import gc, glob, os, re, time, warnings
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
val_t = test["text"].tolist()
print("fit rows", len(fit), flush=True)

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
    ("base4", [("char", dict(analyzer="char", ngram_range=(3, 5), max_features=250000, **CHAR), 1.0),
               ("word", dict(analyzer="word", ngram_range=(1, 3), max_features=180000, **WORD), 0.7),
               ("shape", dict(analyzer="char", ngram_range=(2, 6), max_features=60000, **CHAR), 0.5),
               ("function", dict(analyzer="word", ngram_range=(1, 3), max_features=50000, **WORD), 0.5)]),
    ("char26", [("raw", dict(analyzer="char", ngram_range=(2, 6), max_features=250000, **CHAR), 1.0)]),
    ("char37", [("raw", dict(analyzer="char", ngram_range=(3, 7), max_features=250000, **CHAR), 1.0)]),
    ("word14", [("raw", dict(analyzer="word", ngram_range=(1, 4), max_features=180000, **WORD), 1.0)]),
]
LOGIT = np.zeros((len(val_t), K))
for name, views in MEMBERS:
    P = fit_svc(fit_t, val_t, views)
    LOGIT += np.log(np.clip(P, 1e-9, 1.0))
    del P
    print(f"member {name} done ({time.time() - T0:.0f}s)", flush=True)
P = softmax(LOGIT)

SPACED_ALLOWED = {'claude-opus-4-6', 'claude-opus-5', 'glm-5.3', 'glm-5.3-flash'}
forB = np.array([l not in SPACED_ALLOWED for l in LABELS])
for i, t in enumerate(val_t):
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
print(sub["model"].value_counts().to_string())
print(f"done {time.time() - T0:.0f}s")
