# Local calibration of the SVC x Qwen3-4B head blend weight on CLEAN proxy rows
# (new split minus pvx; head trained on clean rows only to stay honest).
import os, sys, time, glob, pickle, re
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from common import *
os.environ["HF_HOME"] = f"{ROOT}/hf_cache"
os.environ["HF_HUB_DISABLE_XET"] = "1"
import torch
torch.set_num_threads(8)
T0 = time.time()
def log(*a): print(*a)

train, pool, prv = load_data()
prv["pf"] = prv["passes_funnel"].astype(str).str.lower().isin(["true", "1"])
proxy = prv[(prv["pf"]) & (prv["model"].isin(LABELS)) & (prv["split"] == "new")].reset_index(drop=True)
proxy["y"] = proxy["model"].map(LAB2I).to_numpy()
pvx = set(pool[pool["src"] == "pvx"]["text"])
clean_mask = ~proxy["text"].isin(pvx)
log("proxy rows", len(proxy), "clean", int(clean_mask.sum()))

SVCD = "work/svc4"
meta = pickle.load(open(SVCD + "/meta.pkl", "rb"))
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
PREP = {"raw": None, "shape": shape_text, "function": function_text}
texts = proxy["text"].astype(str).tolist()
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
PM = []
for mi, member in enumerate(meta["members"]):
    Xs = []
    for n in member:
        p = PREP.get(meta["views"][n]["prep"])
        t = [p(x) for x in texts] if p else texts
        v = TfidfVectorizer(**VTAG[n]); v.vocabulary_ = pickle.load(open(SVCD + f"/views/{n}_vocab.pkl", "rb"))
        v._tfidf = TfidfTransformer(norm="l2", use_idf=True, smooth_idf=True, sublinear_tf=True)
        v._tfidf.idf_ = np.load(SVCD + f"/views/{n}_idf.npy")
        Xs.append(v.transform(t) * meta["views"][n]["w"])
    X = hstack(Xs, format="csr")
    d = X @ np.load(SVCD + f"/members/m{mi}_coef.npy").T + np.load(SVCD + f"/members/m{mi}_intercept.npy")
    PM.append(softmax(d))
    log(f"SVC member {mi} done ({time.time()-T0:.0f}s)")
    del X, Xs
P_svc = softmax(sum(np.log(np.clip(p, 1e-9, 1)) for p in PM))

# ---- Qwen3-4B features (CPU fp32, MAXLEN 320, layers 1,2,3,5,8) ----
import torch
from transformers import AutoModel, AutoTokenizer
sd = sorted(glob.glob(f"{ROOT}/hf_cache/hub/models--Qwen--Qwen3-4B/snapshots/*"))[0]
LAYERS = (1, 2, 3, 5, 8)
MAXLEN = 320
tok = AutoTokenizer.from_pretrained(sd)
if tok.pad_token is None: tok.pad_token = tok.eos_token
model = AutoModel.from_pretrained(sd, dtype=torch.float32).eval()
H = model.config.hidden_size
@torch.no_grad()
def feats(texts, bs=4):
    ids_all = [tok(t, add_special_tokens=False, truncation=True, max_length=MAXLEN)["input_ids"] + [tok.eos_token_id] for t in texts]
    Z = np.zeros((len(ids_all), H * len(LAYERS)), np.float32)
    order = np.argsort([len(x) for x in ids_all], kind="stable")
    t0 = time.time()
    for s in range(0, len(order), bs):
        idx = order[s:s + bs]
        enc = tok.pad({"input_ids": [ids_all[i] for i in idx]}, return_tensors="pt")
        hs = model(**enc, output_hidden_states=True, use_cache=False).hidden_states
        m = enc["attention_mask"].unsqueeze(-1).float(); m[:, 0] = 0
        den = m.sum(1).clamp(min=1)
        Z[idx] = np.concatenate([((hs[l].float() * m).sum(1) / den).numpy() for l in LAYERS], 1)
        if s % (bs * 50) == 0:
            log(f"  qwen {s+bs}/{len(order)} ({(s+bs)/max(time.time()-t0,1e-9):.1f} rows/s, eta {len(order)-(s+bs):.0f}/{(s+bs)/max(time.time()-t0,1e-9):.0f}s)")
    return Z
Z = feats(texts)
log("qwen features done", Z.shape, f"({time.time()-T0:.0f}s)")
np.savez_compressed("work/calib_local.npz", P_svc=P_svc, Z=Z, y=proxy["y"], clean=clean_mask.to_numpy())

# ---- head on CLEAN rows (LR C=0.05 balanced) ----
from sklearn.linear_model import LogisticRegression
cm = clean_mask.to_numpy()
clf = LogisticRegression(C=0.05, class_weight="balanced", max_iter=2000, solver="lbfgs")
clf.fit(Z[cm], proxy["y"][cm])
P_qw = softmax(clf.decision_function(Z))
log("clean head:", f"BA {ba(proxy['y'][cm], P_qw[cm]):.4f}")

# ---- weight sweep on clean rows ----
Lsvc = np.log(np.clip(P_svc, 1e-9, 1))
Lqw = np.log(np.clip(P_qw, 1e-9, 1))
yc = proxy["y"][cm]; Pc_svc = P_svc[cm]; Pc_qw = P_qw[cm]
best = (-1, None)
for w in (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0):
    P = softmax(1.0 * Lsvc[cm] + w * Lqw[cm])
    b = ba(yc, P); s = ba(yc, sinkhorn(P))
    log(f"w_qwen={w}: BA {b:.4f} / sink {s:.4f}")
    if s > best[0]: best = (s, w)
log("BEST w_qwen:", best, f"({time.time()-T0:.0f}s)")
