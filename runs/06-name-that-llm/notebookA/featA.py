# Name That LLM - Qwen3-4B early-layer feature study (notebook A)
# Train plain + twin logistic heads on prompt-grouped public data; evaluate on
# prvsiyan 'new' holdout (300 unseen prompts); dump heads + proxy probs for local blend calibration.
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

def ba(y, P):
    pred = P.argmax(1); K = 18
    return float(np.mean([(pred[y == k] == k).mean() for k in range(K)]))

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
print("competition inputs copied", flush=True)
pool_src = find_file("ntl_pool_v4.parquet"); prv_src = find_file("passages.csv")
for src, dst in [(pool_src, "/kaggle/working/ntli/pool.parquet"), (prv_src, "/kaggle/working/ntli/passages.csv")]:
    if not os.path.exists(dst):
        with open(src, "rb") as fsrc, open(dst, "wb") as fdst:
            while True:
                buf = fsrc.read(1 << 24)
                if not buf: break
                fdst.write(buf)
print("pools copied", f"({time.time() - T0:.0f}s)", flush=True)
train = pd.read_csv("/kaggle/working/ntli/train.csv")
test = pd.read_csv("/kaggle/working/ntli/test.csv")
test["text"] = test["text"].fillna("").astype(str)
y_train = train["model"].map(LAB2I).to_numpy()

pool = pd.read_parquet("/kaggle/working/ntli/pool.parquet")
prv = pd.read_csv("/kaggle/working/ntli/passages.csv")
prv["passes_funnel"] = prv["passes_funnel"].astype(str).str.lower().isin(["true", "1"])
prv_clean = prv[prv["passes_funnel"] & prv["model"].isin(LABELS)].copy()
prv_clean["y"] = prv_clean["model"].map(LAB2I)
proxy = prv_clean[prv_clean["split"] == "new"].reset_index(drop=True)          # holdout: unseen prompts
prv_fit = prv_clean[prv_clean["split"] != "new"].reset_index(drop=True)

# fit corpus = train + pool (all) + prv non-new clean  (proxy rows excluded => clean eval)
fit = pd.concat([train[["text", "model"]], pool[["text", "model"]], prv_fit[["text", "model"]]], ignore_index=True)
fit = fit[fit["model"].isin(LABELS)].reset_index(drop=True)
fit["y"] = fit["model"].map(LAB2I)
# prompt-group ids for the twin head: pool `group` column; train prompt_id; prv prompt_id
fit["grp"] = fit.index.astype(str)
gmap = {}
for i, g in enumerate(pool["group"].astype(str)):
    gmap[g] = f"P{i}"
n_tr, n_po = len(train), len(pool)
fit = fit.assign(grp=["tr" + str(i) for i in range(n_tr)]
                 + [f"pool_{g}" for g in pool["group"].astype(str)]
                 + ["prv" + g for g in prv_fit["prompt_id"].astype(str)])
print("fit", fit.shape, "proxy", proxy.shape, "test", test.shape, flush=True)

# ---------- Qwen3-4B ----------
os.environ.setdefault("HF_ENDPOINT", "https://huggingface.co")
from huggingface_hub import snapshot_download
from transformers import AutoModel, AutoTokenizer
model_dir = snapshot_download("Qwen/Qwen3-4B", max_workers=4)
print("model at", model_dir, f"({time.time() - T0:.0f}s)", flush=True)

LAYERS = (1, 2, 3, 5, 8)
tok = AutoTokenizer.from_pretrained(model_dir)
tok.padding_side = "right"
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
model = AutoModel.from_pretrained(model_dir, torch_dtype=torch.float16).to("cuda").eval()
H = model.config.hidden_size
MAXLEN = 320

@torch.no_grad()
def feats(texts, tag=""):
    ids_all = [tok(t, add_special_tokens=False, truncation=True, max_length=MAXLEN)["input_ids"] + [tok.eos_token_id] for t in texts]
    Z = np.zeros((len(ids_all), H * len(LAYERS)), np.float32)
    order = np.argsort([len(x) for x in ids_all], kind="stable")
    bs = 16
    t0 = time.time()
    for s in range(0, len(order), bs):
        idx = order[s:s + bs]
        enc = tok.pad({"input_ids": [ids_all[i] for i in idx]}, return_tensors="pt").to("cuda")
        hs = model(**enc, output_hidden_states=True, use_cache=False).hidden_states
        m = enc["attention_mask"].unsqueeze(-1).float().cuda()
        m[:, 0] = 0
        den = m.sum(1).clamp(min=1)
        parts = []
        for l in LAYERS:
            h = hs[l].float().cuda() * m
            parts.append((h.sum(1) / den).half().cpu().numpy().astype(np.float32))
        Z[idx] = np.concatenate(parts, 1)
        if s % (bs * 50) == 0:
            rate = (s + bs) / max(time.time() - t0, 1e-9)
            print(f"  {tag} {s + bs}/{len(order)} ({rate:.0f} rows/s)", flush=True)
        del hs, enc
        torch.cuda.empty_cache()
    return Z

Zfit = feats(fit["text"].astype(str).tolist(), "fit")
Zproxy = feats(proxy["text"].astype(str).tolist(), "proxy")
Ztest = feats(test["text"].astype(str).tolist(), "test")
np.savez_compressed("fit_features.npz", Z=Zfit)
np.savez_compressed("proxy_features.npz", Z=Zproxy)
print("features done", Zfit.shape, f"({time.time() - T0:.0f}s)", flush=True)
del model
gc.collect(); torch.cuda.empty_cache()

# ---------- heads ----------
from sklearn.linear_model import LogisticRegression
yfit = fit["y"].to_numpy()
# plain head
plain = LogisticRegression(C=0.05, max_iter=2000, class_weight="balanced", solver="lbfgs").fit(Zfit, yfit)
Pp = plain.predict_proba(Zproxy)
print("plain head proxy BA", ba(proxy["y"].to_numpy(), Pp), f"({time.time() - T0:.0f}s)", flush=True)
# twin head: centered features of pool groups with 2..18 members
grp_arr = fit["grp"].to_numpy()
import pandas as _pd
cnt = _pd.Series(grp_arr).value_counts()
good = set(g for g, c in cnt.items() if 2 <= c <= 18)
twin_mask = np.isin(grp_arr, np.fromiter(good, dtype=grp_arr.dtype))
Zt = Zfit[twin_mask].copy()
centers = {}
for g in np.unique(grp_arr[twin_mask]):
    rows = twin_mask & (grp_arr == g)
    centers[g] = Zt[rows].mean(0)
Zt -= np.array([centers[g] for g in grp_arr[twin_mask]])
twin = LogisticRegression(C=0.05, max_iter=2000, class_weight="balanced", solver="lbfgs").fit(Zt, yfit[twin_mask])
# proxy has true prompt groups: center within each proxy prompt
Zt2 = Zproxy.copy()
for g in np.unique(proxy["prompt_id"].astype(str)):
    rows = (proxy["prompt_id"].astype(str) == g).to_numpy()
    Zt2[rows] -= Zt2[rows].mean(0, keepdims=True)
Pt = twin.predict_proba(Zt2)
print("twin head proxy BA", ba(proxy["y"].to_numpy(), Pt), f"({time.time() - T0:.0f}s)", flush=True)
# mix
for w in (0.0, 0.5, 1.0, 2.0):
    L = np.log(np.clip(Pp, 1e-9, 1)) + w * np.log(np.clip(Pt, 1e-9, 1))
    print(f"  plain+{w}*twin BA", ba(proxy["y"].to_numpy(), softmax(L)), flush=True)

np.savez_compressed("heads.npz", plain_coef=plain.coef_, plain_intercept=plain.intercept_,
                    twin_coef=twin.coef_, twin_intercept=twin.intercept_)
np.savez_compressed("proxy_probs.npz", plain=Pp, twin=Pt)
print("ALL DONE", f"({time.time() - T0:.0f}s)", flush=True)
