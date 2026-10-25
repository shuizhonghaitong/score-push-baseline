import gc, glob, json, os, re, time
import numpy as np
import pandas as pd
import torch
from torch import nn

T0 = time.time()
INPUT_DIR = "/kaggle/input"

# Blend weights, fit on out-of-fold predictions (log probabilities are multiplied by these and summed).
W_SVC = 1.021
LORAS = {  # checkpoint folder: blend weight, centred-twin weight, batch size, base model to look for
    "lora8b_v1":  dict(w=0.226, twin_w=0.079, bs=16, base=dict(model_type="qwen3", hidden_size=4096, num_hidden_layers=36)),
    "lora14b_v1": dict(w=0.429, twin_w=0.150, bs=8,  base=dict(model_type="qwen3", hidden_size=5120, num_hidden_layers=40)),
}
TWIN = dict(thresh=0.85, min_size=2, max_size=18)   # topic clusters that get the centred twin

# Column order of the checkpoints' heads (checked against each checkpoint's meta.json below).
LABELS = ['gpt-5.5', 'gpt-5.6-luna', 'gpt-5.6-sol', 'gpt-5.6-terra', 'gpt-6-astra',
          'gemini-3.1-pro-preview', 'gemini-3.6-flash', 'gemini-3.7-flash', 'gemini-3.8-flash',
          'claude-opus-4-6', 'claude-opus-4-8', 'claude-opus-5', 'claude-sonnet-5', 'claude-fable-5',
          'qwen3.8-max', 'glm-5.3', 'glm-5.3-flash', 'kimi-k3']
K = len(LABELS)


def find(pattern):
    hits = sorted(glob.glob(f"{INPUT_DIR}/**/{pattern}", recursive=True))
    if not hits:
        raise FileNotFoundError(pattern)
    return hits[0]


def softmax(x):
    e = np.exp(x - x.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


_competition_dir = next(p for p in ["/kaggle/input/competitions/name-that-llm", "/kaggle/input/name-that-llm"] if os.path.exists(p + "/train.csv"))
train = pd.read_csv(_competition_dir + "/train.csv")
test = pd.read_csv(_competition_dir + "/test.csv")
test["text"] = test["text"].fillna("").astype(str)
y = train["model"].map({l: i for i, l in enumerate(LABELS)}).values
LOGIT = np.zeros((len(test), K))   # every member adds w * log p here
print("train", train.shape, "test", test.shape, flush=True)


from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

FUNCTION_WORDS = set("a an the and or but if while although because as so than that which who whom whose this these those it its they them their theirs he him his she her hers we us our ours you your yours i me my mine in on at to from with without for of by over under between through into about is are was were be been being have has had do does did can could will would shall should may might must not no nor all any some each every both either neither other another such only also just very too more most less least here there when where why how what whether then now still yet even much many few little".split())
VIEWS = [("char", 3, 5, 250000, 1.0), ("word", 1, 3, 180000, 0.7), ("shape", 2, 6, 60000, 0.5), ("function", 1, 3, 50000, 0.5)]


def shape_text(t):
    return re.sub(r"[^\W\d_]+", lambda m: "X" if m[0].isupper() else "x", t)


def function_text(t):
    return " ".join(w.lower() if w.lower() in FUNCTION_WORDS or not w.isalnum() else "CONTENT"
                    for w in re.findall(r"\w+|[^\w\s]", t))


pool = pd.read_parquet(find("pool_v1.parquet"))
fit_texts = train.text.astype(str).tolist() + pool.text.tolist()
fit_y = np.r_[y, pool.model.map({l: i for i, l in enumerate(LABELS)}).values]
A, B = [], []
for view, lo, hi, max_features, weight in VIEWS:
    prep = shape_text if view == "shape" else function_text if view == "function" else (lambda s: s)
    analyzer = "char" if view == "shape" else "word" if view == "function" else view
    opts = dict(analyzer=analyzer, ngram_range=(lo, hi), lowercase=False, sublinear_tf=True, min_df=2,
                max_features=max_features, dtype=np.float32)
    if analyzer == "word":
        opts["token_pattern"] = r"(?u)\b\w+\b|[^\w\s]"
    vec = TfidfVectorizer(**opts)
    A.append(vec.fit_transform([prep(t) for t in fit_texts]) * weight)
    B.append(vec.transform([prep(t) for t in test.text]) * weight)
svc = LinearSVC(C=0.5, random_state=42, max_iter=5000, class_weight="balanced").fit(hstack(A, format="csr"), fit_y)
P_svc = softmax(svc.decision_function(hstack(B, format="csr")))
LOGIT += W_SVC * np.log(np.clip(P_svc, 1e-9, 1))
del A, B, svc, pool
gc.collect()
print(f"n-gram model done ({time.time() - T0:.0f}s)", flush=True)


from safetensors.torch import load_file
from transformers import AutoConfig, AutoModel, AutoTokenizer


def find_base(spec):
    for cfg in sorted(glob.glob(f"{INPUT_DIR}/**/config.json", recursive=True)):
        try:
            c = json.load(open(cfg))
        except Exception:
            continue
        if all(c.get(k) == v for k, v in spec.items()):
            return os.path.dirname(cfg)
    raise FileNotFoundError(f"attach the base model {spec}")


def load_trunk(base_dir, keep, dtype):
    """First `keep` decoder layers, split over both T4s when it does not fit on one."""
    mc = AutoConfig.from_pretrained(base_dir)
    mc.num_hidden_layers, mc.use_cache = keep, False
    if getattr(mc, "layer_types", None) is not None:
        mc.layer_types = mc.layer_types[:keep]
    hd = getattr(mc, "head_dim", mc.hidden_size // mc.num_attention_heads)
    gib = 2 * (mc.vocab_size * mc.hidden_size + keep * (2 * mc.hidden_size * mc.num_attention_heads * hd
               + 2 * mc.hidden_size * mc.num_key_value_heads * hd + 3 * mc.hidden_size * mc.intermediate_size)) / 2 ** 30
    if gib > 13 and torch.cuda.device_count() > 1:
        mem = {i: "13GiB" for i in range(torch.cuda.device_count())}
        model = AutoModel.from_pretrained(base_dir, config=mc, dtype=dtype, device_map="auto", max_memory=mem).eval()
    else:
        model = AutoModel.from_pretrained(base_dir, config=mc, dtype=dtype).to("cuda:0").eval()
    model.norm = nn.Identity()
    return model


@torch.no_grad()
def merge_lora(model, adapter_dir):
    acfg = json.load(open(os.path.join(adapter_dir, "adapter_config.json")))
    scale = acfg["lora_alpha"] / acfg["r"]
    sd = load_file(os.path.join(adapter_dir, "adapter_model.safetensors"))
    modules = dict(model.named_modules())
    for k, A in sd.items():
        if ".lora_A." in k:
            name = k.split(".lora_A.")[0]
            lin = modules[name[name.index("layers."):]]
            Bm = sd[k.replace(".lora_A.", ".lora_B.")]
            lin.weight.add_(((Bm.float().to(lin.weight.device) @ A.float().to(lin.weight.device)) * scale).to(lin.weight.dtype))


@torch.no_grad()
def lora_features(ckpt_dir, base_dir, texts, bs):
    """Pooled features Z (n, 8 * hidden) and the checkpoint's heads."""
    meta = json.load(open(os.path.join(ckpt_dir, "meta.json")))
    assert meta["labels"] == LABELS, "label order differs from the checkpoint's heads"
    cfg = meta["cfg"]
    dtype = torch.bfloat16 if torch.cuda.get_device_capability(0)[0] >= 8 else torch.float16
    tok = AutoTokenizer.from_pretrained(base_dir)
    tok.padding_side = "right"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = load_trunk(base_dir, cfg["keep"], dtype)
    merge_lora(model, os.path.join(ckpt_dir, "adapter"))
    heads = load_file(os.path.join(ckpt_dir, "heads.safetensors"))
    layers, H = cfg["layers"], model.config.hidden_size
    norms = nn.ModuleList([nn.LayerNorm(H) for _ in layers]).to("cuda:0").float()
    norms.load_state_dict({k[len("norms."):]: v for k, v in heads.items() if k.startswith("norms.")})
    ids = tok(list(texts), add_special_tokens=False, truncation=True, max_length=cfg["max_len"])["input_ids"]
    ids = [x + [tok.eos_token_id] for x in ids]
    order = np.argsort([len(x) for x in ids], kind="stable")   # similar lengths together = less padding
    Z = np.zeros((len(ids), H * len(layers)), np.float32)
    for s in range(0, len(order), bs):
        idx = order[s:s + bs]
        enc = tok.pad({"input_ids": [ids[i] for i in idx]}, return_tensors="pt").to(model.embed_tokens.weight.device)
        hs = model(**enc, output_hidden_states=True, use_cache=False).hidden_states
        if not all(torch.isfinite(hs[l]).all() for l in layers):   # fp16 overflow: redo this batch in bf16
            model.to(torch.bfloat16)
            hs = model(**enc, output_hidden_states=True, use_cache=False).hidden_states
            model.to(dtype)
        m = enc["attention_mask"].unsqueeze(-1).float().to("cuda:0")
        m[:, 0] = 0   # the first token is an attention sink; leave it out of the mean
        den = m.sum(1).clamp(min=1)
        Z[idx] = torch.cat([nm((hs[l].to("cuda:0").float() * m).sum(1) / den) for l, nm in zip(layers, norms)], -1).cpu().numpy()
    del model
    torch.cuda.empty_cache()
    return Z, {k: v.float().numpy() for k, v in heads.items() if not k.startswith("norms.")}


from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import squareform

Xt = TfidfVectorizer(stop_words="english", sublinear_tf=True).fit_transform(test.text.tolist())
D = np.clip(1 - (Xt @ Xt.T).toarray(), 0, None)
np.fill_diagonal(D, 0)
groups = fcluster(linkage(squareform(D, checks=False), method="average"), TWIN["thresh"], criterion="distance") \
    if len(test) >= 2 else np.ones(len(test), dtype=int)
size = np.bincount(groups)[groups]
twin_rows = (size >= TWIN["min_size"]) & (size <= TWIN["max_size"])
del D
print(f"topic clusters: {int((size == 1).sum())} singletons, {int(twin_rows.sum())} passages in clusters of 2-18", flush=True)

# Gaussian shared-prompt core, credited to prvsiyan.
import itertools
K = 18
DEFAULT = {"red": "wpca", "d": 256, "se_shrink": "lw", "sp_shrink": "lw", "lam_floor": 1e-4, "n_exact": 3, "inj": True,
           "iters": 5, "repel": True, "sp_scale": 1.0}

# =================================================================== primitives ====
def _lse(A, axis=None, keepdims=False):
    m = np.max(A, axis=axis, keepdims=True); m = np.where(np.isfinite(m), m, 0.0)
    out = np.log(np.sum(np.exp(A - m), axis=axis, keepdims=True)) + m
    if keepdims: return out
    return np.squeeze(out, axis=axis) if axis is not None else float(out.item())

def log_softmax(L): return L - _lse(L, axis=1, keepdims=True)

def _lw_shrinkage(X):
    """Ledoit-Wolf shrinkage coefficient for rows X that are already centred (sklearn formula, assume_centered)."""
    from sklearn.covariance import ledoit_wolf_shrinkage
    return float(ledoit_wolf_shrinkage(X, assume_centered=True))

def _shrunk(S, a):
    d = S.shape[0]; return (1 - a) * S + a * (np.trace(S) / d) * np.eye(d)

def _two_way(Z, y, pinv, G, iters=4):
    """additive fit Z = mu[y] + P[prompt] + E by alternating means (balanced design -> converges in 1-2 rounds)."""
    P = np.zeros((G, Z.shape[1])); cnt_p = np.bincount(pinv, minlength=G).astype(float)
    cls = np.unique(y); mu = np.zeros((K, Z.shape[1]))
    for _ in range(iters):
        R = Z - P[pinv]
        for c in cls: mu[c] = R[y == c].mean(0)
        R = Z - mu[y]; P = np.zeros_like(P); np.add.at(P, pinv, R); P /= cnt_p[:, None]
    return mu, P, cnt_p

def fit(X, y, prompts, cfg=None):
    """fit the shared-prompt Gaussian on training rows. X (N, D) raw features, y (N,) labels in 0..17, prompts (N,) group ids.
    Returns a params dict (numpy arrays + floats); temperatures default to 1 (set by the caller's cross-fit)."""
    c = dict(DEFAULT); c.update(cfg or {})
    X = np.asarray(X, dtype=np.float64); y = np.asarray(y); N, D = X.shape
    pu, pinv = np.unique(np.asarray(prompts), return_inverse=True); G = len(pu)
    mx = X.mean(0); sx = X.std(0); sx = np.where(sx > 1e-8, sx, 1.0); Xs = (X - mx) / sx
    if c["red"] == "none":
        V = np.eye(D)
    else:
        if c["red"] == "wpca":
            Pm = np.zeros((G, D)); np.add.at(Pm, pinv, Xs); Pm /= np.bincount(pinv)[:, None]; R = Xs - Pm[pinv]
        elif c["red"] == "pca": R = Xs
        else: raise ValueError(c["red"])
        d = int(min(c["d"], N - G - 1, D))
        # economy SVD through the (N x N) Gram matrix (N << D for the real views)
        if N < D:
            w, U = np.linalg.eigh(R @ R.T); o = np.argsort(w)[::-1][:d]; w, U = np.clip(w[o], 1e-12, None), U[:, o]
            V = (R.T @ U) / np.sqrt(w)
        else:
            _, _, Vt = np.linalg.svd(R, full_matrices=False); V = Vt[:d].T
    Z = Xs @ V; d = Z.shape[1]
    mu, P, ng = _two_way(Z, y, pinv, G)
    E = Z - mu[y] - P[pinv]; dof = max(N - G - len(np.unique(y)) + 1, 1)
    Se_raw = E.T @ E / dof
    ae = _lw_shrinkage(E) if c["se_shrink"] == "lw" else float(c["se_shrink"]); Se = _shrunk(Se_raw, ae)
    Pc = P - P.mean(0); Sp_raw = Pc.T @ Pc / max(G - 1, 1) - np.mean(1.0 / ng) * Se_raw
    ap = _lw_shrinkage(Pc) if c["sp_shrink"] == "lw" else float(c["sp_shrink"])
    tr = max(np.trace(Sp_raw), 1e-8 * np.trace(Se)); Sp = (1 - ap) * Sp_raw + ap * (tr / d) * np.eye(d); Sp *= c["sp_scale"]
    # joint diagonalisation: A Se A' = I, A Sp A' = diag(lam)
    L = np.linalg.cholesky(Se); Li = np.linalg.inv(L); M = Li @ Sp @ Li.T; M = 0.5 * (M + M.T)
    lam, Q = np.linalg.eigh(M); lam = np.clip(lam, c["lam_floor"], None); A = Q.T @ Li
    W = V @ A.T
    return {"mx": mx, "sx": sx, "W": W, "m": mu @ A.T, "lam": lam, "T1": 1.0, "Tg": 1.0,
            "info": {"d": int(d), "a_e": round(ae, 4), "a_p": round(ap, 4), "lam_pct": np.round(np.percentile(lam, [5, 50, 95, 100]), 3).tolist()}, "cfg": c}

def transform(X, prm): return ((np.asarray(X, dtype=np.float64) - prm["mx"]) / prm["sx"]) @ prm["W"]

_MASK = {}
def _inj_mask(n, C):
    if (n, C) not in _MASK:
        grids = np.meshgrid(*[np.arange(C)] * n, indexing="ij"); bad = np.zeros([C] * n, bool)
        for i, j in itertools.combinations(range(n), 2): bad |= grids[i] == grids[j]
        _MASK[(n, C)] = bad
    return _MASK[(n, C)]

def singleton_ll(U, m, lam):
    """(n, C) log N(u; m_c, I + diag lam) up to a constant (p integrated out)."""
    R = U[:, None, :] - m[None]; return -0.5 * (R ** 2 / (1 + lam)).sum(-1)

def joint_terms(U, m, lam):
    """a (n, C) and B (n, n, C, C) of the exact joint log-likelihood of one group (constant in c dropped)."""
    n = len(U); gam = lam / (1 + n * lam); R = U[:, None, :] - m[None]
    a = -0.5 * (R ** 2 * (1 - gam)).sum(-1); G = R * np.sqrt(gam); B = np.einsum("icd,jed->ijce", G, G)
    return a, B

def joint_table(a, B):
    """full tuple table Q[c_1, ..., c_n] = sum_i a_i + sum_{i<j} B_ij."""
    n, C = a.shape; Q = np.zeros([C] * n)
    for i in range(n):
        sh = [1] * n; sh[i] = C; Q = Q + a[i].reshape(sh)
    for i, j in itertools.combinations(range(n), 2):
        sh = [1] * n; sh[i] = C; sh[j] = C; Q = Q + B[i, j].reshape(sh)
    return Q

def exact_marginals(Q, T=1.0, inj=True):
    """per-passage log marginal posteriors (n, C) from a tuple table Q (unnormalised log joint), tempered by 1/T."""
    n = Q.ndim; C = Q.shape[0]; Qt = Q / T
    if inj and n >= 2: Qt = np.where(_inj_mask(n, C), -np.inf, Qt)
    lz = _lse(Qt); out = np.empty((n, C))
    for i in range(n):
        ax = tuple(k for k in range(n) if k != i); out[i] = (_lse(Qt, axis=ax) if ax else Qt) - lz
    return out

def em_marginals(U, m, lam, T=1.0, iters=5, repel=False, q0=None):
    """cavity EM for one group (any n >= 2): returns (n, C) log q."""
    n = len(U); v = lam / (1 + (n - 1) * lam)
    L = singleton_ll(U, m, lam) / T if q0 is None else np.log(np.clip(q0, 1e-300, None))
    q = np.exp(log_softmax(L))
    for _ in range(iters):
        s = U - q @ m; S = s.sum(0); Pm = (S[None] - s) * v                  # cavity prompt-effect means (n, d)
        R = U[:, None, :] - Pm[:, None, :] - m[None]; L = -0.5 * (R ** 2 / (1 + v)).sum(-1) / T
        if repel:   # mean-field at-most-once: discourage classes the other members already hold
            lo = np.log(np.clip(1 - q, 1e-6, None)); L = L + (lo.sum(0)[None] - lo)
        q = np.exp(log_softmax(L))
    return log_softmax(L)

LAM_DIFF = 1e6
def lam_g(prm, cfg=None):
    """lam used for groups: the fitted one, or ~infinity in the paired-difference ablation (cfg 'diff'): then the joint likelihood
    only sees within-group differences (x_i - x_j around mu_a - mu_b with covariance 2 S_e; triples: centred observations vs
    centred prototypes) and the prompt term cancels exactly."""
    c = dict(prm.get("cfg", DEFAULT)); c.update(cfg or {})
    return np.full_like(prm["lam"], LAM_DIFF) if c.get("diff") else prm["lam"]

def group_logpost(U, prm, T1=None, Tg=None, cfg=None):
    c = dict(prm.get("cfg", DEFAULT)); c.update(cfg or {}); n = len(U); lg = lam_g(prm, cfg)
    T1 = prm["T1"] if T1 is None else T1; Tg = prm["Tg"] if Tg is None else Tg
    if n == 1: return log_softmax(singleton_ll(U, prm["m"], prm["lam"]) / T1)
    if n <= c["n_exact"]:
        a, B = joint_terms(U, prm["m"], lg); return exact_marginals(joint_table(a, B), Tg, c["inj"])
    return em_marginals(U, prm["m"], lg, Tg, c["iters"], c["repel"])

def scores(X_pool, lab, prm, T1=None, Tg=None, cfg=None, U=None):
    """(n, 18) log-probabilities for every pool row; lab = detected group labels of the pool rows (singletons allowed)."""
    U = transform(X_pool, prm) if U is None else U; lab = np.asarray(lab); n = len(U); out = np.empty((n, prm["m"].shape[0]))
    sizes = np.bincount(lab); s1 = sizes[lab] == 1
    T1 = prm["T1"] if T1 is None else T1
    if s1.any(): out[s1] = log_softmax(singleton_ll(U[s1], prm["m"], prm["lam"]) / T1)
    for g in np.where(sizes >= 2)[0]:
        ii = np.where(lab == g)[0]; out[ii] = group_logpost(U[ii], prm, T1, Tg, cfg)
    return out




# ---- LoRA members (8B required, 14B optional; skip if attached base can't fit) ----
for ckpt_name, mc in LORAS.items():
    try:
        ckpt = os.path.dirname(find(f"{ckpt_name}/meta.json"))
    except FileNotFoundError:
        print(f"{ckpt_name} not attached - skipping", flush=True)
        continue
    if mc["base"]["hidden_size"] >= 5120 and torch.cuda.device_count() < 2:
        print(f"{ckpt_name} needs 2xT4 - skipping on this machine", flush=True)
        continue
    try:
        if ckpt_name == 'lora8b_v1':
            _all_texts = test.text.tolist() + train.text.astype(str).tolist()
            _all_Z, heads = lora_features(ckpt, find_base(mc['base']), _all_texts, mc['bs'])
            Z, _train_Z = _all_Z[:len(test)], _all_Z[len(test):]
            _prm = fit(_train_Z, y, train['prompt_id'].values,
                       dict(red='wpca', d=128, se_shrink='lw', sp_shrink='lw', lam_floor=1e-4,
                            n_exact=3, inj=True, iters=5, repel=True, sp_scale=1.0))
            _prm['T1'] = _prm['Tg'] = 3.0
            if twin_rows.any():
                _lp = scores(Z[twin_rows], np.unique(groups[twin_rows], return_inverse=True)[1], _prm)
                assert np.isfinite(_lp).all()
                LOGIT[twin_rows] += 0.10 * _lp
                print('NEW_LORA_GAUSSIAN rows', int(twin_rows.sum()), 'd128 T3 weight.10', flush=True)
            else:
                print('NEW_LORA_GAUSSIAN fitted; no eligible public-test groups', flush=True)
            del _all_Z, _train_Z, _prm, _all_texts
        else:
            Z, heads = lora_features(ckpt, find_base(mc['base']), test.text.tolist(), mc['bs'])
        LOGIT += mc["w"] * np.log(np.clip(softmax(Z @ heads["plain.weight"].T + heads["plain.bias"]), 1e-9, 1))
        if twin_rows.any():
            Zc = Z.copy()
            for g in np.unique(groups[twin_rows]):
                Zc[groups == g] -= Zc[groups == g].mean(0)
            centred = Zc[twin_rows] @ heads["centered.weight"].T + heads["centered.bias"]
            LOGIT[twin_rows] += mc["twin_w"] * np.log(np.clip(softmax(centred), 1e-9, 1))
        del Z, heads
        gc.collect()
        torch.cuda.empty_cache()
        print(f"{ckpt_name} done ({time.time() - T0:.0f}s)", flush=True)
    except Exception:
        import traceback; traceback.print_exc()
        print(f"{ckpt_name} FAILED - skipping", flush=True)
        gc.collect(); torch.cuda.empty_cache()
        continue

# ---- blend + guards ----
P = softmax(LOGIT)

SPACED_ALLOWED = {'claude-opus-4-6', 'claude-opus-5', 'glm-5.3', 'glm-5.3-flash'}
forB = np.array([l not in SPACED_ALLOWED for l in LABELS])
for i, t in enumerate(test["text"].tolist()):
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
np.savez_compressed("fork_probs.npz", logit=LOGIT)
print(sub["model"].value_counts().to_string())
print(f"done in {time.time() - T0:.0f}s")
