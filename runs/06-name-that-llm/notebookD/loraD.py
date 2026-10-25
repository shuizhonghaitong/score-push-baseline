# Name That LLM - Qwen3-4B two-stage LoRA trainer (run 06)
import gc, glob, json, os, re, time, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
T0 = time.time()
ROOT = "/kaggle/input"
torch.manual_seed(42); np.random.seed(42)

LABELS = ['gpt-5.5','gpt-5.6-luna','gpt-5.6-sol','gpt-5.6-terra','gpt-6-astra',
          'gemini-3.1-pro-preview','gemini-3.6-flash','gemini-3.7-flash','gemini-3.8-flash',
          'claude-fable-5','claude-opus-4-6','claude-opus-4-8','claude-opus-5','claude-sonnet-5',
          'qwen3.8-max','glm-5.3','glm-5.3-flash','kimi-k3']
K = len(LABELS)
LAB2I = {l: i for i, l in enumerate(LABELS)}

def find_file(name, root=ROOT):
    hits = sorted(glob.glob(os.path.join(root, "**", name), recursive=True))
    if not hits: raise FileNotFoundError(name)
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
train = pd.read_csv("/kaggle/working/ntli/train.csv")
pool = pd.read_parquet("/kaggle/working/ntli/pool.parquet")
prv = pd.read_csv("/kaggle/working/ntli/passages.csv")
prv["passes_funnel"] = prv["passes_funnel"].astype(str).str.lower().isin(["true", "1"])
prv_clean = prv[prv["passes_funnel"] & prv["model"].isin(LABELS)].copy()
print("inputs ready", f"({time.time() - T0:.0f}s)", flush=True)

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
print("model copied", f"({time.time() - T0:.0f}s)", flush=True)

from transformers import AutoModel, AutoTokenizer, get_linear_schedule_with_warmup
tok = AutoTokenizer.from_pretrained(model_dir)
if tok.pad_token is None: tok.pad_token = tok.eos_token
base = AutoModel.from_pretrained(model_dir, torch_dtype=torch.float16).cuda().train()
H = base.config.hidden_size
N_LAYERS = base.config.num_hidden_layers
for p in base.parameters(): p.requires_grad_(False)

R, ALPHA = 16, 32
SCALE = ALPHA / R
ATNS = ("q_proj", "k_proj", "v_proj", "o_proj")
lora_sd = {}
lora_params = []
class LoRA(nn.Module):
    def __init__(self, lin):
        super().__init__()
        self.lin = lin
        self.A = nn.Parameter(torch.empty(R, H, dtype=torch.float16, device="cuda"))
        self.B = nn.Parameter(torch.zeros(lin.weight.shape[0], R, dtype=torch.float16, device="cuda"))
        nn.init.kaiming_uniform_(self.A, a=5 ** 0.5)
        self.A.requires_grad_(True); self.B.requires_grad_(True)
    def forward(self, x):
        base_out = F.linear(x, self.lin.weight, self.lin.bias)
        lora_out = F.linear(F.linear(x, self.A), self.B) * SCALE
        return base_out + lora_out.to(x.dtype)
for li in range(N_LAYERS):
    for an in ATNS:
        lin = getattr(base.layers[li].self_attn, an)
        lora = LoRA(lin).cuda()
        setattr(base.layers[li].self_attn, an, lora)
        lora_sd[f"{li}.{an}"] = True
        lora_params += [lora.A, lora.B]
head = nn.Linear(H, K).cuda()
nn.init.zeros_(head.weight); nn.init.zeros_(head.bias)
all_params = lora_params + [head.weight, head.bias]
n_lora = sum(p.numel() for p in lora_params)
print(f"lora: {n_lora / 1e6:.1f}M trainable; head {H}->{K}", flush=True)

MAXLEN = 256
def make_ds(df, y):
    texts = df["text"].astype(str).tolist()
    enc = tok(texts, add_special_tokens=False, truncation=True, max_length=MAXLEN, padding="max_length", return_tensors="pt")
    return enc["input_ids"].cuda(), (enc["input_ids"] != tok.pad_token_id).cuda(), torch.tensor(y, dtype=torch.long, device="cuda")

fit_pool = pd.concat([pool[["text", "model"]], prv_clean[["text", "model"]]], ignore_index=True)
fit_pool = fit_pool[fit_pool["model"].isin(LABELS)].reset_index(drop=True)
y_pool = fit_pool["model"].map(LAB2I).to_numpy()
y_tr = train["model"].map(LAB2I).to_numpy()

def fit_stage(ids, mask, y, epochs, lr, bs=16, tag=""):
    opt = torch.optim.AdamW(all_params, lr=lr, weight_decay=0.0)
    total_steps = (len(y) // bs) * epochs
    sched = get_linear_schedule_with_warmup(opt, int(total_steps * 0.05), total_steps)
    ce = nn.CrossEntropyLoss()
    t0 = time.time()
    for ep in range(epochs):
        perm = torch.randperm(len(y))
        running, n = 0.0, 0
        for s in range(len(y) // bs):
            idx = perm[s * bs:(s + 1) * bs]
            b_ids, b_mask, b_y = ids[idx], mask[idx], y[idx]
            with torch.no_grad():
                pass
            out = base(input_ids=b_ids, attention_mask=b_mask, use_cache=False)
            h = torch.nan_to_num(out.last_hidden_state.float())
            m = b_mask.unsqueeze(-1).float()
            pooled = (h * m).sum(1) / m.sum(1).clamp(min=1)
            logits = head(pooled)
            loss = F.cross_entropy(logits, b_y)
            if not torch.isfinite(loss):
                print(f"  {tag} non-finite loss at ep{ep} s{s}; skipping step", flush=True)
                opt.zero_grad(set_to_none=True); continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(all_params, 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            running += loss.item(); n += bs
            if s % 100 == 0:
                acc = (logits.argmax(1) == b_y).float().mean().item()
                print(f"  {tag} ep{ep} s{s} loss {running / max(n, 1):.4f} acc {acc:.3f} ({time.time() - t0:.0f}s)", flush=True)
        print(f"  {tag} epoch {ep} done ({time.time() - t0:.0f}s)", flush=True)

ids_pool, mask_pool, y_pool_t = make_ds(fit_pool, y_pool)
ids_tr, mask_tr, y_tr_t = make_ds(train, y_tr)
fit_stage(ids_pool, mask_pool, y_pool_t, epochs=1, lr=1e-4, tag="stage1-pool")
del ids_pool, mask_pool, y_pool_t; gc.collect(); torch.cuda.empty_cache()
fit_stage(ids_tr, mask_tr, y_tr_t, epochs=4, lr=5e-5, tag="stage2-train")

# validation on train (in-sample, informational)
base.eval(); head.eval()
with torch.no_grad():
    out = base(input_ids=ids_tr, attention_mask=mask_tr, use_cache=False)
    h = torch.nan_to_num(out.last_hidden_state.float())
    m = mask_tr.unsqueeze(-1).float()
    pooled = (h * m).sum(1) / m.sum(1).clamp(min=1)
    acc = (head(pooled).argmax(1) == y_tr_t).float().mean().item()
print("train in-sample acc", acc, f"({time.time() - T0:.0f}s)", flush=True)

os.makedirs("/kaggle/working/out", exist_ok=True)
sd = {"head.weight": head.weight.cpu().float(), "head.bias": head.bias.cpu().float()}
for li in range(N_LAYERS):
    for an in ATNS:
        l = getattr(base.layers[li].self_attn, an)
        sd[f"lora/{li}.{an}/A"] = l.A.cpu().float()
        sd[f"lora/{li}.{an}/B"] = l.B.cpu().float()
torch.save(sd, "/kaggle/working/out/lora4b_head.pt")
json.dump({"labels": LABELS, "r": R, "alpha": ALPHA, "scale": SCALE, "attns": list(ATNS),
           "n_layers": N_LAYERS, "hidden": H, "max_len": MAXLEN, "base": "Qwen/Qwen3-4B"},
          open("/kaggle/working/out/meta.json", "w"))
print("SAVED", os.path.getsize("/kaggle/working/out/lora4b_head.pt") // 1e6, "MB", f"({time.time() - T0:.0f}s)", flush=True)
