# Local CPU feature extraction for the 300-prompt proxy (validates featA code path; optional speed check)
import os, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(__file__))
from common import *
os.environ["HF_HOME"] = f"{ROOT}/hf_cache"
os.environ["HF_HUB_DISABLE_XET"] = "1"
import torch
torch.set_num_threads(8)
T0 = time.time()
prv = load_data()[2]
prv["passes_funnel"] = prv["passes_funnel"].astype(str).str.lower().isin(["true", "1"])
proxy = prv[(prv["passes_funnel"]) & (prv["model"].isin(LABELS)) & (prv["split"] == "new")].reset_index(drop=True)
proxy["y"] = proxy["model"].map(LAB2I)
print("proxy rows:", len(proxy))
from transformers import AutoModel, AutoTokenizer
model_dir = f"{ROOT}/hf_cache/hub/models--Qwen--Qwen3-4B/snapshots"
import glob
sd = sorted(glob.glob(model_dir + "/*"))[0]
print("model:", sd)
LAYERS = (1, 2, 3, 5, 8)
MAXLEN = 320
tok = AutoTokenizer.from_pretrained(sd)
if tok.pad_token is None: tok.pad_token = tok.eos_token
model = AutoModel.from_pretrained(sd, dtype=torch.float32).eval()
H = model.config.hidden_size
@torch.no_grad()
def feats(texts, bs=8):
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
        parts = [((hs[l].float() * m).sum(1) / den).numpy() for l in LAYERS]
        Z[idx] = np.concatenate(parts, 1)
        if s % (bs * 25) == 0:
            print(f"  {s + bs}/{len(order)} ({(s + bs) / max(time.time() - t0, 1e-9):.1f} rows/s)", flush=True)
    return Z
Z = feats(proxy["text"].astype(str).tolist())
np.savez_compressed("work/proxy_z_local.npz", Z=Z, y=proxy["y"].to_numpy())
print("DONE", Z.shape, f"({time.time() - T0:.0f}s)", flush=True)
