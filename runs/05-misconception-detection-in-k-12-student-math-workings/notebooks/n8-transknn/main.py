import os, time, json, warnings, gc
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms

print("N8_TAG trans-knn-v1", flush=True)
print("torch:", torch.__version__, "cuda:", torch.cuda.is_available(), flush=True)
device = "cuda" if torch.cuda.is_available() else "cpu"

def upload_dataset_files(files, ds_slug):
    import shutil, subprocess, sys
    os.environ["KAGGLE_API_TOKEN"] = os.environ.get("KAGGLE_API_V1_TOKEN", "")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-U", "kaggle", "kagglehub"],
                   capture_output=True, text=True, timeout=600)
    up = "/kaggle/upload_ds"
    shutil.rmtree(up, ignore_errors=True)
    os.makedirs(up, exist_ok=True)
    for f in files:
        shutil.copy(f, os.path.join(up, os.path.basename(f)))
    json.dump({"title": ds_slug, "id": f"lilac1996/{ds_slug}", "licenses": [{"name": "CC0-1.0"}]},
              open(os.path.join(up, "dataset-metadata.json"), "w"))
    base = [sys.executable, "-m", "kaggle", "datasets"]
    plans = [
        (["create", "-p", up, "-q"], "create"),
        (["version", "create", "-p", up, "-m", "n8 " + ds_slug], "version-create"),
        (["create", "-p", up, "-q", "--version", "2"], "create-v2"),
    ]
    for cmd_tail, tag in plans:
        r = subprocess.run(base + cmd_tail, capture_output=True, text=True, timeout=1800)
        msg = (r.stdout + r.stderr).strip()
        print("DATASET", ds_slug, tag, "rc=", r.returncode, msg[:300], flush=True)
        if r.returncode == 0:
            return True
    return False

DATA = None
for c in ["/kaggle/input/competitions/misconception-detection-in-k-12-student-math-workings",
          "/kaggle/input/misconception-detection-in-k-12-student-math-workings", "/kaggle/input"]:
    if os.path.isdir(c) and os.path.exists(os.path.join(c, "train.csv")):
        DATA = c; break
if DATA is None:
    for root, dirs, files in os.walk("/kaggle/input"):
        if "train.csv" in files:
            DATA = root; break
assert DATA, "no data found"
print("DATA =", DATA, flush=True)
IMG = os.path.join(DATA, "competition_data", "images") if os.path.isdir(os.path.join(DATA, "competition_data")) else os.path.join(DATA, "images")

tr = pd.read_csv(os.path.join(DATA, "train.csv"))
te = pd.read_csv(os.path.join(DATA, "test.csv"))
ul = pd.read_csv(os.path.join(DATA, "unlabelled.csv"))
codes = sorted(tr.root_code.unique())
if "NONE" not in codes: codes.append("NONE")
codes.sort()
ci = {c: i for i, c in enumerate(codes)}
C = len(codes)
y = tr.root_code.map(ci).to_numpy()
print("train", tr.shape, "test", te.shape, "unlabelled", ul.shape, "codes", C, flush=True)

from transformers import CLIPModel, CLIPProcessor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
tfm = transforms.Compose([
    transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073), std=(0.26862954, 0.26130258, 0.27577711)),
])

def embed_df(df, split, bs=64):
    out = np.zeros((len(df), 512), dtype=np.float32)
    t0 = time.time()
    for s in range(0, len(df), bs):
        e = min(s + bs, len(df))
        imgs = []
        for i in range(s, e):
            p = os.path.join(IMG, split, df.iloc[i].image_file)
            try:
                im = Image.open(p).convert("RGB")
            except Exception:
                im = Image.new("RGB", (224, 224), (255, 255, 255))
            imgs.append(tfm(im))
        x = torch.stack(imgs).to(device)
        with torch.no_grad():
            v = model.get_image_features(pixel_values=x)
        v = v / v.norm(dim=1, keepdim=True)
        out[s:e] = v.cpu().numpy()
        if (s // bs) % 50 == 0:
            print(f"{split} {e}/{len(df)} {time.time()-t0:.0f}s", flush=True)
        del x, v
        gc.collect()
    print(f"{split} done {time.time()-t0:.0f}s", flush=True)
    return out

E_tr = embed_df(tr, "train")
E_te = embed_df(te, "test")
E_ul = embed_df(ul, "unlabelled")

tr_q = tr.question_id.to_numpy()
te_q = te.question_id.to_numpy()
ul_q = ul.question_id.to_numpy()

# unlabelled -> train kNN (top 10)
print("ul->train kNN", flush=True)
K_UL = 10
sim = E_ul @ E_tr.T  # (50000, 1500)
top_ul = np.argpartition(-sim, K_UL, axis=1)[:, :K_UL]
top_ul = top_ul[np.arange(len(ul_q))[:, None], np.argsort(-np.take_along_axis(sim, top_ul, axis=1), axis=1)]
del sim; gc.collect()

def build_te(k, bonus, hop_w):
    # test rows: top-k over full pool
    sim_t = E_te @ np.vstack([E_tr, E_ul]).T  # (3498, 54998)
    top = np.argpartition(-sim_t, k, axis=1)[:, :k]
    vals = np.take_along_axis(sim_t, top, axis=1)
    top = top[np.arange(len(te_q))[:, None], np.argsort(-vals, axis=1)]
    del sim_t; gc.collect()
    out = np.zeros((len(te), C), np.float32)
    n_tr = len(tr)
    for r in range(len(te)):
        cnt = np.zeros(C)
        for n in top[r]:
            v = vals[r][np.where(top[r] == n)[0][0]]
            if n < n_tr:  # direct labeled neighbor
                w = 1.0 + bonus * (tr_q[n] == te_q[r])
                cnt[y[n]] += w
            else:  # unlabeled: 2-hop via its train kNN
                j = n - n_tr
                w_same = (ul_q[j] == te_q[r])
                for n2 in top_ul[j]:
                    w = hop_w * (1.0 + bonus * (tr_q[n2] == te_q[r])) * (1.0 + bonus * w_same)
                    cnt[y[n2]] += w
        out[r] = cnt
    s = out.sum(1, keepdims=True)
    return out / np.maximum(s, 1e-9)

def build_oof(k, bonus, hop_w):
    # train rows: top-k over (train excl self) + unlabelled
    sim_t = E_tr @ E_tr.T
    np.fill_diagonal(sim_t, -2)
    sim_full = np.hstack([sim_t, E_tr @ E_ul.T])
    top = np.argpartition(-sim_full, k, axis=1)[:, :k]
    vals = np.take_along_axis(sim_full, top, axis=1)
    top = top[np.arange(len(tr_q))[:, None], np.argsort(-vals, axis=1)]
    del sim_t, sim_full; gc.collect()
    n_tr = len(tr)
    out = np.zeros((len(tr), C), np.float32)
    for r in range(len(tr)):
        cnt = np.zeros(C)
        for n in top[r]:
            if n < n_tr:
                w = 1.0 + bonus * (tr_q[n] == tr_q[r])
                cnt[y[n]] += w
            else:
                j = n - n_tr
                w_same = (ul_q[j] == tr_q[r])
                for n2 in top_ul[j]:
                    if n2 == r: continue
                    w = hop_w * (1.0 + bonus * (tr_q[n2] == tr_q[r])) * (1.0 + bonus * w_same)
                    cnt[y[n2]] += w
        out[r] = cnt
    s = out.sum(1, keepdims=True)
    return out / np.maximum(s, 1e-9)

os.makedirs("/kaggle/output", exist_ok=True)
K, BONUS, HOP = 96, 5.0, 0.3
te_p = build_te(K, BONUS, HOP)
oof_p = build_oof(K, BONUS, HOP)
np.savez_compressed("/kaggle/output/knn_trans_probs.npz", oof=oof_p, te=te_p, codes=np.array(codes))
# quick oof MAP@3
def map3(scores, yy):
    tot = 0.0
    idx = np.argsort(-scores, axis=1)[:, :3]
    for i in range(len(yy)):
        seen = set()
        for kk, c in enumerate(idx[i], 1):
            if c in seen: continue
            seen.add(c)
            if c == yy[i]:
                tot += 1.0 / kk
                break
    return tot / len(yy)
print("oof MAP@3 (trans k96 b5 hop0.3):", round(map3(oof_p, y), 4), flush=True)
upload_dataset_files(["/kaggle/output/knn_trans_probs.npz"], "miscc-knn-trans-v1")
print("DONE", flush=True)
