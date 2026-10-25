import os, time, json, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms

print("torch:", torch.__version__, "cuda:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0).total_memory / 1e9, "GB")

DATA = None
for c in ["/kaggle/input/misconception-detection-in-k-12-student-math-workings", "/kaggle/input"]:
    if os.path.isdir(c) and os.path.exists(os.path.join(c, "train.csv")):
        DATA = c; break
if DATA is None:
    for root, dirs, files in os.walk("/kaggle/input"):
        if "train.csv" in files:
            DATA = root; break
assert DATA, "no data found"
print("DATA =", DATA)
def upload_dataset_files(files, ds_slug):
    import json, shutil, subprocess, sys
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
    r = subprocess.run([sys.executable, "-m", "kaggle", "datasets", "create", "-p", up, "-q"],
                       capture_output=True, text=True, timeout=1200)
    print("DATASET", ds_slug, "rc=", r.returncode, (r.stdout + r.stderr).strip()[:300], flush=True)
    print(f"DATASET_TAG {ds_slug} lilac1996/{ds_slug}", flush=True)
    return r.returncode == 0


print("top:", sorted(os.listdir(DATA))[:15])

tr = pd.read_csv(os.path.join(DATA, "train.csv"))
te = pd.read_csv(os.path.join(DATA, "test.csv"))
print("train:", tr.shape, "test:", te.shape)
img_root = os.path.join(DATA, "competition_data", "images") if os.path.isdir(os.path.join(DATA, "competition_data")) else os.path.join(DATA, "images")
print("img subdirs:", sorted(os.listdir(img_root)))
print("n train imgs on disk:", len(os.listdir(os.path.join(img_root, "train"))) if os.path.isdir(os.path.join(img_root, "train")) else "MISSING")

from transformers import CLIPModel, CLIPProcessor

device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

tfm = transforms.Compose([
    transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
])

def embed_images(df, split):
    d = os.path.join(img_root, split)
    emb = np.zeros((len(df), 512), dtype=np.float32)
    bs = 64
    t0 = time.time()
    for i in range(0, len(df), bs):
        batch = df.iloc[i:i+bs]
        imgs = []
        for _, row in batch.iterrows():
            p = os.path.join(d, row.image_file)
            try:
                with Image.open(p) as im:
                    imgs.append(tfm(im.convert("RGB")))
            except Exception as e:
                print("ERR", p, repr(e)[:100])
                imgs.append(tfm(Image.new("RGB", (224, 224), (255, 255, 255))))
        x = torch.stack(imgs).to(device)
        with torch.no_grad():
            z = model.visual_projection(model.vision_model(pixel_values=x).last_hidden_state[:, 0])
        z = z / z.norm(dim=-1, keepdim=True)
        emb[i:i+len(batch)] = z.float().cpu().numpy()
        if i % (bs * 10) == 0:
            dt = time.time() - t0
            print(f"{split} {i+len(batch)}/{len(df)} {dt:.0f}s rate={(i+len(batch))/max(dt,1e-6):.1f}/s", flush=True)
    return emb

emb_tr = embed_images(tr, "train")
emb_te = embed_images(te, "test")

q = tr.drop_duplicates("question_id")[["question_id", "question_text", "correct_answer"]]
texts = [f"Question: {t}\nAnswer key: {a}" for t, a in zip(q.question_text, q.correct_answer)]
tok = proc(text=texts, return_tensors="pt", padding=True, truncation=True, max_length=77)
with torch.no_grad():
    qemb = model.text_projection(model.text_model(input_ids=tok.input_ids.to(device), attention_mask=tok.attention_mask.to(device)).last_hidden_state[:, 0])
qemb = (qemb / qemb.norm(dim=-1, keepdim=True)).float().cpu().numpy()
print("qtext emb:", qemb.shape)

# ---- DINOv2 features (frozen) ----
from transformers import Dinov2Model
print("loading dinov2...", flush=True)
din = Dinov2Model.from_pretrained("facebook/dinov2-base").to(device).eval()
for pp in din.parameters():
    pp.requires_grad_(False)

def embed_dino(df, split):
    d = os.path.join(img_root, split)
    emb = np.zeros((len(df), 768), dtype=np.float32)
    bs = 32
    t0 = time.time()
    for i in range(0, len(df), bs):
        batch = df.iloc[i:i+bs]
        imgs = []
        for _, row in batch.iterrows():
            p = os.path.join(d, row.image_file)
            try:
                with Image.open(p) as im:
                    imgs.append(tfm(im.convert("RGB")))
            except Exception:
                imgs.append(tfm(Image.new("RGB", (224, 224), (255, 255, 255))))
        x = torch.stack(imgs).to(device)
        with torch.no_grad():
            z = din(pixel_values=x).last_hidden_state[:, 0]
        emb[i:i+len(batch)] = z.float().cpu().numpy()
        if i % (bs * 10) == 0:
            dt = time.time() - t0
            print(f"dino {split} {i+len(batch)}/{len(df)} {dt:.0f}s", flush=True)
    return emb

print("dino train...", flush=True)
dino_tr = embed_dino(tr, "train")
print("dino test...", flush=True)
dino_te = embed_dino(te, "test")
print("dino done", flush=True)

out = "/kaggle/output"
os.makedirs(out, exist_ok=True)

def upload_file(path, tag):
    import subprocess, re, os
    hosts = [
        ("litterbox", ["curl", "-sL", "--max-time", "600", "-F", "reqtype=fileupload",
                       "-F", "time=72h", "-F", f"fileToUpload=@{path}",
                       "https://litterbox.catbox.moe/resources/internals/api.php"]),
        ("tmpfiles", ["curl", "-sL", "--max-time", "240", "-F", f"file=@{path}", "https://tmpfiles.org/api/v1/upload"]),
        ("temp.sh", ["curl", "-sL", "--max-time", "240", "--data-binary", f"@{path}", "https://temp.sh/upload"]),
    ]
    size = os.path.getsize(path)
    for name, cmd in hosts:
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            out = (r.stdout or "").strip().replace("\n", " ")
            print(f"UPLOAD {tag} via {name} size={size} rc={r.returncode}: {out[:300]}", flush=True)
            m = re.findall(r"https?://[^\s\"]+", out)
            if r.returncode == 0 and m:
                print(f"UPLOAD_URL {tag} {name} {m[0]}", flush=True)
                return m[0]
        except Exception as e:
            print(f"UPLOAD {tag} via {name} failed: {repr(e)[:150]}", flush=True)
    print(f"UPLOAD {tag} ALL_FAILED", flush=True)
    return None
codes = np.array(sorted(tr.root_code.unique()))
np.savez_compressed(os.path.join(out, "clip_embeddings.npz"),
                    tr=emb_tr, te=emb_te, q=qemb, qids=q.question_id.values,
                    codes=codes, tr_ids=tr.trace_id.values, te_ids=te.trace_id.values,
                    tr_qids=tr.question_id.values, te_qids=te.question_id.values,
                    tr2=dino_tr, te2=dino_te)

# quick sanity: question prior + global prior submission
ci = {c: i for i, c in enumerate(codes)}
prior = np.zeros((len(te), len(codes)))
for qid, g in tr.groupby("question_id"):
    cnt = np.bincount([ci[v] for v in g.root_code], minlength=len(codes))
    rows = te.index[te.question_id == qid]
    prior[rows] = cnt + 0.5
glob = np.bincount([ci[v] for v in tr.root_code], minlength=len(codes)).astype(float) + 0.5
s = prior.copy()
s[prior.sum(1) <= 1.0] = glob
topk = np.argsort(-s, axis=1)[:, :3]
with open(os.path.join(out, "submission_prior.csv"), "w") as f:
    f.write("trace_id,root_code\n")
    for tid, row in zip(te.trace_id, topk):
        f.write(f"{tid},{ ' '.join(codes[i] for i in row) }\n")
print("output files:", os.listdir(out), flush=True)
ok1 = upload_dataset_files([os.path.join(out, "clip_embeddings.npz"),
                            os.path.join(out, "submission_prior.csv")], "miscc-clip-v9")
if not ok1:
    upload_file(os.path.join(out, "clip_embeddings.npz"), "clip_emb")
    upload_file(os.path.join(out, "submission_prior.csv"), "prior_csv")
print("done")
