import os, time, json, warnings, random
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from sklearn.model_selection import StratifiedKFold

print("torch:", torch.__version__, "cuda:", torch.cuda.is_available())
device = "cuda" if torch.cuda.is_available() else "cpu"
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0), round(torch.cuda.get_device_properties(0).total_memory/1e9,1), "GB")

def find_data():
    for c in ["/kaggle/input/misconception-detection-in-k-12-student-math-workings", "/kaggle/input"]:
        if os.path.isdir(c) and os.path.exists(os.path.join(c, "train.csv")):
            return c
    for root, dirs, files in os.walk("/kaggle/input"):
        if "train.csv" in files:
            return root
    raise SystemExit("no data")

DATA = find_data()
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


tr = pd.read_csv(os.path.join(DATA, "train.csv"))
te = pd.read_csv(os.path.join(DATA, "test.csv"))
codes = sorted(tr.root_code.unique())
ci = {c: i for i, c in enumerate(codes)}
C = len(codes)
y = tr.root_code.map(ci).values
IMG = os.path.join(DATA, "competition_data", "images") if os.path.isdir(os.path.join(DATA, "competition_data")) else os.path.join(DATA, "images")

MEAN = (0.5, 0.5, 0.5)
STD = (0.5, 0.5, 0.5)
def make_tf(train_mode):
    if train_mode:
        return transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.85, 1.0), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ColorJitter(0.08, 0.08, 0.08),
            transforms.ToTensor(),
            transforms.Normalize(MEAN, STD),
            transforms.RandomErasing(p=0.05, scale=(0.02, 0.08)),
        ])
    return transforms.Compose([
        transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

TF_TR, TF_VA = make_tf(True), make_tf(False)

class DS(torch.utils.data.Dataset):
    def __init__(self, df, ysel, split, tf):
        self.paths = [os.path.join(IMG, split, f) for f in df.image_file]
        self.tf = tf
        self.y = ysel
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, i):
        try:
            im = Image.open(self.paths[i]).convert("RGB")
            im.load()
        except Exception:
            im = Image.new("RGB", (224, 224), (255, 255, 255))
        return self.tf(im), int(self.y[i])

from transformers import SiglipModel

DIM = 768
def build_model():
    m = SiglipModel.from_pretrained("google/siglip-base-patch16-224")
    head = nn.Linear(DIM, C)
    nn.init.zeros_(head.bias)
    nn.init.normal_(head.weight, std=0.02)
    return m, head

def feat(model, x):
    v = model.vision_model(pixel_values=x)
    return v.last_hidden_state[:, 0]

def run_epoch(model, head, loader, opt=None, sched=None):
    if opt is None:
        model.eval(); head.eval()
        tot = 0.0
        with torch.no_grad():
            for x, yb in loader:
                x = x.to(device, non_blocking=True)
                f = feat(model, x)
                p = torch.softmax(head(f), dim=-1)
                tot += (p * yb.float().to(device).unsqueeze(1)).sum().item()
        return tot / len(loader.dataset)
    model.train(); head.train()
    for x, yb in loader:
        x = x.to(device, non_blocking=True)
        yb = yb.to(device)
        f = feat(model, x)
        loss = nn.functional.cross_entropy(head(f), yb)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(head.parameters()), 1.0)
        opt.step()
        if sched is not None:
            sched.step()
    return None

def map3_from_scores(s, y, codes):
    idx = np.argsort(-s, axis=1)[:, :3]
    tot = 0.0
    for row, yi in zip(idx, y):
        seen = set(); sc = 0.0
        for k, p in enumerate(row, 1):
            if p in seen: continue
            seen.add(p)
            if p == yi:
                sc = 1.0/k; break
        tot += sc
    return tot / len(y)

EPOCHS = 16
BS = 32
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=7)
oof = np.zeros((len(tr), C), dtype=np.float32)
te_p = np.zeros((len(te), C), dtype=np.float32)

for fold, (a, b) in enumerate(skf.split(np.zeros(len(tr)), y)):
    t0 = time.time()
    vis, head = build_model()
    vis.to(device); head.to(device)
    ds_tr = DS(tr.iloc[a], y[a], "train", TF_TR)
    ds_va = DS(tr.iloc[b], y[b], "train", TF_VA)
    lt = torch.utils.data.DataLoader(ds_tr, BS, shuffle=True, num_workers=2, pin_memory=True)
    lv = torch.utils.data.DataLoader(ds_va, BS, shuffle=False, num_workers=2, pin_memory=True)
    params = list(vis.parameters()) + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=1e-4, weight_decay=0.01)
    total_steps = EPOCHS * len(lt)
    def lr_lambda(step):
        return max(1e-6, (total_steps - step) / total_steps)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    best = None
    best_vis_sd = None
    best_head_sd = None
    for ep in range(EPOCHS):
        run_epoch(vis, head, lt, opt, sched)
        va = run_epoch(vis, head, lv)
        vis.eval(); head.eval()
        with torch.no_grad():
            pv = []
            for x, _ in lv:
                x = x.to(device, non_blocking=True)
                f = feat(vis, x)
                pv.append(torch.softmax(head(f), dim=-1).cpu().numpy())
            oofb = np.concatenate(pv)
        oof[b] = oofb
        m3 = map3_from_scores(oofb, y[b], codes)
        print(f"fold {fold} ep {ep}: val_acc={va:.4f} val_map3={m3:.4f} t={time.time()-t0:.0f}s", flush=True)
        if best is None or m3 > best:
            best = m3
            best_vis_sd = {k: v.detach().cpu().clone() for k, v in vis.state_dict().items()}
            best_head_sd = {k: v.detach().cpu().clone() for k, v in head.state_dict().items()}
    vis.load_state_dict(best_vis_sd, strict=False)
    head.load_state_dict(best_head_sd)
    vis.eval(); head.eval()
    with torch.no_grad():
        ds_te = DS(te, np.zeros(len(te), dtype=int), "test", TF_VA)
        lte = torch.utils.data.DataLoader(ds_te, BS, shuffle=False, num_workers=2, pin_memory=True)
        p = []
        for x, _ in lte:
            x = x.to(device, non_blocking=True)
            f = feat(vis, x)
            p.append(torch.softmax(head(f), dim=-1).cpu().numpy())
    te_p += np.concatenate(p) / 5
    del vis, head
    torch.cuda.empty_cache()
    print(f"fold {fold} best val_map3={best:.4f}", flush=True)

os.makedirs("/kaggle/output", exist_ok=True)
m3 = map3_from_scores(oof, y, codes)
print("OOF MAP@3 siglip finetune:", round(m3, 4))
np.savez_compressed("/kaggle/output/finetune_probs.npz",
                   oof=oof, te=te_p, codes=np.array(codes),
                   tr_ids=tr.trace_id.values, te_ids=te.trace_id.values, y=y)
print("saved finetune_probs.npz")
upload_dataset_files(["/kaggle/output/finetune_probs.npz"], "miscc-siglip-v1")
print("DONE")
