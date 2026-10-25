import os, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
np.set_printoptions(suppress=True)

data_root = "/mnt/disk3/liuyutong_home/workspace/skill/score-push/runs/04-low-cost-motor-imagery-decoding-for-rehab-cross-subject/data"
train_dir = os.path.join(data_root, "Training set")
test_dir = os.path.join(data_root, "Cross participant test set")
print(os.listdir(train_dir)[:5], len(os.listdir(train_dir)))
print(os.listdir(test_dir), len(os.listdir(test_dir)))

# %%

# ---------- loading & segmentation ----------
EEG_COLS = ["Fz", "C3", "Cz", "C4", "PO7", "Pz", "PO8", "Oz"]
FS = 250
CUE_SAMPLES = 5 * FS
BASE_SAMPLES = 3 * FS

def load_trials(path):
    df = pd.read_csv(path)
    data = df[EEG_COLS].to_numpy(dtype=np.float32)
    markers = df["Marker_val"].tolist()
    trials = []
    for i, v in enumerate(markers):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            continue
        v = str(v).strip()
        if v in ("move", "rest", "cue_start") and i + CUE_SAMPLES <= len(data):
            trials.append({
                "label": v,
                "cue": data[i:i + CUE_SAMPLES],
                "baseline": data[max(0, i - BASE_SAMPLES):i],
            })
    return trials

train_files = sorted(os.listdir(train_dir))
test_files = sorted(os.listdir(test_dir))
print(len(train_files), "train files;", len(test_files), "test files")

# %%

# ---------- features ----------
from scipy.signal import butter, filtfilt

BANDS = {"delta": (1.0, 4.0), "theta": (4.0, 8.0), "mu": (8.0, 12.0),
         "beta": (13.0, 20.0), "beta2": (20.0, 30.0)}

def band_powers(x):
    n = x.shape[0]
    xw = (x - x.mean(axis=0)) * np.hanning(n)[:, None]
    spec = np.abs(np.fft.rfft(xw, axis=0)) ** 2
    freqs = np.fft.rfftfreq(n, d=1.0 / FS)
    out = {}
    for b, (lo, hi) in BANDS.items():
        m = (freqs >= lo) & (freqs < hi)
        out[b] = spec[m, :].mean(axis=0)
    return out

def trial_features(t):
    cue, base = t["cue"], t["baseline"]
    pc = band_powers(cue)
    feat = []
    for ch in range(8):
        for b in BANDS:
            feat.append(np.log1p(pc[b][ch]))
    if base.shape[0] >= 2 * FS:
        pb = band_powers(base)
        for b in ("mu", "beta", "beta2"):
            for ch in range(8):
                feat.append(np.log((pc[b][ch] + 1e-8) / (pb[b][ch] + 1e-8)))
    else:
        feat.extend([0.0] * 24)
    total = sum(v.mean() for v in pc.values()) + 1e-8
    for b in BANDS:
        feat.append(pc[b].mean() / total)
    half = cue.shape[0] // 2
    p1 = band_powers(cue[:half]); p2 = band_powers(cue[half:2 * half])
    for b in ("mu", "beta"):
        for ch in (1, 2, 3):
            feat.append((p2[b][ch] - p1[b][ch]) / (p1[b][ch] + 1e-8))
    feat.append(pc["mu"][1] + pc["mu"][2] + pc["mu"][3]
                - pc["mu"][4] - pc["mu"][5] - pc["mu"][6] - pc["mu"][7])
    return np.asarray(feat, dtype=np.float32)

# %%

X_list, y_list, sub_list = [], [], []
for f in train_files:
    subj = f.split("-")[0]
    for t in load_trials(os.path.join(train_dir, f)):
        X_list.append(trial_features(t))
        y_list.append(1 if t["label"] == "move" else 0)
        sub_list.append(subj)
Xtr = np.stack(X_list); ytr = np.array(y_list)
print("train:", Xtr.shape, "move frac", ytr.mean(), "subjects", len(set(sub_list)))

# %%

# subject-grouped 5-fold CV (sanity check of generalization)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

rng = np.random.default_rng(0)
order = rng.permutation(sorted(set(sub_list)))
folds = np.array_split(order, 5)
accs = []
for fold in folds:
    va = np.isin(sub_list, list(fold))
    m = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=3000))
    m.fit(Xtr[~va], ytr[~va])
    accs.append((m.predict(Xtr[va]) == ytr[va]).mean())
print("grouped 5-fold acc:", np.round(accs, 3), "mean %.4f" % np.mean(accs))

# %%

# Conv1D CNN trained on the 4-30 Hz band (subject-grouped 5-fold gives CV probas)
import torch
import torch.nn as nn

class Conv1DClassifier(nn.Module):
    def __init__(self, n_ch=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_ch, 24, 15, stride=4), nn.BatchNorm1d(24), nn.GELU(), nn.Dropout(0.2),
            nn.Conv1d(24, 48, 11, stride=4), nn.BatchNorm1d(48), nn.GELU(), nn.Dropout(0.2),
            nn.Conv1d(48, 64, 9, stride=4), nn.BatchNorm1d(64), nn.GELU(), nn.Dropout(0.2),
            nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(64, 1))
    def forward(self, x):
        return self.net(x).squeeze(-1)

def prep_cnn(cue):
    b, a = butter(3, [4, 30], btype="bandpass", fs=FS)
    x = filtfilt(b, a, cue, axis=0)
    x = (x - x.mean()) / (x.std() + 1e-8)
    return x.T.astype(np.float32)  # (ch, time) for Conv1d

def cnn_predict_proba(Xc, y, Xval, yval=None, seed=0, epochs=8):
    torch.manual_seed(seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Conv1DClassifier().to(dev)
    opt = torch.optim.Adam(model.parameters(), 1e-3, weight_decay=1e-3)
    lossf = nn.BCEWithLogitsLoss()
    Xt = torch.tensor(Xc, dtype=torch.float32, device=dev)
    yt = torch.tensor(y.astype(np.float32), device=dev)
    n = len(yt); perm = torch.randperm(n)
    for ep in range(epochs):
        model.train()
        tot = 0.0
        for i in range(0, n, 128):
            idx = perm[i:i+128]
            opt.zero_grad()
            loss = lossf(model(Xt[idx]), yt[idx])
            loss.backward(); opt.step(); tot += loss.item() * len(idx)
        if ep % 2 == 0:
            model.eval()
            with torch.no_grad():
                pv = torch.sigmoid(model(torch.tensor(Xval, dtype=torch.float32, device=dev))).cpu().numpy()
            if yval is not None:
                print(f"epoch {ep}: loss {tot/n:.4f} cv-acc {((pv > 0.5) == yval).mean():.4f}")
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(torch.tensor(Xval, dtype=torch.float32, device=dev))).cpu().numpy()

# %%

# re-encode train trials for CNN (raw windows kept from load step)
# (we reload to keep memory small)
train_trials = []
for f in train_files:
    train_trials.extend(load_trials(os.path.join(train_dir, f)))
Xc_train = np.stack([prep_cnn(t["cue"]) for t in train_trials])
print("cnn train", Xc_train.shape)

# %%

# grouped-5-fold CNN CV accuracy + OOF probabilities
proba_cnn = np.zeros(len(ytr))
for fold in folds:
    va = np.isin(sub_list, list(fold)); tr = ~va
    p = cnn_predict_proba(Xc_train[tr], ytr[tr], Xc_train[va], yval=ytr[va], seed=0, epochs=8)
    proba_cnn[va] = p
print("CNN grouped-CV acc: %.4f" % ((proba_cnn > 0.5) == ytr).mean())

# %%

# ---------- final models ----------
# 1) LR ensemble on v1 features, all training data
models_lr = [make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000)) for C in (0.1, 0.3, 1.0)]
for m in models_lr:
    m.fit(Xtr, ytr)
# 2) CNN retrained on all data (2 seeds)
cnn_final = []
for s in (0, 1):
    torch.manual_seed(s)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Conv1DClassifier().to(dev)
    opt = torch.optim.Adam(model.parameters(), 1e-3, weight_decay=1e-3)
    lossf = nn.BCEWithLogitsLoss()
    Xt = torch.tensor(Xc_train, dtype=torch.float32, device=dev)
    yt = torch.tensor(ytr.astype(np.float32), device=dev)
    n = len(yt)
    for ep in range(12):
        model.train()
        perm = torch.randperm(n)
        for i in range(0, n, 256):
            idx = perm[i:i+256]
            opt.zero_grad()
            loss = lossf(model(Xt[idx]), yt[idx])
            loss.backward(); opt.step()
    model.eval()
    cnn_final.append(model)
print("final models trained")

# %%

# ---------- test set ----------
test_trials = []
for f in test_files:
    test_trials.extend(load_trials(os.path.join(test_dir, f)))
Xt = np.stack([trial_features(t) for t in test_trials])
Xct = np.stack([prep_cnn(t["cue"]) for t in test_trials])
print("test:", Xt.shape, "files:", len(test_files))

P_lr = np.mean([m.predict_proba(Xt)[:, 1] for m in models_lr], axis=0)
P_cnn = np.mean([
    torch.sigmoid(m(torch.tensor(Xct, dtype=torch.float32, device=m.device))).cpu().numpy()
    for m in cnn_final
], axis=0)
P = 0.6 * P_lr + 0.4 * P_cnn

rows = [{"ID": str(i), "TARGET": "move" if P[i] > 0.5 else "rest"} for i in range(len(P))]
df = pd.DataFrame(rows)
df.to_csv("submission.csv", index=False)
print(df.TARGET.value_counts())
print(df.head())
assert len(df) > 0
print("SUBMISSION WRITTEN")