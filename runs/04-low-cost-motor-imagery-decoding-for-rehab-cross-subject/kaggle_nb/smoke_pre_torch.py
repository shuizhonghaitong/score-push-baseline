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


print("SMOKE OK (pre-torch)")
