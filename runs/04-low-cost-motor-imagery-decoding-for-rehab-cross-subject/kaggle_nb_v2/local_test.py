import os, warnings, re
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")
np.set_printoptions(suppress=True)

data_root = "/mnt/disk3/liuyutong_home/workspace/skill/score-push/runs/04-low-cost-motor-imagery-decoding-for-rehab-cross-subject/data"
train_dir = os.path.join(data_root, "Training set")
test_dir = os.path.join(data_root, "Cross participant test set")
print(len(os.listdir(train_dir)), "train files;", len(os.listdir(test_dir)), "test files")

# %%

import numpy as np
import pandas as pd
import glob, os, re
from scipy import signal

FS = 250.0
CHANNELS = ['Fz', 'C3', 'Cz', 'C4', 'PO7', 'Pz', 'PO8', 'Oz']
CUE_LEN = 5.0
BASE_LEN = 3.0

_NOTCH_CACHE = [None]
def notch(x):
    if _NOTCH_CACHE[0] is None:
        _NOTCH_CACHE[0] = signal.iirnotch(50.0 / (FS / 2.0), 30.0)
    b, a = _NOTCH_CACHE[0]
    return signal.filtfilt(b, a, x, axis=0)

def car(x):
    return x - x.mean(axis=1, keepdims=True)

def read_epochs(path, labeled=True):
    df = pd.read_csv(path, low_memory=False)
    x = df[CHANNELS].to_numpy(np.float64)
    m = df['Marker_val'].astype(str)
    x = car(x)
    x = notch(x)
    if labeled:
        cue_idx = np.flatnonzero(m.isin(['rest', 'move']).to_numpy())
        labels = np.array([1 if v == 'move' else 0 for v in m.iloc[cue_idx]])
    else:
        cue_idx = np.flatnonzero((m == 'cue_start').to_numpy())
        labels = None
    tcue = int(CUE_LEN * FS)
    tbase = int(BASE_LEN * FS)
    cues, bases = [], []
    for i in cue_idx:
        s, e = i, i + tcue
        if e > len(x):
            continue
        cues.append(x[s:e])
        b0 = max(0, i - tbase)
        bases.append(x[b0:i][-tbase:])
    cues = np.stack(cues)
    bases = np.stack(bases)
    return cues, bases, labels


# %%

"""Feature extractors (ML-track compliant: no neural networks)."""
import numpy as np
from scipy import signal
from sklearn.covariance import LedoitWolf

BANDS = [(4, 8), (8, 12), (12, 16), (16, 20), (20, 25), (25, 30)]
BANDS7 = [(4, 6), (6, 8), (8, 10), (10, 12), (12, 16), (16, 20), (20, 30)]
BUTTER = {}
def _butter(l, h, order=4):
    key = (float(l), float(h))
    if key not in BUTTER:
        nyq = FS / 2
        BUTTER[key] = signal.butter(order, [max(0.01, l / nyq), min(0.99, h / nyq)], btype='bandpass')
    return BUTTER[key]

def band_filter(x, l, h):
    b, a = _butter(l, h)
    axis = 1 if x.ndim == 3 else 0
    return signal.filtfilt(b, a, x, axis=axis)

def band_power_features(cue, base, bands=BANDS, off=0.0):
    if off > 0:
        k = int(off * FS)
        cue = cue[:, k:, :]
    """Log band power, cue absolute + relative to baseline. (N, 4*nbands)"""
    N = cue.shape[0]
    feats = []
    for (l, h) in bands:
        cf = band_filter(cue, l, h)
        bf = band_filter(base, l, h)
        pc = np.log(np.maximum((cf ** 2).mean(axis=1), 1e-12))  # (N, 8)
        pb = np.log(np.maximum((bf ** 2).mean(axis=1), 1e-12))
        feats.append(pc)
        feats.append(np.log(np.maximum((cf ** 2).mean(axis=1), 1e-12) / np.maximum((bf ** 2).mean(axis=1), 1e-12)))
    return np.concatenate(feats, axis=1)

def _spdm(cue):
    """Spatio-temporal covariance (N, 8, 8)."""
    return np.array([np.cov(x.T) for x in cue])

def _matrix_log(A):
    w, v = np.linalg.eigh(A)
    return v @ np.diag(np.log(np.maximum(w, 1e-12))) @ v.T

def riemannian_features(cue, bands=BANDS, ref=None, shrink=True, off=0.0):
    if off > 0:
        k = int(off * FS)
        cue = cue[:, k:, :]
    """FB-RTS: per-band band-filtered covariances projected onto log-Euclidean
    tangent space at reference mean. Returns (feats, ref)."""
    est = LedoitWolf()
    covs = []
    for (l, h) in bands:
        cf = band_filter(cue, l, h)
        covs.append(np.array([est.fit(x).covariance_ for x in cf]))
    if ref is None:
        ref = np.stack([np.mean(c, axis=0) for c in covs])
    feats = []
    for ci, r in zip(covs, ref):
        Ginv_sqrt = _matrix_op(r, -0.5)
        f = []
        for C in ci:
            W = Ginv_sqrt @ C @ Ginv_sqrt
            L = _matrix_log(W)
            idx = np.triu_indices(8)
            v = L[idx]
            v[1:] *= np.sqrt(2.0)
            f.append(v)
        feats.append(np.stack(f))
    return np.concatenate(feats, axis=1), ref

def _matrix_op(A, p):
    w, v = np.linalg.eigh(A)
    return v @ np.diag(w ** p) @ v.T

def csp_features(cue, y, bands=((8,12),(12,16)), k=4, off=0.0):
    if off > 0:
        k2 = int(off * FS)
        cue = cue[:, k2:, :]
    """CSP spatial filters per band; diagonal-bandpower features. Returns (feats, filters)."""
    est = LedoitWolf()
    filters = []
    feats = []
    for (l, h) in bands:
        cf = band_filter(cue, l, h)
        C1 = np.mean([np.cov(x.T) for x in cf[y == 1]], axis=0) if np.sum(y == 1) > 2 else np.eye(8)
        C0 = np.mean([np.cov(x.T) for x in cf[y == 0]], axis=0) if np.sum(y == 0) > 2 else np.eye(8)
        C1, C0 = est.fit(C1).covariance_, est.fit(C0).covariance_
        w, V = np.linalg.eigh(np.linalg.pinv(C0) @ C1)
        order = np.argsort(w)
        V = V[:, order]
        F = np.concatenate([V[:, -k:], V[:, :k]], axis=1)  # (8, 2k)
        filters.append(F)
        proj = [x @ F for x in cf]
        d = np.log(np.maximum(np.array([np.diag(np.cov(p.T)) for p in proj]), 1e-12))
        feats.append(d)
    return np.concatenate(feats, axis=1), filters

def csp_transform(cue, filters, bands=((8,12),(12,16)), off=0.0):
    if off > 0:
        k2 = int(off * FS)
        cue = cue[:, k2:, :]
    feats = []
    for (l, h), F in zip(bands, filters):
        cf = band_filter(cue, l, h)
        proj = [x @ F for x in cf]
        d = np.log(np.maximum(np.array([np.diag(np.cov(p.T)) for p in proj]), 1e-12))
        feats.append(d)
    return np.concatenate(feats, axis=1)

def timdomain_features(cue, l=4.0, h=30.0, fs_out=50.0, off=0.0):
    """Bandpass + decimate: (N, 8, T/fs_out*...)."""
    if off > 0:
        k2 = int(off * FS)
        cue = cue[:, k2:, :]
    xf = band_filter(cue, l, h)
    r = int(FS / fs_out)
    return xf[:, ::r, :].reshape(xf.shape[0], -1)


# %%

cues, bases, y, sub = [], [], [], []
for f in sorted(glob.glob(os.path.join(train_dir, "*.csv"))):
    c, b, lab = read_epochs(f, labeled=True)
    subj = re.search(r"(S\d+)-", os.path.basename(f)).group(1)
    cues.append(c); bases.append(b); y.append(lab); sub.append(np.full(len(lab), subj))
cues = np.concatenate(cues); bases = np.concatenate(bases)
y = np.concatenate(y); sub = np.concatenate(sub)
subjects = sorted(set(sub))
print("trials:", cues.shape, "move frac:", y.mean())

# %%

bp_all = band_power_features(cues, bases)
print("bp features:", bp_all.shape)

# %%

# v1 features (older set) for the mixed ensemble
def band_powers(x):
    n = x.shape[0]
    xw = (x - x.mean(axis=0)) * np.hanning(n)[:, None]
    spec = np.abs(np.fft.rfft(xw, axis=0)) ** 2
    freqs = np.fft.rfftfreq(n, d=1.0 / FS)
    out = {}
    for b, (lo, hi) in {"delta": (1, 4), "theta": (4, 8), "mu": (8, 12),
                        "beta": (13, 20), "beta2": (20, 30)}.items():
        m = (freqs >= lo) & (freqs < hi)
        out[b] = spec[m, :].mean(axis=0)
    return out

def v1_features(cue, base):
    pc = band_powers(cue)
    feat = []
    for ch in range(8):
        for b in ("delta", "theta", "mu", "beta", "beta2"):
            feat.append(np.log1p(pc[b][ch]))
    if base.shape[0] >= 2 * 250:
        pb = band_powers(base)
        for b in ("mu", "beta", "beta2"):
            for ch in range(8):
                feat.append(np.log((pc[b][ch] + 1e-8) / (pb[b][ch] + 1e-8)))
    else:
        feat.extend([0.0] * 24)
    total = sum(v.mean() for v in pc.values()) + 1e-8
    for b in ("delta", "theta", "mu", "beta", "beta2"):
        feat.append(pc[b].mean() / total)
    half = cue.shape[0] // 2
    p1 = band_powers(cue[:half]); p2 = band_powers(cue[half:2 * half])
    for b in ("mu", "beta"):
        for ch in (1, 2, 3):
            feat.append((p2[b][ch] - p1[b][ch]) / (p1[b][ch] + 1e-8))
    feat.append(pc["mu"][1] + pc["mu"][2] + pc["mu"][3]
                - pc["mu"][4] - pc["mu"][5] - pc["mu"][6] - pc["mu"][7])
    return np.asarray(feat, dtype=np.float32)

v1_all = np.stack([v1_features(cues[i], bases[i]) for i in range(len(y))])
print("v1 features:", v1_all.shape)

# %%

# ---------- CV: subject-grouped 5-fold, mixed ensemble ----------
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

rng = np.random.default_rng(0)
order = rng.permutation(subjects)
folds = np.array_split(order, 5)

def fold_feats(tr, va):
    rts_tr, ref = riemannian_features(cues[tr])
    cspf_tr, f0 = csp_features(cues[tr], y[tr])
    rts_va, _ = riemannian_features(cues[va], ref=ref)
    cspf_va = csp_transform(cues[va], f0)
    return (np.concatenate([bp_all[tr], rts_tr, cspf_tr], 1),
            np.concatenate([bp_all[va], rts_va, cspf_va], 1))

accs = []
for fold in folds:
    va = np.isin(sub, list(fold)); tr = ~va
    Ftr, Fva = fold_feats(tr, va)
    Pw3 = np.mean([make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=5000)).fit(Ftr, y[tr]).predict_proba(Fva)[:, 1]
                   for c in (0.1, 1.0)], axis=0)
    Pw3 = np.mean([Pw3, LinearDiscriminantAnalysis(solver="lsqr").fit(Ftr, y[tr]).predict_proba(Fva)[:, 1]], axis=0)
    Pv1 = np.mean([make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=5000)).fit(v1_all[tr], y[tr]).predict_proba(v1_all[va])[:, 1]
                   for c in (0.1, 1.0)], axis=0)
    P = 0.5 * Pw3 + 0.5 * Pv1
    accs.append(((P > 0.5).astype(int) == y[va]).mean())
print("mixed ensemble grouped 5-fold:", np.round(accs, 3), "mean %.4f" % np.mean(accs))

# %%

# ---------- final: fit all models on all data ----------
rts_all, ref_all = riemannian_features(cues)
_, csp_filters_all = csp_features(cues, y)
cspf_all = csp_transform(cues, csp_filters_all)
F_all = np.concatenate([bp_all, rts_all, cspf_all], 1)
print("final w3 features:", F_all.shape)
print("final models trained")

# %%

# ---------- test set (sorted file order) ----------
test_files = sorted(os.listdir(test_dir))
test_cues, test_bases, tsub = [], [], []
for f in test_files:
    c, b, _ = read_epochs(os.path.join(test_dir, f), labeled=False)
    test_cues.append(c); test_bases.append(b)
    tsub.append(np.full(len(c), f))
test_cues = np.concatenate(test_cues); test_bases = np.concatenate(test_bases)
tsub = np.concatenate(tsub)
print("test trials:", test_cues.shape, "files:", len(test_files))

bp_t = band_power_features(test_cues, test_bases)
rts_t, _ = riemannian_features(test_cues, ref=ref_all)
cspf_t = csp_transform(test_cues, csp_filters_all)
F_t = np.concatenate([bp_t, rts_t, cspf_t], 1)
v1_t = np.stack([v1_features(test_cues[i], test_bases[i]) for i in range(len(bp_t))])

Pw3 = np.mean([make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=5000)).fit(F_all, y).predict_proba(F_t)[:, 1]
               for c in (0.1, 1.0)], axis=0)
Pw3 = np.mean([Pw3, LinearDiscriminantAnalysis(solver="lsqr").fit(F_all, y).predict_proba(F_t)[:, 1]], axis=0)
Pv1 = np.mean([make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=5000)).fit(v1_all, y).predict_proba(v1_t)[:, 1]
               for c in (0.1, 1.0)], axis=0)
P = 0.5 * Pw3 + 0.5 * Pv1

rows = [{"ID": str(i), "TARGET": "move" if P[i] > 0.5 else "rest"} for i in range(len(P))]
df = pd.DataFrame(rows)
df.to_csv("submission.csv", index=False)
print(df.TARGET.value_counts())
print(df.head())
assert len(df) > 0
print("SUBMISSION WRITTEN, rows:", len(df))