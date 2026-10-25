import json, os

root = os.path.dirname(os.path.abspath(__file__))
common_src = open(os.path.join(root, '..', 'src', 'common.py')).read()
feat_src = open(os.path.join(root, '..', 'src', 'features.py')).read()
# trim common.py: keep only what the notebook needs
keep_common = '''import numpy as np
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
'''
cells = []
def md(s): cells.append({"cell_type": "markdown", "metadata": {}, "source": s.splitlines(keepends=True)})
def code(s): cells.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": s.splitlines(keepends=True)})

md("# Cross-Subject EEG Motor-Imagery Decoding — v4 (v2 recipe, run-3 x2 weight, per-file balanced threshold)")
code('import os, warnings, re\nimport numpy as np\nimport pandas as pd\nwarnings.filterwarnings("ignore")\nnp.set_printoptions(suppress=True)\n\ndata_root = "/kaggle/input/competitions/low-cost-motor-imagery-decoding-for-rehab-cross-subject"\ntrain_dir = os.path.join(data_root, "Training set")\ntest_dir = os.path.join(data_root, "Cross participant test set")\nprint(len(os.listdir(train_dir)), "train files;", len(os.listdir(test_dir)), "test files")')
code(keep_common)
code(feat_src.replace("from sklearn.covariance import LedoitWolf", "from sklearn.covariance import LedoitWolf").replace("import numpy as np\nfrom scipy import signal\nfrom sklearn.covariance import LedoitWolf\nfrom common import FS", "import numpy as np\nfrom scipy import signal\nfrom sklearn.covariance import LedoitWolf"))
code('''cues, bases, y, sub, run = [], [], [], [], []
for f in sorted(glob.glob(os.path.join(train_dir, "*.csv"))):
    c, b, lab = read_epochs(f, labeled=True)
    subj = re.search(r"(S\\d+)-", os.path.basename(f)).group(1)
    rid = re.search(r"-(\d+)_eeg", os.path.basename(f)).group(1)
    cues.append(c); bases.append(b); y.append(lab); sub.append(np.full(len(lab), subj))
    run.append(np.full(len(lab), rid))
cues = np.concatenate(cues); bases = np.concatenate(bases)
y = np.concatenate(y); sub = np.concatenate(sub); run = np.concatenate(run)
subjects = sorted(set(sub))
print("trials:", cues.shape, "move frac:", y.mean(), "run3 trials:", int((run == "003").sum()))
R3W = 2
sw_all = 1.0 + (R3W - 1) * (run == "003")
rep_all = np.repeat(np.arange(len(y)), sw_all.astype(int))''')
code('''bp_all = band_power_features(cues, bases)
print("bp features:", bp_all.shape)''')
code('''# v1 features (older set) for the mixed ensemble
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
print("v1 features:", v1_all.shape)''')
code('''# ---------- CV: subject-grouped 5-fold, mixed ensemble ----------
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

rng = np.random.default_rng(0)
order = rng.permutation(subjects)
folds = np.array_split(order, 5)

accs, accs_bal = [], []
for fold in folds:
    va = np.isin(sub, list(fold)); tr = ~va
    trw = np.repeat(np.arange(len(y[tr])), (1 + (R3W - 1) * (run[tr] == "003")).astype(int))
    sw = sw_all[tr]
    rts_tr, ref = riemannian_features(cues[tr])
    _, f0 = csp_features(cues[trw], y[trw])
    cspf_tr = csp_transform(cues[tr], f0)
    rts_va, _ = riemannian_features(cues[va], ref=ref)
    cspf_va = csp_transform(cues[va], f0)
    Ftr = np.concatenate([bp_all[tr], rts_tr, cspf_tr], 1)
    Fva = np.concatenate([bp_all[va], rts_va, cspf_va], 1)
    Pw3 = np.mean([make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=5000)).fit(Ftr, y[tr], logisticregression__sample_weight=sw).predict_proba(Fva)[:, 1]
                   for c in (0.1, 1.0)], axis=0)
    Pw3 = np.mean([Pw3, LinearDiscriminantAnalysis(solver="lsqr").fit(Ftr[trw], y[trw]).predict_proba(Fva)[:, 1]], axis=0)
    Pv1 = np.mean([make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=5000)).fit(v1_all[tr], y[tr], logisticregression__sample_weight=sw).predict_proba(v1_all[va])[:, 1]
                   for c in (0.1, 1.0)], axis=0)
    P = 0.5 * Pw3 + 0.5 * Pv1
    accs.append(((P > 0.5).astype(int) == y[va]).mean())
    ab = []
    for s in list(fold):
        m = sub == s
        if m.sum() < 10: continue
        k = int(m.sum()) // 2
        pred = np.zeros(int(m.sum()), int); pred[np.argpartition(-P[m], k)[:k]] = 1
        ab.append((pred == y[m]).mean())
    accs_bal.append(np.mean(ab))
print("mixed ensemble grouped 5-fold @0.5:", np.round(accs, 3), "mean %.4f" % np.mean(accs))
print("per-file balanced:", np.round(accs_bal, 3), "mean %.4f" % np.mean(accs_bal))''')
code('''# ---------- final: fit all models on all data (run-3 x2 weight) ----------
rts_all, ref_all = riemannian_features(cues)
_, csp_filters_all = csp_features(cues[rep_all], y[rep_all])
cspf_all = csp_transform(cues, csp_filters_all)
F_all = np.concatenate([bp_all, rts_all, cspf_all], 1)
lr_all = {c: make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=5000)).fit(F_all, y, logisticregression__sample_weight=sw_all) for c in (0.1, 1.0)}
lda_all = LinearDiscriminantAnalysis(solver="lsqr").fit(F_all[rep_all], y[rep_all])
lr_v1_all = {c: make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=5000)).fit(v1_all, y, logisticregression__sample_weight=sw_all) for c in (0.1, 1.0)}
print("final w3 features:", F_all.shape)
print("final models trained (run-3 weight", R3W, ")")''')
code('''# ---------- test set (sorted file order) ----------
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

Pw3 = np.mean([lr_all[c].predict_proba(F_t)[:, 1] for c in (0.1, 1.0)], axis=0)
Pw3 = np.mean([Pw3, lda_all.predict_proba(F_t)[:, 1]], axis=0)
Pv1 = np.mean([lr_v1_all[c].predict_proba(v1_t)[:, 1] for c in (0.1, 1.0)], axis=0)
P = 0.5 * Pw3 + 0.5 * Pv1

# per-file balanced threshold: every session file is a balanced design (k = n//2 move)
pred = np.zeros(len(P), dtype=int)
pred05 = (P > 0.5).astype(int)
print("=== per-file: n, meanP, move@0.5, Ptop10/Pbot10 ===")
for f in test_files:
    m = np.flatnonzero(tsub == f)
    k = len(m) // 2
    pred[m[np.argpartition(-P[m], k)[:k]]] = 1
    pf = P[m]
    print(f, "n=", len(m), "meanP=%.3f" % pf.mean(),
          "move@0.5=", int(pred05[m].sum()),
          "Ptop10=%.3f" % np.sort(pf)[-10:].mean(),
          "Pbot10=%.3f" % np.sort(pf)[:10:].mean())
print("total move: bal=", int(pred.sum()), " @0.5=", int(pred05.sum()), " flip=", int((pred != pred05).sum()))

rows = [{"ID": str(i), "TARGET": "move" if pred[i] == 1 else "rest"} for i in range(len(P))]
df = pd.DataFrame(rows)
df.to_csv("submission.csv", index=False)
print(df.TARGET.value_counts())
print(df.head())
assert len(df) > 0
print("SUBMISSION WRITTEN, rows:", len(df))''')

nb = {"cells": cells,
      "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                   "language_info": {"name": "python"}},
      "nbformat": 4, "nbformat_minor": 5}
json.dump(nb, open(os.path.join(root, "source.ipynb"), "w"), indent=1)
print("cells:", len(cells))
