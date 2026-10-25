"""Prior-matching thresholds: per-file threshold that enforces 50/50 balance."""
import sys, glob, re, time
import numpy as np
sys.path.insert(0, '.')
from common import read_epochs, DATA
from fast_riemannian import fast_riemannian_features
from features import band_power_features, csp_features, csp_transform
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

t0 = time.time()
files = sorted(glob.glob(DATA + '/Training set/*.csv'))
cues, bases, y, sub, run = [], [], [], [], []
for f in files:
    c, b, lab = read_epochs(f, labeled=True)
    subj = re.search(r'(S\d+)-', f).group(1)
    run_id = re.search(r'-(\d+)_eeg', f).group(1)
    cues.append(c); bases.append(b); y.append(lab)
    sub.append(np.full(len(lab), subj)); run.append(np.full(len(lab), run_id))
cues = np.concatenate(cues); bases = np.concatenate(bases)
y = np.concatenate(y); sub = np.concatenate(sub); run = np.concatenate(run)
subjects = sorted(set(sub))
bp_all = band_power_features(cues, bases)
print('loaded', round(time.time()-t0, 1), 's', flush=True)

def w3_feats(tr, va):
    rts_tr, ref = fast_riemannian_features(cues[tr])
    cspf_tr, filters = csp_features(cues[tr], y[tr])
    rts_va, _ = fast_riemannian_features(cues[va], ref=ref)
    cspf_va = csp_transform(cues[va], filters)
    return (np.concatenate([bp_all[tr], rts_tr, cspf_tr], 1),
            np.concatenate([bp_all[va], rts_va, cspf_va], 1))

# per (subject, run) file simulation: cross-subject proba (others), then threshold variants
def file_eval(thr_mode):
    accs = []
    for s in subjects:
        for r in ('001', '002', '003'):
            va = (sub == s) & (run == r)
            if va.sum() < 20:
                continue
            tr = ~va
            Ftr, Fva = w3_feats(tr, va)
            P = np.mean([make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=5000)).fit(Ftr, y[tr]).predict_proba(Fva)[:, 1]
                         for c in (0.1, 1.0)] +
                        [LinearDiscriminantAnalysis(solver='lsqr').fit(Ftr, y[tr]).predict_proba(Fva)[:, 1]], axis=0)
            yv = y[va]
            n = len(yv)
            if thr_mode == '0.5':
                pred = (P > 0.5).astype(int)
            elif thr_mode == 'balance':
                # threshold so predicted balance matches 50/50
                k = n // 2
                thr = np.sort(P)[n - k]
                pred = (P > thr).astype(int)
            accs.append((pred == yv).mean())
    return np.mean(accs), np.array(accs)

for mode in ('0.5', 'balance'):
    m, accs = file_eval(mode)
    print(f'thr={mode}: mean={m:.4f}', flush=True)
print('DONE', round(time.time()-t0, 1), flush=True)
