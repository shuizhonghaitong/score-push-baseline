"""Cross-subject LOSO with run-03 feature set (CAR+notch, BP, FB-RTS, CSP)."""
import sys, glob, re, time, pickle
import numpy as np
sys.path.insert(0, '.')
from common import FS, read_epochs, DATA
from features import (band_power_features, riemannian_features,
                      csp_features, csp_transform, BANDS)
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

files = sorted(glob.glob(DATA + '/Training set/*.csv'))
print(len(files), 'files')
cues, bases, y, sub = [], [], [], []
for f in files:
    c, b, lab = read_epochs(f, labeled=True)
    subj = re.search(r'(S\d+)-', f).group(1)
    cues.append(c); bases.append(b); y.append(lab); sub.append(np.full(len(lab), subj))
cues = np.concatenate(cues); bases = np.concatenate(bases)
y = np.concatenate(y); sub = np.concatenate(sub)
subjects = sorted(set(sub))
print(cues.shape, bases.shape, y.mean())
t0 = time.time()

def all_feats(cue, base, yy, ref=None, csp_filters=None):
    bp = band_power_features(cue, base)
    rts, ref = riemannian_features(cue, ref=ref)
    if csp_filters is None:
        cspf, csp_filters = csp_features(cue, yy)
    else:
        cspf = csp_transform(cue, csp_filters)
    return np.concatenate([bp, rts, cspf], axis=1), ref, csp_filters

# precompute BP (does not depend on fold)
bp_all = band_power_features(cues, bases)
print('bp', bp_all.shape, round(time.time()-t0,1))

def loso(model_factory, feats, tag):
    accs = []
    for s in subjects:
        va = sub == s; tr = ~va
        m = model_factory()
        m.fit(feats[tr], y[tr])
        accs.append((m.predict(feats[va]) == y[va]).mean())
    return np.mean(accs), np.array(accs)

models = {
    'lr_0.1': lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=5000)),
    'lr_1': lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000)),
    'lr_10': lambda: make_pipeline(StandardScaler(), LogisticRegression(C=10.0, max_iter=5000)),
    'lda': lambda: LinearDiscriminantAnalysis(solver='lsqr'),
}

# fold-wise: rts ref + csp filters from train
print('building fold features...')
F_all = np.zeros((len(y), 0))
results = {}
for name, mf in models.items():
    accs = []
    for s in subjects:
        va = sub == s; tr = ~va
        rts_tr, ref = riemannian_features(cues[tr])
        cspf_tr, filters0 = csp_features(cues[tr], y[tr])
        rts_va, _ = riemannian_features(cues[va], ref=ref)
        cspf_va = csp_transform(cues[va], filters0)
        Ftr = np.concatenate([bp_all[tr], rts_tr, cspf_tr], axis=1)
        Fva = np.concatenate([bp_all[va], rts_va, cspf_va], axis=1)
        m = mf()
        m.fit(Ftr, y[tr])
        accs.append((m.predict(Fva) == y[va]).mean())
    results[name] = np.mean(accs)
    print(f'{name:6s} bp+rts+csp LOSO: {np.mean(accs):.4f}  {np.round(accs,2)}')

# BP only
for name, mf in models.items():
    mean, accs = loso(mf, bp_all, name)
    print(f'{name:6s} bp only        LOSO: {mean:.4f}')
    results[name + '_bp'] = mean
