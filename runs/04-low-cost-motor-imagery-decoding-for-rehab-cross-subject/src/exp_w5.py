"""12-model w3 ensemble LOSO (feature groups x models) + mixed with v1."""
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
cues, bases, y, sub = [], [], [], []
for f in files:
    c, b, lab = read_epochs(f, labeled=True)
    subj = re.search(r'(S\d+)-', f).group(1)
    cues.append(c); bases.append(b); y.append(lab); sub.append(np.full(len(lab), subj))
cues = np.concatenate(cues); bases = np.concatenate(bases)
y = np.concatenate(y); sub = np.concatenate(sub)
subjects = sorted(set(sub))
bp_all = band_power_features(cues, bases)
Xv1 = np.load('../outputs/train_feats.npy')
print('loaded', round(time.time()-t0, 1), flush=True)

models = {
    'lr_0.1': lambda: make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=5000)),
    'lr_1': lambda: make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000)),
    'lr_10': lambda: make_pipeline(StandardScaler(), LogisticRegression(C=10.0, max_iter=5000)),
    'lda': lambda: LinearDiscriminantAnalysis(solver='lsqr'),
}
groups = ['all', 'rts', 'bp_rts']

P_all = np.zeros((len(y), len(groups) * 4))
for s_i, s in enumerate(subjects):
    va = sub == s; tr = ~va
    rts_tr, ref = fast_riemannian_features(cues[tr])
    cspf_tr, filters = csp_features(cues[tr], y[tr])
    rts_va, _ = fast_riemannian_features(cues[va], ref=ref)
    cspf_va = csp_transform(cues[va], filters)
    F = {'all': np.concatenate([bp_all[va], rts_va, cspf_va], 1),
         'rts': rts_va, 'bp_rts': np.concatenate([bp_all[va], rts_va], 1)}
    Ftr = {'all': np.concatenate([bp_all[tr], rts_tr, cspf_tr], 1),
           'rts': rts_tr, 'bp_rts': np.concatenate([bp_all[tr], rts_tr], 1)}
    for gi, g in enumerate(groups):
        for mi, (name, mf) in enumerate(models.items()):
            m = mf()
            m.fit(Ftr[g], y[tr])
            P_all[va, gi * 4 + mi] = m.predict_proba(F[g])[:, 1]
    print(f'subject {s} done {round(time.time()-t0, 1)}s', flush=True)
np.save('../outputs/P_all_groups.npy', P_all)

# ensemble variants
def acc(Psel):
    return ((Psel.mean(axis=1) > 0.5).astype(int) == y).mean()
Pv1 = np.load('../outputs/P_v1_loso.npy')  # (n,2)
print('12-model all-group: %.4f' % acc(P_all[:, :4]))
print('12-model rts-group:  %.4f' % acc(P_all[:, 4:8]))
print('12-model bp_rts:     %.4f' % acc(P_all[:, 8:12]))
print('12-model all groups: %.4f' % acc(P_all))
print('12-model + v1:       %.4f' % acc(np.concatenate([P_all, Pv1], 1)))
# group-then-average (balanced weights)
ga = np.stack([P_all[:, :4].mean(1), P_all[:, 4:8].mean(1), P_all[:, 8:12].mean(1)], 1).mean(1)
ga1 = np.concatenate([ga[None], Pv1.mean(1)[None]], 0).mean(0)
print('group-avg:           %.4f' % acc(np.stack([P_all[:, :4].mean(1), P_all[:, 4:8].mean(1), P_all[:, 8:12].mean(1)], 1)))
print('group-avg + v1:      %.4f' % ((ga1 > 0.5).astype(int).__eq__(y).mean()))
print('DONE', round(time.time()-t0, 1), flush=True)
