"""Adaptation study v2 (fast features), run-3 file simulation + all-runs pool."""
import sys, glob, re, time
import numpy as np
sys.path.insert(0, '.')
from common import read_epochs, DATA
from fast_riemannian import fast_riemannian_features
from features import band_power_features, csp_features, csp_transform, band_filter, BANDS
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.cluster import KMeans

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
print('loaded', cues.shape, round(time.time()-t0, 1), 's', flush=True)

def w3_feats(tr, va, ref=None, filters=None):
    rts_tr, ref = fast_riemannian_features(cues[tr], ref=ref)
    cspf_tr, filters = csp_features(cues[tr], y[tr])
    rts_va, _ = fast_riemannian_features(cues[va], ref=ref)
    cspf_va = csp_transform(cues[va], filters)
    return (np.concatenate([bp_all[tr], rts_tr, cspf_tr], 1),
            np.concatenate([bp_all[va], rts_va, cspf_va], 1))

def crosssub(va, tr):
    Ftr, Fva = w3_feats(tr, va)
    m = make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=5000))
    m.fit(Ftr, y[tr])
    return m.predict_proba(Fva)[:, 1]

def subj_z(bp, va):
    mu = bp[va].mean(axis=0); sd = bp[va].std(axis=0) + 1e-8
    return (bp[va] - mu) / sd

def pseudo_iters(xv, P0, iters, kind):
    Pv = P0.copy()
    for it in range(iters):
        km = KMeans(2, n_init=10, random_state=0).fit(xv)
        lab = km.labels_
        d = P0[lab == 0].mean() > P0[lab == 1].mean()
        plab = (lab == (0 if d else 1)).astype(int)
        if kind == 'lda':
            try:
                m = LinearDiscriminantAnalysis(solver='lsqr').fit(xv, plab)
                Pv = m.predict_proba(xv)[:, 1]
            except Exception:
                Pv = plab.astype(float)
        elif kind == 'csp':
            # per-subject CSP pseudo-labeled on band-passed cue windows
            try:
                cftr = cues_pool  # raw windows for this subject
                cf = np.stack([band_filter(cftr[i], 8, 12) for i in range(len(cftr))])
                cf2 = np.stack([band_filter(cftr[i], 12, 16) for i in range(len(cftr))])
                f1, _ = csp_features_from(cf, plab)
                f2, _ = csp_features_from(cf2, plab)
                F = np.concatenate([f1, f2], 1)
                m = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000))
                m.fit(F, plab)
                Pv = m.predict_proba(F)[:, 1]
            except Exception:
                Pv = plab.astype(float)
        P0 = plab.astype(float)
    return Pv

def csp_features_from(cf, yy, bands=((8, 12), (12, 16)), k=4):
    from sklearn.covariance import LedoitWolf
    est = LedoitWolf()
    feats = []
    for (l, h) in bands:
        C1 = np.mean([np.cov(x.T) for x in cf[yy == 1]], axis=0)
        C0 = np.mean([np.cov(x.T) for x in cf[yy == 0]], axis=0)
        C1, C0 = est.fit(C1).covariance_, est.fit(C0).covariance_
        w, V = np.linalg.eigh(np.linalg.pinv(C0) @ C1)
        V = V[:, np.argsort(w)]
        F = np.concatenate([V[:, -k:], V[:, :k]], axis=1)
        proj = [x @ F for x in cf]
        feats.append(np.log(np.maximum(np.array([np.diag(np.cov(p.T)) for p in proj]), 1e-12)))
    return np.concatenate(feats, axis=1), None

def study(tag, pool_run, kinds=('lda', 'csp')):
    print(f'--- pool={pool_run} ---', flush=True)
    for kind in kinds:
        for iters in (1, 3):
            for blend in (0.0, 0.5, 0.8):
                accs = []
                for s in subjects:
                    va = (sub == s) & (run == pool_run) if pool_run != 'all' else (sub == s)
                    tr = ~va
                    if va.sum() < 10:
                        continue
                    global cues_pool
                    cues_pool = cues[va]
                    P0 = crosssub(va, tr)
                    xv = subj_z(bp_all, va)
                    Pv = pseudo_iters(xv, P0, iters, kind)
                    Pf = (1 - blend) * Pv + blend * P0
                    accs.append(((Pf > 0.5).astype(int) == y[va]).mean())
                print(f'{tag} kind={kind} it={iters} blend={blend}: mean={np.mean(accs):.4f} {np.round(accs,2)}', flush=True)

# baseline first
accs = []
for s in subjects:
    va = (sub == s) & (run == '003'); tr = ~va
    if va.sum() < 10:
        continue
    P0 = crosssub(va, tr)
    accs.append(((P0 > 0.5).astype(int) == y[va]).mean())
print('BASELINE run3-file: mean=%.4f %s' % (np.mean(accs), np.round(accs, 2)), flush=True)

study('adapt-run3', '003')
study('adapt-all', 'all', kinds=('lda',))
print('DONE', round(time.time()-t0, 1), 's', flush=True)
