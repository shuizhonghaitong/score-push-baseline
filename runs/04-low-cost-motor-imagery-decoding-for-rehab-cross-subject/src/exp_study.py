"""Unified study on per-subject LOSO (50 files, 1795 trials incl. run-3):
  A) precompute band-filtered data once, reproduce v2 w3 features exactly
  B) 12-model ensemble (3 groups x 4 models) acc
  C) per-file threshold study: 0.5 vs exact-balance (k=n//2), per run size
  D) subject-adaptive pseudo-labeling (kmeans + LDA on w3 feats)
Run: cd <run dir> && OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python src/exp_study.py
"""
import os, sys, re, time, glob, json
import numpy as np
for v in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(v, '4')
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
from common import read_epochs, DATA
from fast_riemannian import fast_riemannian_features, _triu_vec, BANDS, band_filter
from features import band_power_features, csp_features, csp_transform, _butter
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.cluster import KMeans
from sklearn.covariance import LedoitWolf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, 'outputs')
t0 = time.time()
def log(*a):
    print('[%6.1fs]' % (time.time()-t0), *a, flush=True)

def lr(c): return make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=5000))
def lda(): return LinearDiscriminantAnalysis(solver='lsqr')

# ---------------- load ----------------
files = sorted(glob.glob(DATA + '/Training set/*.csv'))
cues, bases, y, sub, run = [], [], [], [], []
for f in files:
    c, b, lab = read_epochs(f, labeled=True)
    subj = re.search(r'(S\d+)-', f).group(1)
    run_id = re.search(r'-(\d+)_eeg', f).group(1)
    cues.append(c); bases.append(b); y.append(lab)
    sub.append(np.full(len(lab), subj)); run.append(np.full(len(lab), run_id))
cues = np.concatenate(cues).astype(np.float32)
bases = np.concatenate(bases).astype(np.float32)
y = np.concatenate(y); sub = np.concatenate(sub); run = np.concatenate(run)
assert len(y) == 1795, len(y)
subjects = sorted(set(sub))
log('loaded', cues.shape, 'subjects', len(subjects))

# ---------------- precompute band-filtered cue data (6 bands) ----------------
xb = os.path.join(OUT, 'Xbf6.npy')
if os.path.exists(xb):
    Xbf6 = np.load(xb)
    log('loaded Xbf6', Xbf6.shape)
else:
    Xbf6 = np.stack([band_filter(cues, l, h).astype(np.float32) for (l, h) in BANDS])
    np.save(xb, Xbf6)
    log('saved Xbf6', Xbf6.shape)
t, c = cues.shape[1], cues.shape[2]

def rts_feats(Xsub, ref):
    feats = []
    for bi, r in enumerate(ref):
        cf = Xsub[bi]
        cov = np.einsum('nta,ntb->nab', cf, cf) / (t - 1)
        mu = np.trace(cov, axis1=1, axis2=2) / c
        cov = (1 - 0.05) * cov + 0.05 * mu[:, None, None] * np.eye(c)[None]
        w, v = np.linalg.eigh(r)
        G = v @ np.diag(np.maximum(w, 1e-12) ** -0.5) @ v.T
        W = np.einsum('ab,nbc,cd->nad', G, cov, G)
        w2, v2 = np.linalg.eigh(W)
        L = (v2 * np.log(np.maximum(w2, 1e-12))[:, None, :]) @ v2.transpose(0, 2, 1)
        feats.append(np.stack([_triu_vec(L[i]) for i in range(len(L))]))
    return np.concatenate(feats, axis=1)

def rts_ref(Xall_sub):
    refs = []
    for bi in range(6):
        cf = Xall_sub[bi]
        cov = np.einsum('nta,ntb->nab', cf, cf) / (t - 1)
        refs.append(cov.mean(axis=0))
    return np.stack(refs)

def csp_fit(Xc1, ytr):
    """Xc1: (2, n, t, 8) for bands (8-12),(12-16); returns (filters,)."""
    est = LedoitWolf(); filters = []
    for bi in range(2):
        cf = Xc1[bi]
        C1 = np.mean([np.cov((cf[i] - cf[i].mean(0)).T) for i in np.flatnonzero(ytr == 1)], axis=0)
        C0 = np.mean([np.cov((cf[i] - cf[i].mean(0)).T) for i in np.flatnonzero(ytr == 0)], axis=0)
        C1, C0 = est.fit(C1).covariance_, est.fit(C0).covariance_
        w, V = np.linalg.eigh(np.linalg.pinv(C0) @ C1)
        order = np.argsort(w)
        V = V[:, order]
        filters.append(np.concatenate([V[:, -4:], V[:, :4]], axis=1))
    return np.stack(filters)

def csp_proj(Xc1, filters):
    feats = []
    for bi in range(2):
        p = np.einsum('nta,ab->ntb', Xc1[bi], filters[bi])
        pm = p - p.mean(axis=1, keepdims=True)
        d = np.einsum('nta,nta->na', pm, pm) / (t - 1)
        feats.append(np.log(np.maximum(d, 1e-12)))
    return np.concatenate(feats, axis=1)

# sanity: reproduce original functions for S001
s0 = 'S001'; va = sub == s0; tr = ~va
Ftr_orig = None
rts_tr_o, ref_o = fast_riemannian_features(cues[tr])
cspf_tr_o, filt_o = csp_features(cues[tr], y[tr])
rts_va_o, _ = fast_riemannian_features(cues[va], ref=ref_o)
cspf_va_o = csp_transform(cues[va], filt_o)
Xs = Xbf6[:, va]
rts_va_new = rts_feats(Xs, ref_o)
cspf_va_new = csp_proj(Xs[:, [1, 2]], filt_o)
err1 = np.abs(rts_va_new - rts_va_o).max(); err2 = np.abs(cspf_va_new - cspf_va_o).max()
log('repro check S001: rts err %.3e csp err %.3e' % (err1, err2))
assert err1 < 1e-4 and err2 < 1e-4

bp_all = band_power_features(cues.astype(np.float64), bases.astype(np.float64)).astype(np.float32)
v1_all = np.load(os.path.join(OUT, 'train_feats.npy')).astype(np.float32)
log('features ready', bp_all.shape, v1_all.shape)

# ---------------- Phase A: per-subject LOSO features + probs ----------------
P_all = np.zeros((len(y), 3 + 12 + 2), dtype=np.float32)   # [P3(3) | P12(12) | Pv1(2)]
Fva_all = {}
for s in subjects:
    va = sub == s; tr = ~va
    Xtr, Xva = Xbf6[:, tr], Xbf6[:, va]
    ref = rts_ref(Xtr)
    rts_tr, rts_va = rts_feats(Xtr, ref), rts_feats(Xva, ref)
    filt = csp_fit(Xtr[:, [1, 2]], y[tr])
    csp_tr, csp_va = csp_proj(Xtr[:, [1, 2]], filt), csp_proj(Xva[:, [1, 2]], filt)
    Ftr = {'all': np.concatenate([bp_all[tr], rts_tr, csp_tr], 1),
           'rts': rts_tr, 'bp_rts': np.concatenate([bp_all[tr], rts_tr], 1)}
    Fva = {'all': np.concatenate([bp_all[va], rts_va, csp_va], 1),
           'rts': rts_va, 'bp_rts': np.concatenate([bp_all[va], rts_va], 1)}
    # P3: 3 models on 'all'
    P3 = np.stack([lr(0.1).fit(Ftr['all'], y[tr]).predict_proba(Fva['all'])[:, 1],
                   lr(1.0).fit(Ftr['all'], y[tr]).predict_proba(Fva['all'])[:, 1],
                   lda().fit(Ftr['all'], y[tr]).predict_proba(Fva['all'])[:, 1]], 1)
    P12 = np.zeros((va.sum(), 12))
    for gi, g in enumerate(['all', 'rts', 'bp_rts']):
        for mi, mf in enumerate([lambda: lr(0.1), lambda: lr(1.0), lambda: lr(10.0), lda]):
            P12[:, gi * 4 + mi] = mf().fit(Ftr[g], y[tr]).predict_proba(Fva[g])[:, 1]
    Pv1 = np.stack([lr(0.1).fit(v1_all[tr], y[tr]).predict_proba(v1_all[va])[:, 1],
                    lr(1.0).fit(v1_all[tr], y[tr]).predict_proba(v1_all[va])[:, 1]], 1)
    P_all[va] = np.concatenate([P3, P12, Pv1], 1).astype(np.float32)
    Fva_all[s] = Fva['all']
    log('subject', s, 'done')
np.save(os.path.join(OUT, 'P_loso_all.npy'), P_all)
np.savez_compressed(os.path.join(OUT, 'Fva_all_loso.npz'), **{s: Fva_all[s] for s in subjects})
log('Phase A done, saved')

# ---------------- Phase B: evaluations ----------------
def acc05(P, yy): return ((P > 0.5).astype(int) == yy).mean()
def balanced_pred(P, k):
    out = np.zeros(len(P), int); out[np.argpartition(-P, k)[:k]] = 1
    return out
def acc_bal(P, yy): return (balanced_pred(P, len(yy)//2) == yy).mean()

P3 = P_all[:, :3].mean(1); P12 = P_all[:, 3:15]; Pv1 = P_all[:, 15:].mean(1)
Pmix = 0.5 * P3 + 0.5 * Pv1
P12mix = 0.5 * P12.mean(1) + 0.5 * Pv1
log('=== overall LOSO acc @0.5 ===')
for name, P in [('P3(3mod)', P3), ('P12 all', P12.mean(1)), ('P12 grp-avg',
                 np.stack([P12[:, :4].mean(1), P12[:, 4:8].mean(1), P12[:, 8:12].mean(1)], 1).mean(1)),
                ('Pmix v2recipe', Pmix), ('P12+v1 mix', P12mix)]:
    log('  %-16s %.4f' % (name, acc05(P, y)))

log('=== per-file acc (51 files), thr 0.5 vs balanced(k=n//2) ===')
def file_table(P):
    rows = []
    for s in subjects:
        for r in ('001', '002', '003'):
            m = (sub == s) & (run == r)
            if m.sum() < 10: continue
            rows.append((s, r, int(m.sum()), acc05(P[m], y[m]), acc_bal(P[m], y[m])))
    return rows
for name, P in [('Pmix', Pmix), ('P12mix', P12mix), ('P3', P3)]:
    rows = file_table(P)
    a05 = np.mean([r[3] for r in rows]); abal = np.mean([r[4] for r in rows])
    r50 = [r for r in rows if r[2] >= 40]; r10 = [r for r in rows if r[2] < 20]
    log('  %-8s mean 0.5=%.4f bal=%.4f (n=%d) | files>=40: %.4f/%.4f | 10-trial: %.4f/%.4f'
        % (name, a05, abal, len(rows),
           np.mean([r[3] for r in r50]), np.mean([r[4] for r in r50]),
           np.mean([r[3] for r in r10]) if r10 else float('nan'), np.mean([r[4] for r in r10]) if r10 else float('nan')))
    for r in rows:
        if r[4] - r[3] < -0.1: log('    hurt:', r)

# ---------------- Phase D: adaptation ----------------
log('=== adaptation (pseudo-label kmeans->LDA on w3 feats, 1-2 iters) ===')
def adapt_eval(P0, iters):
    accs = []
    for s in subjects:
        va = sub == s; tr = ~va
        # standardize with the held-out subject's own feature stats
        mu = Fva_all[s].mean(0); sd = Fva_all[s].std(0) + 1e-8
        xv = (Fva_all[s] - mu) / sd
        P = P0[va].copy()
        for it in range(iters):
            km = KMeans(2, n_init=10, random_state=0).fit(xv)
            lab = km.labels_
            plab = (lab == (0 if P[lab == 0].mean() > P[lab == 1].mean() else 1)).astype(int)
            m = lda().fit(xv, plab)
            P = m.predict_proba(xv)[:, 1]
        for r in ('001', '002', '003'):
            msk = (sub == s) & (run == r)
            if msk.sum() < 10: continue
            accs.append(acc_bal(P[msk], y[msk]))
    return np.mean(accs)
for name, P0 in [('P3', P3), ('Pmix', Pmix)]:
    log('  %-6s 1iter bal=%.4f  2iter bal=%.4f' % (name, adapt_eval(P0, 1), adapt_eval(P0, 2)))

log('STUDY DONE')
