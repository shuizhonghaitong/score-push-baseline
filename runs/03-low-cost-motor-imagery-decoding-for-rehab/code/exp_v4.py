"""Compare ensemble strategies for v4 on 5fold OOF + run3 holdout (17 subjects).

Configs = v3's 11 (bandset A off 0.0 + bandset B off 0.5) plus a few extras
(csp-only, bp-only, td). Weights: uniform vs per-config OOF-acc vs run3-acc.
"""
import sys, os, time, json, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('OMP_NUM_THREADS', '1')
from common import load_subject, TEST_SUBJECTS, FS
import features as F
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

W3 = int(os.environ.get('LCMID_W3', '3'))
SEED = 42
BANDSETS = {
    'A': (F.BANDS, ((8, 12), (12, 16))),
    'B': (F.BANDS7, ((8, 12), (12, 16), (16, 20))),
    'C': ([(2, 4), (4, 6), (6, 8), (8, 10), (10, 12), (12, 14), (14, 16), (16, 20), (20, 25), (25, 30)], ((8, 12), (12, 16))),
    'D': ([(2, 3.5), (3.5, 5), (5, 6.5), (6.5, 8), (8, 9.5), (9.5, 11), (11, 12.5), (12.5, 14), (14, 15.5), (15.5, 17), (17, 19), (19, 22), (22, 26), (26, 30)], ((8, 12), (12, 16))),
}
CONFIGS_FULL = [
    ('all', 'lr_0.1', 0.0, 'A'), ('all', 'lr_1', 0.0, 'A'), ('all', 'lr_10', 0.0, 'A'),
    ('rts', 'lr_0.1', 0.0, 'A'), ('rts', 'lr_1', 0.0, 'A'),
    ('bp_rts', 'lr_0.1', 0.0, 'A'),
    ('all', 'lda_0.5', 0.0, 'A'), ('rts', 'lda_0.5', 0.0, 'A'),
    ('all', 'lr_0.1', 0.5, 'B'),
    ('rts', 'lr_0.1', 0.5, 'B'),
    ('bp_rts', 'lr_0.1', 0.5, 'B'),
    ('csp', 'lr_0.1', 0.0, 'A'), ('csp', 'lda_0.5', 0.0, 'A'),
    ('bp', 'lr_1', 0.0, 'A'),
    ('td', 'lr_1', 0.0, 'A'),
]

CONFIGS_EXTRA = [
    ('all', 'lr_0.1', 0.25, 'A'), ('rts', 'lr_0.1', 0.25, 'A'), ('bp_rts', 'lr_0.1', 0.25, 'A'),
    ('all', 'lr_0.1', 0.75, 'A'), ('rts', 'lr_0.1', 0.75, 'A'), ('bp_rts', 'lr_0.1', 0.75, 'A'),
    ('all', 'lr_0.1', 0.0, 'A', 1000), ('rts', 'lr_0.1', 0.0, 'A', 1000), ('bp_rts', 'lr_0.1', 0.0, 'A', 1000),
    ('all', 'lr_0.1', 1.0, 'A'), ('rts', 'lr_0.1', 1.0, 'A'), ('bp_rts', 'lr_0.1', 1.0, 'A'),
]
def _cfg(c):
    return c + (None,) * (5 - len(c))
CONFIGS_FULL = [_cfg(c) for c in CONFIGS_FULL]
CONFIGS = CONFIGS_FULL[:11] if os.environ.get('LCMID_11CFG') == '1' else CONFIGS_FULL
if os.environ.get('LCMID_WIDE') == '1':
    CONFIGS = CONFIGS_FULL[:11] + [_cfg(c) for c in CONFIGS_EXTRA]
EXTRA6 = [
    ('all', 'lr_0.1', 0.25, 'B'), ('rts', 'lr_0.1', 0.25, 'B'), ('bp_rts', 'lr_0.1', 0.25, 'B'),
    ('all', 'lr_0.1', 0.75, 'B'), ('rts', 'lr_0.1', 0.75, 'B'), ('bp_rts', 'lr_0.1', 0.75, 'B'),
    ('all', 'lr_0.3', 0.0, 'A'), ('rts', 'lr_0.3', 0.0, 'A'), ('bp_rts', 'lr_0.3', 0.0, 'A'),
    ('all', 'lda_0.3', 0.0, 'A'), ('rts', 'lda_0.3', 0.0, 'A'), ('bp_rts', 'lda_0.3', 0.0, 'A'),
]
if os.environ.get('LCMID_V6') == '1':
    CONFIGS = CONFIGS_FULL[:11] + [_cfg(c) for c in CONFIGS_EXTRA] + [_cfg(c) for c in EXTRA6]
if os.environ.get('LCMID_V7') == '1':
    extraB0 = [('all', 'lr_0.1', 0.0, 'B'), ('rts', 'lr_0.1', 0.0, 'B'), ('bp_rts', 'lr_0.1', 0.0, 'B'),
               ('all', 'lr_0.3', 0.0, 'B'), ('rts', 'lda_0.3', 0.0, 'B')]
    CONFIGS = CONFIGS_FULL[:11] + [_cfg(c) for c in CONFIGS_EXTRA] + [_cfg(c) for c in EXTRA6] + [_cfg(c) for c in extraB0]
if os.environ.get('LCMID_V13') == '1':
    extraO = [('rts', 'lr_0.1', o, 'A') for o in (0.125, 0.375, 0.625, 0.875)] + \
             [('all', 'lr_0.1', o, 'A') for o in (0.125, 0.375, 0.625, 0.875)]
    CONFIGS = CONFIGS_FULL[:11] + [_cfg(c) for c in CONFIGS_EXTRA] + [_cfg(c) for c in EXTRA6] + [_cfg(c) for c in extraO]
if os.environ.get('LCMID_V16') == '1':
    EB = [(2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 10), (10, 11), (11, 12), (12, 13), (13, 14), (14, 15), (15, 16), (16, 18), (18, 20), (20, 22), (22, 24), (24, 26), (26, 28), (28, 30)]
    BANDSETS['E'] = (EB, ((8, 12), (12, 16)))
    extraE = [('all', 'lr_0.1', 0.0, 'E'), ('rts', 'lr_0.1', 0.0, 'E'), ('bp_rts', 'lr_0.1', 0.0, 'E')]
    CONFIGS = CONFIGS_FULL[:11] + [_cfg(c) for c in CONFIGS_EXTRA] + [_cfg(c) for c in EXTRA6] + [_cfg(c) for c in extraE]
if os.environ.get('LCMID_V17') == '1':
    EB = [(2, 3), (3, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 10), (10, 11), (11, 12), (12, 13), (13, 14), (14, 15), (15, 16), (16, 18), (18, 20), (20, 22), (22, 24), (24, 26), (26, 28), (28, 30)]
    BANDSETS['E'] = (EB, ((8, 12), (12, 16)))
    extraBands = [('all', 'lr_0.1', 0.0, 'C'), ('rts', 'lr_0.1', 0.0, 'C'), ('bp_rts', 'lr_0.1', 0.0, 'C'),
                  ('all', 'lr_0.1', 0.0, 'D'), ('rts', 'lr_0.1', 0.0, 'D'), ('bp_rts', 'lr_0.1', 0.0, 'D'),
                  ('all', 'lr_0.1', 0.0, 'E'), ('rts', 'lr_0.1', 0.0, 'E'), ('bp_rts', 'lr_0.1', 0.0, 'E')]
    CONFIGS = CONFIGS_FULL[:11] + [_cfg(c) for c in CONFIGS_EXTRA] + [_cfg(c) for c in EXTRA6] + [_cfg(c) for c in extraBands]
if os.environ.get('LCMID_V14') == '1':
    extraBands = [('all', 'lr_0.1', 0.0, 'C'), ('rts', 'lr_0.1', 0.0, 'C'), ('bp_rts', 'lr_0.1', 0.0, 'C'),
                  ('all', 'lr_0.1', 0.0, 'D'), ('rts', 'lr_0.1', 0.0, 'D'), ('bp_rts', 'lr_0.1', 0.0, 'D')]
    CONFIGS = CONFIGS_FULL[:11] + [_cfg(c) for c in CONFIGS_EXTRA] + [_cfg(c) for c in EXTRA6] + [_cfg(c) for c in extraBands]
if os.environ.get('LCMID_TRIM20') == '1':
    keep = [c for c in CONFIGS_EXTRA if not (len(c) == 4 and c[2] == 1.0)]
    CONFIGS = CONFIGS_FULL[:11] + [_cfg(c) for c in keep]

def make_model(kind):
    if kind.startswith('lda_'):
        return LinearDiscriminantAnalysis(solver='lsqr', shrinkage=float(kind[4:]))
    if kind.startswith('lr_'):
        return LogisticRegression(C=float(kind[3:]), max_iter=2000)
    raise ValueError(kind)

def run3_rep(n, r3):
    idx = list(range(n))
    for i in r3:
        idx.extend([i] * (W3 - 1))
    return np.array(idx)

def featurize(name, cues, bases, y, state, off=0.0, bs='A', cue_end=None):
    if cue_end:
        cues = cues[:, :cue_end]
    bands, cbands = BANDSETS[bs]
    if name in ('bp', 'td'):
        if name == 'bp':
            return F.band_power_features(cues, bases, bands=bands, off=off)
        return F.timdomain_features(cues, off=off)
    if state is None:
        if name == 'rts':
            x, ref = F.riemannian_features(cues, bands=bands, off=off)
            return x, ref
        if name == 'bp_rts':
            x, ref = F.riemannian_features(cues, bands=bands, off=off)
            return np.concatenate([F.band_power_features(cues, bases, bands=bands, off=off), x], 1), ref
        if name == 'all':
            x, ref = F.riemannian_features(cues, bands=bands, off=off)
            c, filt = F.csp_features(cues, y, bands=cbands, off=off)
            return np.concatenate([F.band_power_features(cues, bases, bands=bands, off=off), x, c], 1), (ref, filt)
        if name == 'csp':
            return F.csp_features(cues, y, bands=cbands, off=off)
        raise ValueError(name)
    if name == 'rts':
        x, _ = F.riemannian_features(cues, bands=bands, ref=state, off=off)
        return x
    if name == 'bp_rts':
        x, _ = F.riemannian_features(cues, bands=bands, ref=state, off=off)
        return np.concatenate([F.band_power_features(cues, bases, bands=bands, off=off), x], 1)
    if name == 'all':
        ref, filt = state
        x, _ = F.riemannian_features(cues, bands=bands, ref=ref, off=off)
        c = F.csp_transform(cues, filt, bands=cbands, off=off)
        return np.concatenate([F.band_power_features(cues, bases, bands=bands, off=off), x, c], 1)
    if name == 'csp':
        return F.csp_transform(cues, state, bands=cbands, off=off)
    raise ValueError(name)

def fit_predict(name, kind, cues, bases, y, tr_i, va_i, off, bs, cue_end=None, tr3=None):
    if name in ('bp', 'td'):
        Xtr = featurize(name, cues[tr_i], bases[tr_i], None, None, off, bs, cue_end)
        Xva = featurize(name, cues[va_i], bases[va_i], None, None, off, bs, cue_end)
    else:
        Xtr, state = featurize(name, cues[tr_i], bases[tr_i], y[tr_i], None, off, bs, cue_end)
        Xva = featurize(name, cues[va_i], bases[va_i], None, state, off, bs, cue_end)
    m = make_pipeline(StandardScaler(), make_model(kind))
    if tr3 is not None and W3 > 1:
        rep = run3_rep(len(tr_i), tr3)
        m.fit(Xtr[rep], y[tr_i][rep])
    else:
        m.fit(Xtr, y[tr_i])
    return m.predict_proba(Xva)[:, 1]

def best_thr(oof, y, lo=0.40, hi=0.60):
    best_t, best_a = 0.5, accuracy_score(y, (oof >= 0.5).astype(int))
    for t in np.arange(lo, hi + 1e-9, 0.01):
        a = accuracy_score(y, (oof >= t).astype(int))
        if a > best_a:
            best_a, best_t = a, t
    return float(best_t)

def eval_subject(sub, configs):
    d = load_subject(sub)
    keys = [k for k in ('tr01', 'tr02', 'tr03') if k in d]
    cues = np.concatenate([d[k][0] for k in keys])
    bases = np.concatenate([d[k][1] for k in keys])
    y = np.concatenate([d[k][2] for k in keys])
    pos, r3 = 0, []
    for k in keys:
        n = len(d[k][2])
        if k == 'tr03':
            r3 = list(range(pos, pos + n))
        pos += n
    r3a = np.array(r3)
    mask3 = np.ones(len(y), bool)
    mask3[r3a] = False
    def _oof(seed):
        skf2 = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
        o = np.zeros((len(y), len(configs)))
        for j, (fn, mk, off, bs, ce) in enumerate(configs):
            for tr_i, va_i in skf2.split(cues, y):
                tr3 = [int(np.searchsorted(tr_i, g)) for g in r3
                       if np.searchsorted(tr_i, g) < len(tr_i) and tr_i[np.searchsorted(tr_i, g)] == g]
                o[va_i, j] = fit_predict(fn, mk, cues, bases, y, tr_i, va_i, off, bs, ce, tr3)
        return o
    oof = _oof(SEED)
    if os.environ.get('LCMID_2SEED') == '1':
        oof7 = _oof(7)
    acc5 = np.array([accuracy_score(y, (oof[:, j] >= 0.5).astype(int)) for j in range(len(configs))])
    # run3 holdout per config
    have3 = len(r3) >= 4 and int(np.bincount(y[r3]).min()) >= 2
    p3 = np.zeros((len(r3), len(configs))) if have3 else None
    acc3 = np.full(len(configs), np.nan)
    if have3:
        for j, (fn, mk, off, bs, ce) in enumerate(configs):
            if fn in ('bp', 'td'):
                Xtr = featurize(fn, cues[mask3], bases[mask3], None, None, off, bs, ce)
                Xva = featurize(fn, cues[r3a], bases[r3a], None, None, off, bs, ce)
            else:
                Xtr, state = featurize(fn, cues[mask3], bases[mask3], y[mask3], None, off, bs, ce)
                Xva = featurize(fn, cues[r3a], bases[r3a], None, state, off, bs, ce)
            m = make_pipeline(StandardScaler(), make_model(mk))
            m.fit(Xtr, y[mask3])
            p3[:, j] = m.predict_proba(Xva)[:, 1]
            acc3[j] = accuracy_score(y[r3a], (p3[:, j] >= 0.5).astype(int))
    w_u = np.ones(len(configs))
    w_acc5 = np.clip(acc5 - 0.5, 1e-3, None)
    w_run3 = np.clip(acc3 - 0.5, 1e-3, None) if have3 else None
    w_comb = w_acc5 * (w_run3 if w_run3 is not None else 1.0)
    strategies = {'u': w_u, 'a5': w_acc5, 'a3': (w_run3 if w_run3 is not None else None),
                  'a5a3': (w_comb if w_run3 is not None else None)}
    for k in (5, 6, 8, 10):
        if k < len(configs):
            topk = np.argsort(acc5)[::-1][:k]
            wk = np.zeros(len(configs))
            wk[topk] = 1.0
            strategies['tk%d' % k] = wk
    for tag, fl in (('a5h72', 0.72), ('a5h74', 0.74), ('a5h75', 0.75), ('a5h76', 0.76), ('a5h77', 0.77), ('a5h78', 0.78)):
        wh = np.clip(acc5 - fl, 0.0, None)
        if wh.sum() > 0:
            strategies[tag] = wh
        strategies[tag + 'f'] = wh if wh.sum() > 0 else np.ones(len(acc5))
    for tag, fl in (('a5h_med', float(np.median(acc5))), ('a5h_mean', float(acc5.mean()))):
        wh = np.clip(acc5 - fl, 0.0, None)
        if wh.sum() > 0:
            strategies[tag] = wh
    if os.environ.get('LCMID_2SEED') == '1':
        acc5b = np.array([accuracy_score(y, (oof7[:, j] >= 0.5).astype(int)) for j in range(len(configs))])
        topk = np.argsort((acc5 + acc5b) / 2)[::-1][:5]
        w2 = np.zeros(len(configs)); w2[topk] = 1.0
        P2 = (oof + oof7) / 2 @ (w2 / w2.sum())
        best_t, best_a = 0.5, accuracy_score(y, (P2 >= 0.5).astype(int))
        for t in np.arange(0.40, 0.601, 0.01):
            a = accuracy_score(y, (P2 >= t).astype(int))
            if a > best_a:
                best_a, best_t = a, t
        strategies['tk5_2s'] = w2 / w2.sum()
        strategies['tk5_2s_P'] = P2
        strategies_meta = {'tk5_2s': best_a}
    if have3:
        score = (acc5 + acc3) / 2
        topk = np.argsort(score)[::-1][:5]
        wk = np.zeros(len(configs))
        wk[topk] = 1.0
        strategies['tk5r'] = wk
    # half-split validation of top-k selection (select on odd rows, eval on even rows)
    sel_rows = np.arange(1, len(y), 2)
    ev_rows = np.arange(0, len(y), 2)
    acc5_sel = np.array([accuracy_score(y[sel_rows], (oof[sel_rows, j] >= 0.5).astype(int)) for j in range(len(configs))])
    topk = np.argsort(acc5_sel)[::-1][:6]
    wsel = np.zeros(len(configs)); wsel[topk] = 1.0
    P_ev = oof[ev_rows] @ (wsel / wsel.sum())
    split_a = accuracy_score(y[ev_rows], (P_ev >= 0.5).astype(int))
    split_b = accuracy_score(y[sel_rows], ((oof[sel_rows] @ (wsel / wsel.sum())) >= 0.5).astype(int))
    extra_out = {}
    if os.environ.get('LCMID_2SEED') == '1':
        P2 = strategies.pop('tk5_2s_P')
        extra_out['tk5_2s'] = {'acc5_u50': float(accuracy_score(y, (P2 >= 0.5).astype(int))),
                               'acc5_thr': float(strategies_meta['tk5_2s']), 'run3_thr': None, 'thr': None}
    out = {'acc5': acc5.tolist(), 'acc3': acc3.tolist(), 'tk6_split': {'acc5_u50': float(split_a), 'acc5_thr': None, 'run3_thr': None, 'thr': 0.5, 'both_halves': float((split_a + split_b) / 2)}}
    out.update(extra_out)
    for sn, w in list(strategies.items()):
        if w is None or sn == 'tk5_2s':
            continue
        w = w / w.sum()
        P = oof @ w
        t = best_thr(P, y)
        out[sn] = {'acc5_u50': float(accuracy_score(y, (P >= 0.5).astype(int))),
                   'acc5_thr': float(accuracy_score(y, (P >= t).astype(int))), 'thr': t}
        if p3 is not None:
            Q = p3 @ w
            out[sn]['run3_thr'] = float(accuracy_score(y[r3a], (Q >= t).astype(int)))
    return out

def _w(s):
    return s, eval_subject(s, CONFIGS)

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--subjects', nargs='+', default=None)
    ap.add_argument('--out', default='state/cv_v4_ens.json')
    ap.add_argument('--workers', type=int, default=8)
    args = ap.parse_args()
    subs = args.subjects or TEST_SUBJECTS
    t0 = time.time()
    per = json.load(open(args.out)) if os.path.exists(args.out) else {}
    todo = [s for s in subs if s not in per]
    print('done:', len(per), 'todo:', todo, flush=True)
    if todo:
        if args.workers > 1:
            from concurrent.futures import ProcessPoolExecutor
            with ProcessPoolExecutor(max_workers=args.workers) as ex:
                results = list(ex.map(_w, todo, timeout=None))
        else:
            results = [(s, eval_subject(s, CONFIGS)) for s in todo]
        for s, r in results:
            per[s] = r
            json.dump(per, open(args.out, 'w'), indent=1)
            print('saved', s, flush=True)
    agg = {}
    for s, r in per.items():
        for sn in ('u', 'a5', 'a3', 'a5a3'):
            if sn in r:
                for kk, vv in r[sn].items():
                    a = agg.setdefault(sn + '/' + kk, [])
                    if vv == vv:
                        a.append(vv)
    print('\n=== AGGREGATE (mean over %d subjects) ===' % len(per), flush=True)
    for k in sorted(agg):
        v = agg[k]
        print('%-22s n=%2d  mean=%.4f' % (k, len(v), float(np.mean(v))), flush=True)
    print('elapsed %.0fs' % (time.time() - t0), flush=True)

