"""Per-subject CV evaluation of feature x model configs (ML track)."""
import sys, os, time, json, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_subject, TEST_SUBJECTS
import features as F
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

CFG = {'bands': F.BANDS, 'csp_bands': ((8, 12), (12, 16)), 'off': 0.0}
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score

def make_model(kind):
    if kind.startswith('lda_'):
        return LinearDiscriminantAnalysis(solver='lsqr', shrinkage=float(kind[4:]))
    if kind.startswith('lr_'):
        return LogisticRegression(C=float(kind[3:]), max_iter=2000)
    if kind.startswith('ridge_'):
        return RidgeClassifier(alpha=float(kind[6:]))
    raise ValueError(kind)

def static_featurize(name, cues, bases):
    B, OF = CFG['bands'], CFG['off']
    if name in ('bp', 'td', 'bp_td'):
        if name == 'bp':
            return F.band_power_features(cues, bases, bands=B, off=OF)
        if name == 'td':
            return F.timdomain_features(cues, off=OF)
        return np.concatenate([F.band_power_features(cues, bases, bands=B, off=OF), F.timdomain_features(cues, off=OF)], 1)
    return None

def featurize(name, cues, bases, y, state):
    """state: None on fit (returns (X, state)); existing state on transform (returns X)."""
    B, CB, OF = CFG['bands'], CFG['csp_bands'], CFG['off']
    if name == 'csp':
        if state is None:
            x, filt = F.csp_features(cues, y, bands=CB, off=OF)
            return x, filt
        return F.csp_transform(cues, state, bands=CB, off=OF)
    if name == 'rts':
        if state is None:
            x, ref = F.riemannian_features(cues, bands=B, off=OF)
            return x, ref
        x, _ = F.riemannian_features(cues, bands=B, ref=state, off=OF)
        return x
    if name == 'bp_rts':
        if state is None:
            x, ref = F.riemannian_features(cues, bands=B, off=OF)
            return np.concatenate([F.band_power_features(cues, bases, bands=B, off=OF), x], 1), ref
        x, _ = F.riemannian_features(cues, bands=B, ref=state, off=OF)
        return np.concatenate([F.band_power_features(cues, bases, bands=B, off=OF), x], 1)
    if name == 'bp_csp':
        if state is None:
            c, filt = F.csp_features(cues, y, bands=CB, off=OF)
            return np.concatenate([F.band_power_features(cues, bases, bands=B, off=OF), c], 1), filt
        return np.concatenate([F.band_power_features(cues, bases, bands=B, off=OF), F.csp_transform(cues, state, bands=CB, off=OF)], 1)
    if name == 'all':
        if state is None:
            x, ref = F.riemannian_features(cues, bands=B, off=OF)
            c, filt = F.csp_features(cues, y, bands=CB, off=OF)
            return np.concatenate([F.band_power_features(cues, bases, bands=B, off=OF), x, c], 1), (ref, filt)
        ref, filt = state
        x, _ = F.riemannian_features(cues, bands=B, ref=ref, off=OF)
        c = F.csp_transform(cues, filt, bands=CB, off=OF)
        return np.concatenate([F.band_power_features(cues, bases, bands=B, off=OF), x, c], 1)
    raise ValueError(name)

def fit_predict_proba(name, kind, cues, bases, y, Xstatic, tr_i, va_i):
    if name in ('bp', 'td', 'bp_td'):
        Xtr, Xva = Xstatic[tr_i], Xstatic[va_i]
    else:
        Xtr, state = featurize(name, cues[tr_i], bases[tr_i], y[tr_i], None)
        Xva = featurize(name, cues[va_i], bases[va_i], None, state)
    clf = make_pipeline(StandardScaler(), make_model(kind))
    clf.fit(Xtr, y[tr_i])
    return clf.predict_proba(Xva)[:, 1]

def eval_subject(sub, feat_names, model_kinds, n_folds=5, seed=42):
    d = load_subject(sub)
    runs = [k for k in ('tr01', 'tr02', 'tr03') if k in d]
    cues = np.concatenate([d[k][0] for k in runs])
    bases = np.concatenate([d[k][1] for k in runs])
    y = np.concatenate([d[k][2] for k in runs])
    pos = 0
    r3 = []
    for k in runs:
        n = len(d[k][2])
        if k == 'tr03':
            r3 = list(range(pos, pos + n))
        pos += n
    r3 = np.array(r3) if r3 else np.array([], int)
    mask3 = np.ones(len(y), bool)
    mask3[r3] = False

    Xtr_static, Xva_static = {}, {}
    for fn in feat_names:
        xs = static_featurize(fn, cues, bases)
        if xs is not None:
            Xtr_static[fn] = xs
            Xva_static[fn] = xs

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    res = {}
    for fn in feat_names:
        for mk in model_kinds:
            oof = np.zeros(len(y))
            for tr_i, va_i in skf.split(cues, y):
                oof[va_i] = fit_predict_proba(fn, mk, cues, bases, y, Xtr_static.get(fn), tr_i, va_i)
            acc5 = accuracy_score(y, (oof >= 0.5).astype(int))
            if len(r3) >= 4 and np.bincount(y[r3]).min() >= 2:
                p = fit_predict_proba(fn, mk, cues, bases, y, Xtr_static.get(fn), mask3, r3)
                acc3 = accuracy_score(y[r3], (p >= 0.5).astype(int))
            else:
                acc3 = np.nan
            res[f'{fn}|{mk}'] = (float(acc5), float(acc3))
    return res

def _worker(sub_feats_models_folds):
    sub, feats, models, folds = sub_feats_models_folds
    return sub, eval_subject(sub, feats, models, folds)

if __name__ == '__main__':
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    ap = argparse.ArgumentParser()
    ap.add_argument('--feats', nargs='+', default=['bp', 'rts', 'csp', 'td', 'bp_rts', 'bp_csp', 'all'])
    ap.add_argument('--models', nargs='+', default=['lda_0', 'lda_0.1', 'lda_0.3', 'lda_0.5', 'lr_0.1', 'lr_1', 'lr_10'])
    ap.add_argument('--subjects', nargs='+', default=None)
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--out', default='cv_results.json')
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--offset', type=float, default=0.0)
    ap.add_argument('--bands', choices=['6', '7'], default='6')
    ap.add_argument('--csp-bands', choices=['2', '3'], default='2')
    args = ap.parse_args()
    CFG['off'] = args.offset
    CFG['bands'] = F.BANDS7 if args.bands == '7' else F.BANDS
    CFG['csp_bands'] = ((8, 12), (12, 16), (16, 20)) if args.csp_bands == '3' else ((8, 12), (12, 16))
    subs = args.subjects or TEST_SUBJECTS
    t0 = time.time()
    per = {}
    if os.path.exists(args.out):
        per = json.load(open(args.out)).get('per', {})
    todo = [s for s in subs if s not in per]
    print('done:', len(per), 'todo:', todo, flush=True)
    if todo:
        if args.workers > 1:
            from concurrent.futures import ProcessPoolExecutor
            with ProcessPoolExecutor(max_workers=args.workers) as ex:
                results = list(ex.map(_worker, [(s, args.feats, args.models, args.folds) for s in todo]))
        else:
            results = [(s, eval_subject(s, args.feats, args.models, args.folds)) for s in todo]
        for s, r in results:
            per[s] = r
            json.dump({'per': per, 'feats': args.feats, 'models': args.models}, open(args.out, 'w'))
            print('saved', s, flush=True)
    agg = {}
    for s, r in results:
        for k, (a5, a3) in r.items():
            a = agg.setdefault(k, [0.0, 0.0, 0])
            a[0] += a5
            if not np.isnan(a3):
                a[1] += a3
                a[2] += 1
    print('\n=== AGGREGATE (mean over subjects) ===', flush=True)
    rows = []
    for k, (s5, s3, n3) in agg.items():
        rows.append((k, s5 / len(subs), s3 / n3 if n3 else float('nan'), n3))
    rows.sort(key=lambda r: -r[1])
    for k, a5, a3, n3 in rows:
        print(f'{k:18s} 5fold={a5:.4f}  run3={a3:.4f} (n={n3})', flush=True)
    json.dump({'per': per, 'agg': agg, 'feats': args.feats, 'models': args.models}, open(args.out, 'w'), indent=1)
    print('elapsed %.0fs' % (time.time() - t0), flush=True)
