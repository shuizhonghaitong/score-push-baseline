"""v4: 3-seed averaged per-variant test preds + ridge blend (weights from seed-0 OOF)."""
import warnings; warnings.filterwarnings('ignore')
import json
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb, xgboost as xgb
from scripts.pipeline_v2 import load_ext, prep, COLS, rmse

ROOT = Path(__file__).resolve().parent.parent
CKPT = ROOT/'models/oof_stack'

tr = pd.read_csv(ROOT/'data/train.csv', dtype={'Zip': str})
te = pd.read_csv(ROOT/'data/test.csv', dtype={'Zip': str})
tr['mon'] = pd.to_datetime(tr['Month']).dt.month
ext = load_ext()

VARIANTS = {
 'lgb63':  dict(kind='lgb', p=dict(num_leaves=63, learning_rate=0.03, min_child_samples=30, reg_lambda=10.0), drop=()),
 'lgb31':  dict(kind='lgb', p=dict(num_leaves=31, learning_rate=0.01, min_child_samples=40, reg_lambda=20.0), drop=()),
 'lgb127': dict(kind='lgb', p=dict(num_leaves=127, learning_rate=0.03, min_child_samples=20, reg_lambda=5.0), drop=()),
 'lgb63nc':dict(kind='lgb', p=dict(num_leaves=63, learning_rate=0.03, min_child_samples=30, reg_lambda=10.0), drop=('City',)),
 'xgb7':   dict(kind='xgb', p=dict(max_depth=7, min_child_weight=30, learning_rate=0.03, reg_lambda=10.0), drop=()),
 'xgb5':   dict(kind='xgb', p=dict(max_depth=5, min_child_weight=50, learning_rate=0.03, reg_lambda=20.0), drop=()),
}
BASE_P = dict(subsample=0.9, colsample_bytree=0.8, random_state=42, n_jobs=16, verbose=-1)
CATS = ['State','Metro','City']
allcats = {c: pd.CategoricalDtype(np.sort(pd.unique(pd.concat([tr[c], te[c]]).fillna('__NA__'))).tolist()) for c in CATS}
def prep2(df, e, drop=()):
    d = prep(df, e)
    for c in drop: d = d.drop(columns=[c])
    for c, dt in allcats.items():
        if c in d.columns: d[c] = d[c].astype(dt)
    return d

old = np.load(CKPT/'test_preds.npz')
test_avg = {n: old[n].copy() for n in VARIANTS}   # seed 0
for seed in (1, 2):
    for name, v in VARIANTS.items():
        zs = np.sort(tr.Zip.unique()); r = np.random.RandomState(seed); r.shuffle(zs)
        folds = np.array_split(zs, 5)
        tpred = np.zeros(len(te))
        for fi, f in enumerate(folds):
            val = tr[tr.Zip.isin(set(f))]; trn = tr[~tr.Zip.isin(set(f))]
            ev = val[val.mon==6]
            drop = v.get('drop', ())
            cols = [c for c in COLS if c not in drop]
            dtrn, dev, dte = prep2(trn, ext, drop), prep2(ev, ext, drop), prep2(te, ext, drop)
            if v['kind'] == 'lgb':
                m = lgb.LGBMRegressor(objective='regression', metric='rmse', n_estimators=6000,
                    categorical_feature=[c for c in CATS if c not in drop], **BASE_P, **v['p'])
                m.fit(dtrn[cols], trn.AffordabilityPercentageTrue.values, eval_set=[(dev[cols], ev.AffordabilityPercentageTrue.values)],
                      callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(0)])
                tpred += m.predict(dte[cols])
            else:
                m = xgb.XGBRegressor(n_estimators=6000, subsample=0.9, colsample_bytree=0.8, random_state=42,
                    n_jobs=16, tree_method='hist', eval_metric='rmse', enable_categorical=True, early_stopping_rounds=200, **v['p'])
                m.fit(dtrn[cols], trn.AffordabilityPercentageTrue.values, eval_set=[(dev[cols], ev.AffordabilityPercentageTrue.values)], verbose=False)
                tpred += m.predict(dte[cols], iteration_range=(0, m.best_iteration+1))
            print(name, seed, fi, flush=True)
        test_avg[name] += tpred/5
Xt = np.column_stack([test_avg[n] for n in VARIANTS])
np.save(CKPT/'test_3seed.npz', Xt)
summ = json.load(open(CKPT/'summary.json'))
coef = np.array(summ['coef'])
blend = Xt @ coef + (0.0)  # ridge intercept from summary? recompute below
# recompute intercept from OOF
oof = json.load(open(CKPT/'oof.json'))
evz = np.array([z for z in tr[tr.mon==6].Zip])  # order not needed if we align via dict
# rebuild OOF matrix in same column order as Xt
import numpy as _np
y6 = tr[tr.mon==6].set_index('Zip').AffordabilityPercentageTrue
zall = np.sort(tr[tr.mon==6].Zip.unique())
Xo = np.column_stack([_np.array([oof[n].get(z, _np.nan) for z in zall]) for n in VARIANTS])
yo = y6.loc[zall].values
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
Xo = SimpleImputer().fit_transform(Xo)
rg = Ridge(alpha=summ['alpha'], fit_intercept=True).fit(Xo, yo)
blend = Xt @ rg.coef_ + rg.intercept_
print('blend coef', rg.coef_.round(3), 'intercept', round(rg.intercept_,3))
np.save(CKPT/'blend_3seed.npy', blend)
# assemble final v4 submission
may = tr[tr.mon==5].set_index('Zip')['AffordabilityPercentageTrue']
jun = tr[tr.mon==6].set_index('Zip')['AffordabilityPercentageTrue']
idx = may.index.intersection(jun.index)
a, b = may.loc[idx].values, jun.loc[idx].values
slope = np.cov(a,b)[0,1]/np.var(a); intercept = b.mean()-slope*a.mean()
out = te[['Zip','State']].copy()
is_seen = out.Zip.isin(jun.index)
pred = np.empty(len(out))
pred[is_seen.values] = slope*jun.loc[out.Zip[is_seen.values]].values + intercept
pred[~is_seen.values] = blend[~is_seen.values]
out['AffordabilityPercentageTrue'] = np.clip(pred, 0, 100)
out['Zip'] = out.Zip.str.zfill(5)
out.to_csv(ROOT/'subs/v4_stack.csv', index=False)
seen_rmse = rmse(b, slope*a+intercept)
new_rmse = summ['blend_oof']
print(f'v4: seen {seen_rmse:.3f} new(OOF) {new_rmse:.3f} comp {0.2*seen_rmse+0.8*new_rmse:.3f}')
