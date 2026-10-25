"""OOF stacking: several diverse models on geo folds; ridge-blend validated on OOF."""
import warnings; warnings.filterwarnings('ignore')
import json
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb, xgboost as xgb
from sklearn.linear_model import Ridge
from scripts.pipeline_v2 import load_ext, prep, COLS, rmse

ROOT = Path(__file__).resolve().parent.parent
CKPT = ROOT/'models/oof_stack'
CKPT.mkdir(parents=True, exist_ok=True)

tr = pd.read_csv(ROOT/'data/train.csv', dtype={'Zip': str})
te = pd.read_csv(ROOT/'data/test.csv', dtype={'Zip': str})
tr['mon'] = pd.to_datetime(tr['Month']).dt.month
ext = load_ext()

VARIANTS = {
 'lgb63':  dict(kind='lgb', p=dict(num_leaves=63, learning_rate=0.03, min_child_samples=30, reg_lambda=10.0)),
 'lgb31':  dict(kind='lgb', p=dict(num_leaves=31, learning_rate=0.01, min_child_samples=40, reg_lambda=20.0)),
 'lgb127': dict(kind='lgb', p=dict(num_leaves=127, learning_rate=0.03, min_child_samples=20, reg_lambda=5.0)),
 'lgb63nc':dict(kind='lgb', p=dict(num_leaves=63, learning_rate=0.03, min_child_samples=30, reg_lambda=10.0), drop=['City']),
 'xgb7':   dict(kind='xgb', p=dict(max_depth=7, min_child_weight=30, learning_rate=0.03, reg_lambda=10.0)),
 'xgb5':   dict(kind='xgb', p=dict(max_depth=5, min_child_weight=50, learning_rate=0.03, reg_lambda=20.0)),
}
BASE_P = dict(subsample=0.9, colsample_bytree=0.8, random_state=42, n_jobs=16, verbose=-1)
CATS = ['State','Metro','City']
zips = np.sort(tr.Zip.unique()); r = np.random.RandomState(0); r.shuffle(zs0 := zips)
folds = np.array_split(zs0, 5)

allcats = {c: pd.CategoricalDtype(np.sort(pd.unique(pd.concat([tr[c], te[c]]).fillna('__NA__'))).tolist()) for c in CATS}
def prep2(df, e, drop=()):
    d = prep(df, e)
    for c in drop: d = d.drop(columns=[c])
    for c, dt in allcats.items():
        if c in d.columns: d[c] = d[c].astype(dt)
    return d

oof_store = {}
test_store = {}
for name, v in VARIANTS.items():
    oof = {}
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
            p, pte = m.predict(dev[cols]), m.predict(dte[cols])
        else:
            m = xgb.XGBRegressor(n_estimators=6000, subsample=0.9, colsample_bytree=0.8, random_state=42,
                n_jobs=16, tree_method='hist', eval_metric='rmse', enable_categorical=True, early_stopping_rounds=200, **v['p'])
            m.fit(dtrn[cols], trn.AffordabilityPercentageTrue.values, eval_set=[(dev[cols], ev.AffordabilityPercentageTrue.values)], verbose=False)
            p = m.predict(dev[cols], iteration_range=(0, m.best_iteration+1))
            pte = m.predict(dte[cols], iteration_range=(0, m.best_iteration+1))
        for z, pv in zip(ev.Zip, p): oof[z] = float(pv)
        tpred += pte
        print(name, 'fold', fi, 'done', flush=True)
    oof_store[name] = oof
    test_store[name] = tpred/5
    json.dump({n: oof_store[n] for n in oof_store}, open(CKPT/'oof.json','w'))
    np.savez(CKPT/'test_preds.npz', **{n: test_store[n] for n in test_store})

# evaluate
evz = []
for f in folds:
    evz += list(np.intersect1d(f, tr[tr.mon==6].Zip.unique()))
y = tr[tr.mon==6].set_index('Zip').AffordabilityPercentageTrue
Xo = np.column_stack([np.array([oof_store[n].get(z, np.nan) for z in evz]) for n in VARIANTS])
yo = y.loc[evz].values
for n in VARIANTS:
    print(n, 'oof rmse', round(rmse(yo, Xo[:, list(VARIANTS).index(n)]),3))
from sklearn.impute import SimpleImputer
Xo = SimpleImputer().fit_transform(Xo)
best = None
for a in [0, 1, 10, 100]:
    rg = Ridge(alpha=a, fit_intercept=True).fit(Xo, yo)
    e = rmse(yo, rg.predict(Xo))
    print(f'ridge a={a}: oof rmse {e:.3f} coefs', np.round(rg.coef_,3))
    if best is None or e < best[0]: best = (e, a, rg)
e, a, rg = best
print('best blend a=', a, 'oof rmse', round(e,3))
# final test prediction from blend
Xt = np.column_stack([test_store[n] for n in VARIANTS])
blend_test = rg.predict(Xt)
np.save(CKPT/'blend_test.npy', blend_test)
voof = {n: round(rmse(yo, np.array([oof_store[n].get(z, np.nan) for z in evz])), 3) for n in VARIANTS}
json.dump({'variant_oof': voof, 'blend_oof': round(e,3), 'alpha': a, 'coef': rg.coef_.tolist()},
          open(CKPT/'summary.json','w'), indent=1)
