"""Modeling core: feature prep with per-fit target encodings, LGBM training, geo-CV."""
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import lightgbm as lgb
from features import NUM_FEATS, TARGET, add_derived
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BASE = ['SampleSize','RentIndex','AnnualMedianHouseholdIncome','OwnerPercent','RenterPercent',
        'MedianHomeLengthofResidence','AvgNumberofChildren','AvgNumberofAdults','AvgHouseholdSize',
        'AvgGenerationsinHousehold','YoungAdultinHouseholdPercent','SeniorAdultinHouseholdPercent',
        'RentIncomeRatio','LogRent','LogIncome','LogN','InvSqrtN']
TE_COLS = {'Metro': 200.0, 'City': 50.0, 'State': 1e9}   # smoothing weights
CATS = ['State', 'Metro', 'City']

def load():
    tr = pd.read_csv(ROOT/'data/train.csv', dtype={'Zip': str})
    te = pd.read_csv(ROOT/'data/test.csv', dtype={'Zip': str})
    tr['mon'] = pd.to_datetime(tr['Month']).dt.month
    return tr, te

def prep_base(df):
    d = add_derived(df)
    d['RentIncomeRatio'] = (d.RentIndex*12.0)/d.AnnualMedianHouseholdIncome.replace(0, np.nan)
    # within-state z-scores
    for c, col in [('RentZst','RentIndex'), ('IncZst','AnnualMedianHouseholdIncome'), ('OwnerZst','OwnerPercent')]:
        g = d.groupby('State')[col]
        d[c] = (d[col] - g.transform('mean'))/g.transform('std').replace(0, np.nan)
    for c in CATS:
        d[c] = d[c].fillna('__NA__').astype('category')
    return d

def target_encode(trn: pd.DataFrame, val: pd.DataFrame, col: str, sm: float, global_mean: float):
    """Mean target of col groups in trn, shrinkage-blended with global mean."""
    stats = trn.groupby(col)[TARGET].agg(['mean','count'])
    w = sm/(sm + stats['count'])
    enc = w*global_mean + (1-w)*stats['mean']
    out = val[col].map(enc).fillna(global_mean)
    return out.values

def build_X(trn, val, cols=None):
    cols = cols or (BASE + ['RentZst','IncZst','OwnerZst'] + [f'TE_{c}' for c in CATS])
    dval = prep_base(val)
    X = dval[BASE + ['RentZst','IncZst','OwnerZst']].copy()
    gm = trn[TARGET].mean()
    for c in CATS:
        X[f'TE_{c}'] = target_encode(trn, dval, c, TE_COLS[c], gm)
    X[CATS] = dval[CATS]
    return X[cols], dval

def lgb_params(**kw):
    p = dict(objective='regression', metric='rmse', learning_rate=0.03, num_leaves=127, max_depth=-1,
             min_child_samples=25, subsample=0.9, subsample_freq=1, colsample_bytree=0.8,
             reg_lambda=15.0, verbose=-1, n_estimators=4000, random_state=42, n_jobs=16,
             categorical_feature=CATS)
    p.update(kw)
    return p

def rmse(y, p): return float(np.sqrt(np.mean((np.asarray(y,float)-np.asarray(p,float))**2)))

def geo_cv(tr, n_folds=5, seed=0, params=None, use_june=True):
    zips = np.sort(tr['Zip'].unique())
    rng = np.random.RandomState(seed)
    rng.shuffle(zips)
    errs, preds = [], {}
    for f in np.array_split(zips, n_folds):
        val = tr[tr['Zip'].isin(set(f))]
        trn = tr[~tr['Zip'].isin(set(f))]
        ev = val[val.mon==6] if use_june else val
        Xtr, _ = build_X(trn, trn)
        Xv, _ = build_X(trn, ev)
        m = lgb.LGBMRegressor(**(params or lgb_params()))
        m.fit(Xtr, trn[TARGET].values, eval_set=[(Xv, ev[TARGET].values)],
              callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(0)])
        p = m.predict(Xv)
        errs.append(rmse(ev[TARGET].values, p))
        preds.update(zip(ev['Zip'], p))
    return errs, preds

if __name__ == '__main__':
    tr, te = load()
    errs, preds = geo_cv(tr)
    print('geo-CV new-ZIP proxy RMSE per fold:', [round(e,3) for e in errs], 'mean =', round(np.mean(errs),3))
    np.save('models/oof_new.npy', np.array([preds[z] for z in sorted(preds)]))
