"""Local CV harness simulating the 80/20 seen/new split."""
import warnings
warnings.filterwarnings('ignore')
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from features import NUM_FEATS, TARGET, add_derived

ROOT = Path(__file__).resolve().parent.parent
DERIVED = ['RentMonthly_x12','RentIncomeRatio','RentIncomeRatio30','LogRent','LogIncome',
           'OwnerXIncome','InvSqrtN','LogN','ChildrenXIncome','AdultsXIncome','ResidenceXOwner','SeniorXIncome']
FEATS = NUM_FEATS + DERIVED
CATS = ['State','Metro']

def load():
    tr = pd.read_csv(ROOT/'data/train.csv', dtype={'Zip': str})
    te = pd.read_csv(ROOT/'data/test.csv', dtype={'Zip': str})
    tr['mon'] = pd.to_datetime(tr['Month']).dt.month
    return tr, te

def prep(df):
    d = add_derived(df)
    for c in CATS:
        d[c] = d[c].fillna('__NA__').astype('category')
    return d[FEATS + CATS]

def lgb_params(**kw):
    p = dict(objective='regression', metric='rmse', learning_rate=0.03, num_leaves=63,
             max_depth=-1, min_child_samples=30, subsample=0.9, subsample_freq=1,
             colsample_bytree=0.8, reg_lambda=10.0, verbose=-1, n_estimators=3000,
             random_state=42, n_jobs=8, categorical_feature=CATS)
    p.update(kw)
    return p

def rmse(y, p): return float(np.sqrt(np.mean((np.asarray(y, float)-np.asarray(p, float))**2)))

def new_zip_cv(tr, n_folds=5, seed=0, params=None, verbose=True):
    zips = np.sort(tr['Zip'].unique())
    rng = np.random.RandomState(seed)
    rng.shuffle(zips)
    folds = np.array_split(zips, n_folds)
    errs = []
    for f in folds:
        val = tr[tr['Zip'].isin(set(f))]
        trn = tr[~tr['Zip'].isin(set(f))]
        m = lgb.LGBMRegressor(**(params or lgb_params()))
        m.fit(prep(trn), trn[TARGET].values, eval_set=[(prep(val[val.mon==6]), val[val.mon==6][TARGET].values)],
              callbacks=[lgb.early_stopping(150, verbose=False), lgb.log_evaluation(0)])
        p = m.predict(prep(val[val.mon==6]))
        errs.append(rmse(val[val.mon==6][TARGET].values, p))
        if verbose: print(f'  fold best_iter={m.best_iteration_} rmse={errs[-1]:.3f}')
    return errs

if __name__ == '__main__':
    tr, te = load()
    errs = new_zip_cv(tr)
    print('new-ZIP proxy RMSE mean =', round(np.mean(errs),3))
