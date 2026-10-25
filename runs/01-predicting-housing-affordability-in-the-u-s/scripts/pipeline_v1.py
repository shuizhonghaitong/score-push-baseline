"""v1: seen -> June target (+ drift correction), new -> LightGBM geo-CV ensemble."""
import warnings; warnings.filterwarnings('ignore')
import json, time
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb

ROOT = Path(__file__).resolve().parent.parent
FEAT = ['SampleSize','RentIndex','AnnualMedianHouseholdIncome','OwnerPercent','RenterPercent',
        'MedianHomeLengthofResidence','AvgNumberofChildren','AvgNumberofAdults','AvgHouseholdSize',
        'AvgGenerationsinHousehold','YoungAdultinHouseholdPercent','SeniorAdultinHouseholdPercent',
        'RentIncomeRatio','LogRent','LogIncome','LogN','InvSqrtN','RentZst','IncZst','OwnerZst']
CATS = ['State','Metro','City']

def load():
    tr = pd.read_csv(ROOT/'data/train.csv', dtype={'Zip': str})
    te = pd.read_csv(ROOT/'data/test.csv', dtype={'Zip': str})
    tr['mon'] = pd.to_datetime(tr['Month']).dt.month
    return tr, te

def prep(df):
    d = df.copy()
    d['RentIncomeRatio'] = (d.RentIndex*12.0)/d.AnnualMedianHouseholdIncome.replace(0, np.nan)
    d['LogRent'] = np.log1p(d.RentIndex.clip(lower=0))
    d['LogIncome'] = np.log1p(d.AnnualMedianHouseholdIncome.clip(lower=0))
    d['LogN'] = np.log1p(d.SampleSize)
    d['InvSqrtN'] = 1.0/np.sqrt(d.SampleSize)
    for c, col in [('RentZst','RentIndex'), ('IncZst','AnnualMedianHouseholdIncome'), ('OwnerZst','OwnerPercent')]:
        g = d.groupby('State')[col]
        d[c] = (d[col]-g.transform('mean'))/g.transform('std').replace(0, np.nan)
    for c in CATS:
        d[c] = d[c].fillna('__NA__').astype('category')
    return d

def rmse(y, p): return float(np.sqrt(np.mean((np.asarray(y,float)-np.asarray(p,float))**2)))

def train_folds(tr, te, n_folds=5, seeds=(0,1,2), params=None, save_dir=ROOT/'models'):
    """ZIP-level CV; returns (oof_new_preds for test zip, list of models, fold errs)."""
    p = dict(objective='regression', metric='rmse', learning_rate=0.03, num_leaves=63,
             min_child_samples=30, subsample=0.9, subsample_freq=1, colsample_bytree=0.8,
             reg_lambda=10.0, verbose=-1, n_estimators=4000, random_state=42, n_jobs=16,
             categorical_feature=CATS)
    p.update(params or {})
    test_pred = np.zeros(len(te))
    all_errs = []
    models = []
    for seed in seeds:
        zips = np.sort(tr['Zip'].unique())
        rng = np.random.RandomState(seed); rng.shuffle(zips)
        folds = np.array_split(zips, n_folds)
        pred_seed = np.zeros(len(te))
        for f in folds:
            val = tr[tr['Zip'].isin(set(f))]; trn = tr[~tr['Zip'].isin(set(f))]
            ev = val[val.mon==6]
            Xtr, Xv = prep(trn)[FEAT+CATS], prep(ev)[FEAT+CATS]
            Xte = prep(te)[FEAT+CATS]
            m = lgb.LGBMRegressor(**p)
            m.fit(Xtr, trn['AffordabilityPercentageTrue'].values,
                  eval_set=[(Xv, ev['AffordabilityPercentageTrue'].values)],
                  callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(0)])
            all_errs.append(rmse(ev['AffordabilityPercentageTrue'].values, m.predict(Xv)))
            pred_seed += m.predict(Xte)
            models.append(m)
        test_pred += pred_seed
    test_pred /= (n_folds*len(seeds))
    return test_pred, all_errs, models

def seen_fit(tr):
    may = tr[tr.mon==5].set_index('Zip')['AffordabilityPercentageTrue']
    jun = tr[tr.mon==6].set_index('Zip')['AffordabilityPercentageTrue']
    idx = may.index.intersection(jun.index)
    a, b = may.loc[idx].values, jun.loc[idx].values
    # linear: b = slope*a + intercept
    slope = np.cov(a,b)[0,1]/np.var(a)
    intercept = b.mean() - slope*a.mean()
    return slope, intercept

def main():
    t0 = time.time()
    tr, te = load()
    slope, intercept = seen_fit(tr)
    print('seen drift fit: slope', round(slope,4), 'intercept', round(intercept,3))
    seen_pred, errs, models = train_folds(tr, te, n_folds=5, seeds=(0,1,2))
    print('geo-CV new RMSE mean:', round(np.mean(errs),3), 'time', round(time.time()-t0,1))
    jun = tr[tr.mon==6].set_index('Zip')['AffordabilityPercentageTrue']
    out = te[['Zip','State']].copy()
    is_seen = out.Zip.isin(jun.index)
    pred = np.empty(len(out))
    pred[is_seen.values] = slope*jun.loc[out.Zip[is_seen.values]].values + intercept
    pred[~is_seen.values] = seen_pred[~is_seen.values]
    out['AffordabilityPercentageTrue'] = np.clip(pred, 0, 100)
    out['Zip'] = out.Zip.str.zfill(5)
    out.to_csv(ROOT/'subs/v1_baseline.csv', index=False)
    print(out.head(3).to_string())
    print('seen count:', is_seen.sum(), 'new count:', (~is_seen).sum())
    # local composite proxy
    may = tr[tr.mon==5].set_index('Zip')['AffordabilityPercentageTrue']
    seen_rmse = rmse(jun.values, slope*may.values+intercept)
    print('LOCAL PROXY: seen', round(seen_rmse,3), 'new', round(np.mean(errs),3),
          'composite', round(0.2*seen_rmse+0.8*np.mean(errs),3))

if __name__ == '__main__':
    main()
