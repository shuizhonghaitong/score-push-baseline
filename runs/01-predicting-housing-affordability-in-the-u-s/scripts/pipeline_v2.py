"""v2: new-ZIP LGBM with external medianHomeValue features + derived owner-cost ratios.
seen: drift-corrected June target."""
import warnings; warnings.filterwarnings('ignore')
import json, time
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb

ROOT = Path(__file__).resolve().parent.parent

def load_ext():
    cyb = pd.read_csv(ROOT/'external/us-zip-codes-demographics/zips.csv', dtype={'zip': str})
    cyb['zip'] = cyb['zip'].str.zfill(5)
    return cyb.set_index('zip')

KEY = ['medianHomeValue','perCapitaIncome','populationDensity','medianAge','ownerOccupiedPct','housingUnits','population','medianGrossRent']
BASE = ['SampleSize','RentIndex','AnnualMedianHouseholdIncome','OwnerPercent','RenterPercent',
        'MedianHomeLengthofResidence','AvgNumberofChildren','AvgNumberofAdults','AvgHouseholdSize',
        'AvgGenerationsinHousehold','YoungAdultinHouseholdPercent','SeniorAdultinHouseholdPercent',
        'ratio','LogRent','LogIncome','State','Metro','City']
RATES = (0.04,0.05,0.06,0.07,0.08)
DERIV = [f'ocr{int(r*1000)}' for r in RATES] + ['rentRatio','LogHV','HVVIncome','burden6','burden6_log']

def prep(df, ext):
    d = df.merge(ext, left_on='Zip', right_index=True, how='left')
    d['ratio'] = (d.RentIndex*12.0)/d.AnnualMedianHouseholdIncome.replace(0, np.nan)
    d['LogRent'] = np.log1p(d.RentIndex.clip(0)); d['LogIncome'] = np.log1p(d.AnnualMedianHouseholdIncome.clip(0))
    mon_inc = (d.AnnualMedianHouseholdIncome/12.0).replace(0, np.nan)
    for r in RATES:
        d[f'ocr{int(r*1000)}'] = d.medianHomeValue*r/mon_inc
    d['rentRatio'] = d['ratio']
    d['LogHV'] = np.log1p(d.medianHomeValue.clip(0))
    d['HVVIncome'] = d.medianHomeValue/d.AnnualMedianHouseholdIncome.replace(0, np.nan)
    d['burden6'] = d.OwnerPercent/100.0*d['ocr60'] + d.RenterPercent/100.0*d['ratio']
    d['burden6_log'] = np.log1p(d.burden6.clip(lower=0))
    for c in ['State','Metro','City']: d[c] = d[c].fillna('__NA__').astype('category')
    return d

COLS = BASE + KEY + DERIV

def rmse(y,p): return float(np.sqrt(np.mean((np.asarray(y,float)-np.asarray(p,float))**2)))

def main():
    t0 = time.time()
    tr = pd.read_csv(ROOT/'data/train.csv', dtype={'Zip': str})
    te = pd.read_csv(ROOT/'data/test.csv', dtype={'Zip': str})
    tr['mon'] = pd.to_datetime(tr['Month']).dt.month
    ext = load_ext()
    # seen drift fit
    may = tr[tr.mon==5].set_index('Zip')['AffordabilityPercentageTrue']
    jun = tr[tr.mon==6].set_index('Zip')['AffordabilityPercentageTrue']
    idx = may.index.intersection(jun.index)
    a, b = may.loc[idx].values, jun.loc[idx].values
    slope = np.cov(a,b)[0,1]/np.var(a); intercept = b.mean()-slope*a.mean()
    print('seen drift: slope', round(slope,4), 'intercept', round(intercept,4))

    seeds = (0,1,2,3,4)
    total = np.zeros(len(te)); errs = []
    p = dict(objective='regression', metric='rmse', learning_rate=0.03, num_leaves=63,
             min_child_samples=30, subsample=0.9, subsample_freq=1, colsample_bytree=0.8,
             reg_lambda=10.0, verbose=-1, n_estimators=4000, random_state=42, n_jobs=16,
             categorical_feature=['State','Metro','City'])
    for seed in seeds:
        zs = np.sort(tr.Zip.unique()); r = np.random.RandomState(seed); r.shuffle(zs)
        for f in np.array_split(zs, 5):
            val = tr[tr.Zip.isin(set(f))]; trn = tr[~tr.Zip.isin(set(f))]
            ev = val[val.mon==6]
            dtrn, dev, dte = prep(trn, ext), prep(ev, ext), prep(te, ext)
            m = lgb.LGBMRegressor(**p)
            m.fit(dtrn[COLS], trn.AffordabilityPercentageTrue.values,
                  eval_set=[(dev[COLS], ev.AffordabilityPercentageTrue.values)],
                  callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(0)])
            errs.append(rmse(ev.AffordabilityPercentageTrue.values, m.predict(dev[COLS])))
            total += m.predict(dte[COLS])
            if len(errs) % 5 == 0: print(f'  seed {seed} done, err so far {np.mean(errs):.3f}', flush=True)
    total /= (5*len(seeds))
    new_rmse = float(np.mean(errs))
    print('new geo-CV RMSE:', round(new_rmse,3))

    out = te[['Zip','State']].copy()
    is_seen = out.Zip.isin(jun.index)
    pred = np.empty(len(out))
    pred[is_seen.values] = slope*jun.loc[out.Zip[is_seen.values]].values + intercept
    pred[~is_seen.values] = total[~is_seen.values]
    out['AffordabilityPercentageTrue'] = np.clip(pred, 0, 100)
    out['Zip'] = out.Zip.str.zfill(5)
    out.to_csv(ROOT/'subs/v2_ext.csv', index=False)
    seen_rmse = rmse(jun.loc[idx].values, slope*may.loc[idx].values+intercept)
    comp = 0.2*seen_rmse + 0.8*new_rmse
    print(f'LOCAL PROXY: seen {seen_rmse:.3f} new {new_rmse:.3f} composite {comp:.3f}')
    print('time', round(time.time()-t0,1))
    json.dump({'seen_rmse': seen_rmse, 'new_rmse': new_rmse, 'composite': comp,
               'per_seed': [round(float(np.mean(errs[i*5:(i+1)*5]),3)) for i in range(len(seeds))]},
              open(ROOT/'models/v2_cv.json','w'), indent=1)

if __name__ == '__main__':
    main()
