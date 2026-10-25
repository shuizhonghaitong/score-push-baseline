"""Geo-CV with external features; prints RMSE per config."""
import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, lightgbm as lgb, json, sys

tr = pd.read_csv('data/train.csv', dtype={'Zip': str})
te = pd.read_csv('data/test.csv', dtype={'Zip': str})
tr['mon'] = pd.to_datetime(tr['Month']).dt.month

cyb = pd.read_csv('external/us-zip-codes-demographics/zips.csv', dtype={'zip': str})
cyb['zip'] = cyb['zip'].str.zfill(5)
CYB = ['perCapitaIncome','medianHomeValue','medianGrossRent','populationDensity','medianAge','ownerOccupiedPct','housingUnits','population']

gr = pd.read_csv('external/acs-gross-rent-us-statistics/kaggle_gross_rent.csv', dtype={'Zip_Code': str}, encoding='latin-1', low_memory=False)
inc = pd.read_csv('external/us-household-income-stats-geo-locations/kaggle_income.csv', dtype={'Zip_Code': str}, encoding='latin-1', low_memory=False)
gr['Zip_Code'] = gr['Zip_Code'].str.zfill(5)
gr = gr.sort_values('Samples', ascending=False).drop_duplicates('Zip_Code', keep='first')
inc['Zip_Code'] = inc['Zip_Code'].str.zfill(5)
inc = inc.sort_values('sum_w', ascending=False).drop_duplicates('Zip_Code', keep='first')
rent_df, inc_df = gr, inc
EXT_R = ['rMean','rMedian','rStdev','rSamples','rCV','rGap']
EXT_I = ['iMean','iMedian','iStdev','iW','iCV','iGap']
rent_df = rent_df.rename(columns={'Mean':'rMean','Median':'rMedian','Stdev':'rStdev','Samples':'rSamples'})
inc_df = inc_df.rename(columns={'Mean':'iMean','Median':'iMedian','Stdev':'iStdev','sum_w':'iW'})
rent_df['rCV'] = rent_df.rStdev/rent_df.rMean.replace(0,np.nan)
rent_df['rGap'] = (rent_df.rMean-rent_df.rMedian)/rent_df.rMean.replace(0,np.nan)
inc_df['iCV'] = inc_df.iStdev/inc_df.iMean.replace(0,np.nan)
inc_df['iGap'] = (inc_df.iMean-inc_df.iMedian)/inc_df.iMean.replace(0,np.nan)

def load_ext():
    e = cyb.set_index('zip')[CYB].copy()
    e = e.join(rent_df.set_index('Zip_Code')[EXT_R])
    e = e.join(inc_df.set_index('Zip_Code')[EXT_I])
    return e

EXT = CYB + EXT_R + EXT_I

def prep(df, ext):
    d = df.merge(ext, left_on='Zip', right_index=True, how='left')
    d['ratio'] = (d.RentIndex*12.0)/d.AnnualMedianHouseholdIncome.replace(0, np.nan)
    d['LogRent'] = np.log1p(d.RentIndex.clip(0)); d['LogIncome'] = np.log1p(d.AnnualMedianHouseholdIncome.clip(0))
    for c in ['State','Metro','City']: d[c] = d[c].fillna('__NA__').astype('category')
    return d

BASE = ['SampleSize','RentIndex','AnnualMedianHouseholdIncome','OwnerPercent','RenterPercent','MedianHomeLengthofResidence','AvgNumberofChildren','AvgNumberofAdults','AvgHouseholdSize','AvgGenerationsinHousehold','YoungAdultinHouseholdPercent','SeniorAdultinHouseholdPercent','ratio','LogRent','LogIncome','State','Metro','City']

def rmse(y,p): return np.sqrt(np.mean((np.asarray(y,float)-np.asarray(p,float))**2))

def geo_cv(cols, n_folds=5, seeds=(0,1,2), leaves=63, mcs=30, lam=10.0, lr=0.03, verbose=True):
    ext = load_ext()
    total = np.zeros(len(te)); errs = []
    for seed in seeds:
        zips = np.sort(tr.Zip.unique()); rng = np.random.RandomState(seed); rng.shuffle(zips)
        folds = np.array_split(zips, n_folds)
        pred = np.zeros(len(te))
        for f in folds:
            val = tr[tr.Zip.isin(set(f))]; trn = tr[~tr.Zip.isin(set(f))]
            ev = val[val.mon==6]
            dtrn, dev, dte = prep(trn, ext), prep(ev, ext), prep(te, ext)
            m = lgb.LGBMRegressor(objective='regression', metric='rmse', learning_rate=lr, num_leaves=leaves,
                min_child_samples=mcs, subsample=0.9, subsample_freq=1, colsample_bytree=0.8, reg_lambda=lam,
                verbose=-1, n_estimators=4000, random_state=42, n_jobs=16, categorical_feature=['State','Metro','City'])
            m.fit(dtrn[cols], trn.AffordabilityPercentageTrue.values, eval_set=[(dev[cols], ev.AffordabilityPercentageTrue.values)],
                  callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(0)])
            errs.append(rmse(ev.AffordabilityPercentageTrue.values, m.predict(dev[cols])))
            pred += m.predict(dte[cols])
        total += pred
    total /= (n_folds*len(seeds))
    if verbose:
        print('  folds:', [round(e,3) for e in errs], 'mean', round(np.mean(errs),3), flush=True)
    return total, np.mean(errs)

if __name__ == '__main__':
    ext = load_ext()
    print('external coverage train:', tr.Zip.isin(set(ext.index)).mean().round(3), 'test:', te.Zip.isin(set(ext.index)).mean().round(3))
    print('base only:'); p0, e0 = geo_cv(BASE)
    print('base + cybermaxtools:'); p1, e1 = geo_cv(BASE+CYB)
    print('base + all ext:'); p2, e2 = geo_cv(BASE+EXT)
    np.save('models/geo_base.npy', p0); np.save('models/geo_cyb.npy', p1); np.save('models/geo_ext.npy', p2)
    json.dump({'base': e0, 'cyb': e1, 'ext': e2}, open('models/geo_ext_scores.json','w'))
