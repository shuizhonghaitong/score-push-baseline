import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, lightgbm as lgb
from scripts.pipeline_v2 import load_ext, prep, COLS, rmse
tr = pd.read_csv('data/train.csv', dtype={'Zip':str})
tr['mon'] = pd.to_datetime(tr['Month']).dt.month
ext = load_ext()
zs = np.sort(tr.Zip.unique()); r = np.random.RandomState(1); r.shuffle(zs)
f = np.array_split(zs, 5)[0]
val = tr[tr.Zip.isin(set(f))]; trn = tr[~tr.Zip.isin(set(f))]
ev = val[val.mon==6]
dtrn, dev = prep(trn, ext), prep(ev, ext)
m = lgb.LGBMRegressor(objective='regression', metric='rmse', learning_rate=0.03, num_leaves=63,
    min_child_samples=30, subsample=0.9, subsample_freq=1, colsample_bytree=0.8, reg_lambda=10.0,
    verbose=-1, n_estimators=4000, random_state=42, n_jobs=16, categorical_feature=['State','Metro','City'])
m.fit(dtrn[COLS], trn.AffordabilityPercentageTrue.values, eval_set=[(dev[COLS], ev.AffordabilityPercentageTrue.values)],
      callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(0)])
p = m.predict(dev[COLS])
print('quick fold rmse:', round(rmse(ev.AffordabilityPercentageTrue.values, p),3), flush=True)
