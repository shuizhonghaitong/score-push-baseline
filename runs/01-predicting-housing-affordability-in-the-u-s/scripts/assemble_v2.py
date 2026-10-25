"""Assemble v2 submission from fold checkpoints."""
import json, warnings; warnings.filterwarnings('ignore')
from pathlib import Path
import numpy as np, pandas as pd
from scripts.pipeline_v2 import rmse

ROOT = Path(__file__).resolve().parent.parent
CKPT = ROOT/'models/v2_folds'
seeds = (0,1,2,3,4)
preds, errs = [], []
for s in seeds:
    for f in range(5):
        p = CKPT/f's{s}_f{f}_test.npy'
        e = CKPT/f's{s}_f{f}_err.txt'
        if not (p.exists() and e.exists()):
            print('MISSING', s, f); continue
        preds.append(np.load(p)); errs.append(float(e.read_text().split()[0]))
assert len(preds) == 25, len(preds)
new_pred = np.mean(preds, axis=0)
tr = pd.read_csv(ROOT/'data/train.csv', dtype={'Zip': str})
te = pd.read_csv(ROOT/'data/test.csv', dtype={'Zip': str})
tr['mon'] = pd.to_datetime(tr['Month']).dt.month
may = tr[tr.mon==5].set_index('Zip')['AffordabilityPercentageTrue']
jun = tr[tr.mon==6].set_index('Zip')['AffordabilityPercentageTrue']
idx = may.index.intersection(jun.index)
a, b = may.loc[idx].values, jun.loc[idx].values
slope = np.cov(a,b)[0,1]/np.var(a); intercept = b.mean()-slope*a.mean()
out = te[['Zip','State']].copy()
is_seen = out.Zip.isin(jun.index)
pred = np.empty(len(out))
pred[is_seen.values] = slope*jun.loc[out.Zip[is_seen.values]].values + intercept
pred[~is_seen.values] = new_pred[~is_seen.values]
out['AffordabilityPercentageTrue'] = np.clip(pred, 0, 100)
out['Zip'] = out.Zip.str.zfill(5)
out.to_csv(ROOT/'subs/v2_ext.csv', index=False)
seen_rmse = rmse(jun.loc[idx].values, slope*may.loc[idx].values+intercept)
new_rmse = float(np.mean(errs))
print(f'v2: seen {seen_rmse:.3f} new {new_rmse:.3f} composite {0.2*seen_rmse+0.8*new_rmse:.3f}')
json.dump({'seen_rmse': seen_rmse, 'new_rmse': new_rmse, 'composite': 0.2*seen_rmse+0.8*new_rmse},
          open(ROOT/'models/v2_cv.json','w'), indent=1)
