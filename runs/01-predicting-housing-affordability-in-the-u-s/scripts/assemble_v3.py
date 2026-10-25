"""v3 = blend of LGBM (v2 folds) + XGB (v3 folds)."""
import json, warnings; warnings.filterwarnings('ignore')
from pathlib import Path
import numpy as np, pandas as pd
from scripts.pipeline_v2 import rmse

ROOT = Path(__file__).resolve().parent.parent
def load_ckpt(folder, seeds):
    preds, errs = [], []
    for s in seeds:
        for f in range(5):
            p = folder/f's{s}_f{f}_test.npy'; e = folder/f's{s}_f{f}_err.txt'
            assert p.exists() and e.exists(), (s, f)
            preds.append(np.load(p)); errs.append(float(e.read_text().split()[0]))
    return np.mean(preds, axis=0), float(np.mean(errs))

lgbp, lgb_err = load_ckpt(ROOT/'models/v2_folds', (0,1,2,3,4))
xgbp, xgb_err = load_ckpt(ROOT/'models/v3_xgb', (0,1,2))
print('LGBM new', round(lgb_err,3), 'XGB new', round(xgb_err,3))
te = pd.read_csv(ROOT/'data/test.csv', dtype={'Zip': str})
tr = pd.read_csv(ROOT/'data/train.csv', dtype={'Zip': str})
tr['mon'] = pd.to_datetime(tr['Month']).dt.month
may = tr[tr.mon==5].set_index('Zip')['AffordabilityPercentageTrue']
jun = tr[tr.mon==6].set_index('Zip')['AffordabilityPercentageTrue']
idx = may.index.intersection(jun.index)
a, b = may.loc[idx].values, jun.loc[idx].values
slope = np.cov(a,b)[0,1]/np.var(a); intercept = b.mean()-slope*a.mean()
seen_rmse = rmse(b, slope*a+intercept)

# blend weights: inverse CV error, try few options
for w in [0.5, 0.55, 0.6, 0.65, 0.7]:
    newp = w*lgbp + (1-w)*xgbp
    out = te[['Zip','State']].copy()
    is_seen = out.Zip.isin(jun.index)
    pred = np.empty(len(out))
    pred[is_seen.values] = slope*jun.loc[out.Zip[is_seen.values]].values + intercept
    pred[~is_seen.values] = newp[~is_seen.values]
    out['AffordabilityPercentageTrue'] = np.clip(pred, 0, 100)
    out['Zip'] = out.Zip.str.zfill(5)
    new_err = w*lgb_err + (1-w)*xgb_err
    out.to_csv(ROOT/f'subs/v3_w{int(w*100)}.csv', index=False)
    print(f'w={w}: new {new_err:.3f} comp {0.2*seen_rmse+0.8*new_err:.3f}')
json.dump({'lgb_err': lgb_err, 'xgb_err': xgb_err, 'seen_rmse': seen_rmse}, open(ROOT/'models/v3_cv.json','w'), indent=1)
