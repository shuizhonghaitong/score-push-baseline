"""One (seed, fold) XGB test prediction for v3 ensemble."""
import sys, warnings; warnings.filterwarnings('ignore')
from pathlib import Path
import numpy as np, pandas as pd, xgboost as xgb
from scripts.pipeline_v2 import load_ext, prep, COLS, rmse

ROOT = Path(__file__).resolve().parent.parent
CKPT = ROOT/'models/v3_xgb'
CKPT.mkdir(parents=True, exist_ok=True)
tr0 = pd.read_csv(ROOT/'data/train.csv', dtype={'Zip': str})
te0 = pd.read_csv(ROOT/'data/test.csv', dtype={'Zip': str})

def main(seed, fold_i):
    tag = f's{seed}_f{fold_i}'
    outp = CKPT/f'{tag}_test.npy'; errp = CKPT/f'{tag}_err.txt'
    if outp.exists() and errp.exists():
        print(tag, 'exists, skip', flush=True); return
    tr = tr0.copy(); te = te0.copy()
    tr['mon'] = pd.to_datetime(tr['Month']).dt.month
    ext = load_ext()
    allcats = {c: pd.CategoricalDtype(np.sort(pd.unique(pd.concat([tr[c], te[c]]).fillna('__NA__'))).tolist()) for c in ['State','Metro','City']}
    def prep2(df, e):
        d = prep(df, e)
        for c, dt in allcats.items(): d[c] = d[c].astype(dt)
        return d
    zs = np.sort(tr.Zip.unique()); r = np.random.RandomState(seed); r.shuffle(zs)
    f = np.array_split(zs, 5)[fold_i]
    val = tr[tr.Zip.isin(set(f))]; trn = tr[~tr.Zip.isin(set(f))]
    ev = val[val.mon==6]
    dtrn, dev, dte = prep2(trn, ext), prep2(ev, ext), prep2(te, ext)
    m = xgb.XGBRegressor(n_estimators=4000, learning_rate=0.03, max_depth=7, min_child_weight=30,
                         subsample=0.9, colsample_bytree=0.8, reg_lambda=10.0, n_jobs=16, tree_method='hist',
                         random_state=42, eval_metric='rmse', enable_categorical=True, early_stopping_rounds=200)
    m.fit(dtrn[COLS], trn.AffordabilityPercentageTrue.values, eval_set=[(dev[COLS], ev.AffordabilityPercentageTrue.values)], verbose=False)
    err = rmse(ev.AffordabilityPercentageTrue.values, m.predict(dev[COLS], iteration_range=(0, m.best_iteration+1)))
    np.save(outp, m.predict(dte[COLS], iteration_range=(0, m.best_iteration+1)))
    errp.write_text(f'{err:.4f} {m.best_iteration}\n')
    print(tag, 'rmse', round(err,3), 'iters', m.best_iteration, flush=True)

if __name__ == '__main__':
    main(int(sys.argv[1]), int(sys.argv[2]))
