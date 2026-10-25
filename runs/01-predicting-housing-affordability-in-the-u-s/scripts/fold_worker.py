"""Compute one (seed, fold) LGBM test prediction + geo-CV error; checkpoint to disk."""
import sys, warnings; warnings.filterwarnings('ignore')
from pathlib import Path
import numpy as np, pandas as pd, lightgbm as lgb
from scripts.pipeline_v2 import load_ext, prep, COLS, rmse

ROOT = Path(__file__).resolve().parent.parent
CKPT = ROOT/'models/v2_folds'
CKPT.mkdir(parents=True, exist_ok=True)

def main(seed, fold_i):
    tag = f's{seed}_f{fold_i}'
    outp = CKPT/f'{tag}_test.npy'; errp = CKPT/f'{tag}_err.txt'
    if outp.exists() and errp.exists():
        print(tag, 'exists, skip', flush=True); return
    tr = pd.read_csv(ROOT/'data/train.csv', dtype={'Zip': str})
    te = pd.read_csv(ROOT/'data/test.csv', dtype={'Zip': str})
    tr['mon'] = pd.to_datetime(tr['Month']).dt.month
    ext = load_ext()
    zs = np.sort(tr.Zip.unique()); r = np.random.RandomState(seed); r.shuffle(zs)
    f = np.array_split(zs, 5)[fold_i]
    val = tr[tr.Zip.isin(set(f))]; trn = tr[~tr.Zip.isin(set(f))]
    ev = val[val.mon==6]
    dtrn, dev, dte = prep(trn, ext), prep(ev, ext), prep(te, ext)
    m = lgb.LGBMRegressor(objective='regression', metric='rmse', learning_rate=0.03, num_leaves=63,
        min_child_samples=30, subsample=0.9, subsample_freq=1, colsample_bytree=0.8, reg_lambda=10.0,
        verbose=-1, n_estimators=4000, random_state=42, n_jobs=16, categorical_feature=['State','Metro','City'])
    m.fit(dtrn[COLS], trn.AffordabilityPercentageTrue.values, eval_set=[(dev[COLS], ev.AffordabilityPercentageTrue.values)],
          callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(0)])
    err = rmse(ev.AffordabilityPercentageTrue.values, m.predict(dev[COLS]))
    np.save(outp, m.predict(dte[COLS]))
    errp.write_text(f'{err:.4f} {m.best_iteration_}\n')
    print(tag, 'rmse', round(err,3), 'iters', m.best_iteration_, flush=True)

if __name__ == '__main__':
    seed, fold_i = int(sys.argv[1]), int(sys.argv[2])
    main(seed, fold_i)
