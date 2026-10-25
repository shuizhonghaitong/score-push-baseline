import warnings; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, lightgbm as lgb
from model import load, prep_base, build_X, rmse, TE_COLS
from features import NUM_FEATS, TARGET

tr, te = load()
zips = np.sort(tr['Zip'].unique())
rng = np.random.RandomState(0); rng.shuffle(zips)
folds = np.array_split(zips, 5)

def run(use_te, cats, leaves, mcs, lam, lr=0.03, nfold=5, n_est=4000):
    errs = []
    for f in folds[:nfold]:
        val = tr[tr['Zip'].isin(set(f))]; trn = tr[~tr['Zip'].isin(set(f))]
        ev = val[val.mon==6]
        dval = prep_base(ev); dtrn = prep_base(trn)
        Xtr = dtrn[NUM_FEATS + ['RentIncomeRatio','LogRent','LogIncome','LogN','InvSqrtN','RentZst','IncZst','OwnerZst']].copy()
        Xv = dval[NUM_FEATS + ['RentIncomeRatio','LogRent','LogIncome','LogN','InvSqrtN','RentZst','IncZst','OwnerZst']].copy()
        gm = trn[TARGET].mean()
        if use_te:
            from model import target_encode
            for c in cats:
                Xtr[f'TE_{c}'] = target_encode(trn, dtrn, c, TE_COLS[c], gm)
                Xv[f'TE_{c}'] = target_encode(trn, dval, c, TE_COLS[c], gm)
        Xtr[cats] = dtrn[cats]; Xv[cats] = dval[cats]
        cols = list(Xtr.columns)
        m = lgb.LGBMRegressor(objective='regression', metric='rmse', learning_rate=lr, num_leaves=leaves,
            min_child_samples=mcs, subsample=0.9, subsample_freq=1, colsample_bytree=0.8, reg_lambda=lam,
            verbose=-1, n_estimators=n_est, random_state=42, n_jobs=16, categorical_feature=cats)
        m.fit(Xtr, trn[TARGET].values, eval_set=[(Xv, ev[TARGET].values)],
              callbacks=[lgb.early_stopping(200, verbose=False), lgb.log_evaluation(0)])
        errs.append(rmse(ev[TARGET].values, m.predict(Xv)))
    return round(float(np.mean(errs)),3), [round(e,3) for e in errs]

for cfg in [
    dict(use_te=False, cats=['State','Metro'], leaves=63, mcs=30, lam=10.0),
    dict(use_te=False, cats=['State','Metro','City'], leaves=63, mcs=30, lam=10.0),
    dict(use_te=False, cats=['State','Metro'], leaves=31, mcs=50, lam=30.0),
    dict(use_te=False, cats=['State','Metro'], leaves=127, mcs=20, lam=5.0),
    dict(use_te=True, cats=['State'], leaves=63, mcs=30, lam=10.0),
]:
    name = f"te={cfg['use_te']} cats={cfg['cats']} L={cfg['leaves']} mcs={cfg['mcs']} lam={cfg['lam']}"
    print(name, run(**cfg), flush=True)
