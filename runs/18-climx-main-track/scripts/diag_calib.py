import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "scripts")
import numpy as np
import train_v5 as t5
import train_v4 as t4

model = t5.build_model(["ssp126", "ssp585"], sigma=2.0)
calib = t5.calibrate(model, ["ssp126", "ssp585"], sigma_c=1.5)
v5 = t5.V5(model, calib)
S, per = t5.eval_v5(v5, "ssp370")
print(f"holdout ssp370: S={S:+.5f}")

# per-cell detail for TNn and WSDI
Cn = 216
n = 170
I_pool = {}
idx = 0
for s in ["ssp126", "ssp585"]:
    gt = t4.gt_indices_npz(__import__("pathlib").Path("models/cache"), s)
    for k in ["TNn", "TR", "FD", "CSDI", "TXx", "SU", "WSDI"]:
        I_pool.setdefault(k, np.empty((n, 12, 18)))
        I_pool[k][idx:idx+85] = gt[k][:85]
    idx += 85
F_pool = np.concatenate([model.F_test[s][:85] for s in ["ssp126", "ssp585"]])
mu = {k: I_pool[k].mean(axis=0) for k in I_pool}
beta = {}
for k in I_pool:
    _, bk = t5.fit_line(I_pool[k].reshape(n, Cn), F_pool)
    beta[k] = bk.reshape(12, 18)

# predicted index lines (at F=0.15, using the calibrated field, single year)
ths = {
    "tasmax": t5.percentile_doy_window(model.base["tasmax"], 0.90),
    "tasmin": t5.percentile_doy_window(model.base["tasmin"], 0.10),
    "pr": model.th_r95p,
}
Fp = 0.15
for v, keys in [("tasmax", ["TXx", "SU", "WSDI"]), ("tasmin", ["TNn", "TR", "FD", "CSDI"])]:
    u = calib["u"][v]; d = calib["d"][v]
    field = model.S[v] + u * model.m[v] * (Fp - model.Fbar) + d
    r = t5.var_indices_year(v, field, ths)
    print(f"\n{v} at F=0.15 (pred vs target):")
    for k in keys:
        p = r[k].astype(np.float64)
        t = mu[k] + beta[k] * Fp
        err = p - t
        # per-cell: noise std of GT
        resid = I_pool[k] - (mu[k][None] + beta[k][None] * F_pool[:, None, None])
        sig_eta = resid.std(axis=0)
        # R2 proxy if level matched at Fp and slope exact:
        gt_std = I_pool[k].std(axis=0)
        r2_if_perfect_level = 1 - (sig_eta**2) / gt_std**2
        print(f"  {k:6s} err_med={np.median(err):+8.3f}  err/rng_med={np.median(np.abs(err)/np.maximum(sig_eta,1e-6)):+.2f}  R2_if_perfect_level_med={np.median(r2_if_perfect_level):+.3f}")
