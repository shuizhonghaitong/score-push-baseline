"""Decompose V5 CV loss: per var, per cell, level bias / slope mismatch / noise."""
import sys, numpy as np
sys.path.insert(0, 'scripts')
import train_v5 as t5
import train_v4 as t4
from climx_indices import compute_indices, base_period_slice, percentile_doy_window
from rnNSE_lite import score_lite
from climx_common import lite_file, SUB_COLUMNS
import netCDF4 as nc

holdout = "ssp370"
train_ssps = [s for s in t4.SSPS if s != holdout]
model = t5.build_model(train_ssps, sigma=2.0)
calib = t5.calibrate(model, train_ssps, sigma_c=0.0, quick=False)
v5 = t5.V5(model, calib)

gt = t4.gt_indices_npz(__import__("pathlib").Path("models/cache"), holdout)
daily = v5.predict_daily(holdout)
idx = compute_indices(daily, model.base)
S, per = score_lite(gt, idx)
print(f"V5 CV holdout={holdout}: S={S:+.5f}")

F = model.F_test[holdout][:85]
Fc = F - F.mean()
varF = (Fc**2).sum()

for k in SUB_COLUMNS:
    G = gt[k][:85].reshape(85, -1)
    P = idx[k][:85].reshape(85, -1)
    n = G.shape[1]
    # per-cell R2
    gt_var = ((G - G.mean(0, keepdims=True))**2).sum(0)
    resid = G - P
    ss_res = (resid**2).sum(0)
    valid = gt_var > 1e-6
    r2 = 1 - ss_res/gt_var
    r2v = r2[valid]
    # level bias = mean residual; slope mismatch = cov(resid,F)/varF
    bias = resid.mean(0)
    slope_err = (resid * Fc[:, None]).sum(0) / varF
    nnoise = np.sqrt((G - G.mean(0, keepdims=True))**2).mean(0)  # per-cell std of GT
    print(f"{k:8s} R2mean={r2v.mean():+.4f} median={np.median(r2v):+.4f} "
          f"|bias|med={np.median(np.abs(bias[valid])):8.3f} slope_err_med={np.median(slope_err[valid]):+.4f} "
          f"noise_med={np.median(nnoise[valid]):8.3f} nvalid={valid.sum()}")
