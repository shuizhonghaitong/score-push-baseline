import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "scripts")
import numpy as np
import train_v4 as t4
from climx_common import lite_file, SUB_COLUMNS
import train_v5 as t5

Cn = 216
model = t4.V4(hist_start=1961, sigma=2.0)
model.fit(["ssp126", "ssp370", "ssp585"])

lines = {}
for s in t4.SSPS:
    gt = t4.gt_indices_npz(__import__("pathlib").Path("models/cache"), s)
    F = model.F_test[s][:85]
    lines[s] = {}
    for k in ["TXx", "TNn", "WSDI", "CSDI", "SU", "TR", "FD", "GSL", "CDD", "R95p", "R10mm", "SDII", "CWD", "Rx5day", "ID"]:
        I = gt[k][:85].reshape(85, Cn)
        a, b = t5.fit_line(I, F)
        lines[s][k] = (a, b)

print(f"{'index':8s} | per-scenario line deviation from pooled-2-SSP line (median over cells), units: index | sigma_eta")
for k in ["TXx", "TNn", "WSDI", "CSDI", "SU", "TR", "FD", "GSL", "CDD", "R95p", "R10mm", "SDII", "CWD", "Rx5day", "ID"]:
    row = []
    for hold in t4.SSPS:
        pool = [s for s in t4.SSPS if s != hold]
        Fp = np.concatenate([model.F_test[s][:85] for s in pool])
        Ip = np.concatenate([t4.gt_indices_npz(__import__("pathlib").Path("models/cache"), s)[k][:85].reshape(85, Cn) for s in pool], axis=0)
        a_p, b_p = t5.fit_line(Ip, Fp)
        a_h, b_h = lines[hold][k]
        # deviation of holdout line from pooled line over the test F range (0..0.5)
        Ftest = np.linspace(0, 0.5, 50)[:, None]
        dev = (a_h + b_h * Ftest) - (a_p + b_p * Ftest)  # (50, Cn)
        # noise sigma for holdout
        F_h = model.F_test[hold][:85]
        I_h = lines[hold][k]
        gt_h = t4.gt_indices_npz(__import__("pathlib").Path("models/cache"), hold)[k][:85].reshape(85, Cn)
        resid = gt_h - (a_h[None] + b_h[None] * F_h[:, None])
        sig = resid.std(axis=0)
        row.append(f"dev_med={np.median(np.abs(dev).mean(axis=0)):7.3f} sig={np.median(sig):6.3f} ratio={np.median(np.abs(dev).mean(axis=0)/np.maximum(sig,1e-9)):5.2f}")
    print(f"{k:8s} | " + " | ".join(row))
