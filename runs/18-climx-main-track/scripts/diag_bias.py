import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, "scripts")
import numpy as np, netCDF4 as nc
import train_v4 as t4
from climx_indices import compute_indices, base_period_slice
from rnNSE_lite import score_lite
from climx_common import lite_file, SUB_COLUMNS

for scen in ["ssp370", "ssp126"]:
    model = t4.V4(hist_start=1961, sigma=2.0)
    model.fit([s for s in t4.SSPS if s != scen])
    ds2 = nc.Dataset(str(lite_file("hist_late")))
    base = base_period_slice({v: np.asarray(ds2.variables[v][0], dtype=np.float64) for v in t4.VARS})
    ds2.close()
    model.base = base
    gt = t4.gt_indices_npz(__import__("pathlib").Path("models/cache"), scen)
    d = model.predict_daily(scen, dict(tasmax=1.0, tasmin=1.0, tas=1.0, pr=0.0))
    idx = compute_indices(d, base)
    print(f"\n===== {scen} (pr s=0, temps s=1) =====")
    for k in SUB_COLUMNS:
        g = gt[k].astype(np.float64); p = idx[k].astype(np.float64)
        diff = p - g  # (86, 12, 18)
        bias = diff.mean(axis=0)  # (12,18)
        noise_std = diff.std(axis=0)
        gvar = g.var(axis=0)
        # R2 if we could remove bias (predict E[I|t]):
        r2_now = 1 - (diff**2).mean(axis=0) / gvar
        r2_debiased = 1 - (noise_std**2) / gvar
        med = lambda a: np.median(a)
        print(f"  {k:8s} bias_med={med(bias):+9.2f}  noise_std_med={med(noise_std):8.2f}  "
              f"gt_std_med={med(np.sqrt(gvar)):8.2f}  R2_now_med={med(r2_now):+.3f}  R2_debiased_med={med(r2_debiased):+.3f}")
    # pr slope stats
    print("  m_pr range:", model.m["pr"].min(), model.m["pr"].max(), "median", np.median(model.m["pr"]))
    print("  m_tas range:", model.m["tas"].min(), model.m["tas"].max())
    print("  m_tasmax range:", model.m["tasmax"].min(), model.m["tasmax"].max())
