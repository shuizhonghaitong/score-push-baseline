"""Compare target estimators on the 3-fold CV: F-window median (v1) vs age-paired
weighted median vs age+F, with/without spatial smoothing."""
import sys, numpy as np
sys.path.insert(0, 'scripts')
import train_v6 as v6
import train_v4 as t4
from climx_common import SUB_COLUMNS
from scipy.ndimage import gaussian_filter
from pathlib import Path
from rnNSE_lite import score_lite

F = v6.load_F()
gt = {s: t4.gt_indices_npz(Path("models/cache"), s) for s in t4.SSPS}

def score_targets(ho, T):
    T86 = {k: np.concatenate([T[k], T[k][-1:]], axis=0) for k in T}
    S, per = score_lite(gt[ho], T86)
    return S, per

for ho in ["ssp370", "ssp585"]:
    tr = [s for s in t4.SSPS if s != ho]
    Fh = F[ho]
    print(f"== holdout {ho}")
    # variant configs: (name, age_tau, h, spatial)
    for (name, tau, h, spat) in [
        ("Fwin0.15        ", 0.0, 0.15, 0.0),
        ("Fwin0.25        ", 0.0, 0.25, 0.0),
        ("age15+F0.15     ", 15.0, 0.15, 0.0),
        ("age15+F0.25     ", 15.0, 0.25, 0.0),
        ("age20+F0.25+sp1 ", 20.0, 0.25, 1.0),
        ("age10+F0.30+sp1 ", 10.0, 0.30, 1.0),
        ("age30+F0.35+sp1 ", 30.0, 0.35, 1.0),
    ]:
        T = {}
        for k in SUB_COLUMNS:
            Itr = {s: gt[s][k][:85].reshape(85, -1) for s in tr}  # (85,216)
            Ftr = {s: F[s] for s in tr}
            Tk = np.zeros((85, 216))
            for t in range(85):
                vals = []
                ws = []
                for s in tr:
                    for yp in range(85):
                        d_age = abs(yp - t)
                        d_F = abs(Ftr[s][yp] - Fh[t])
                        if tau > 0 and h > 0:
                            w = np.exp(-(d_age / tau) ** 2) * np.exp(-(d_F / h) ** 2)
                        elif tau > 0:
                            w = np.exp(-(d_age / tau) ** 2)
                        else:
                            w = 1.0 if d_F < 0.6 else 0.0
                        if w > 1e-3:
                            vals.append(Itr[s][yp])
                            ws.append(w)
                vals = np.stack(vals)  # (n, 216)
                ws = np.array(ws)
                ws = ws / ws.sum()
                order = np.argsort(vals, axis=0)
                vsorted = np.take_along_axis(vals, order, axis=0)
                wsorted = np.take_along_axis(ws[:, None], order, axis=0)
                cws = np.cumsum(wsorted, axis=0)
                half = cws[-1] / 2
                med_idx = np.argmax(cws >= half[None, :], axis=0)
                Tk[t] = np.take_along_axis(vsorted, med_idx[None, :], axis=0)[0]
            Tk3 = Tk.reshape(85, 12, 18)
            if spat > 0:
                Tk3 = gaussian_filter(Tk3, (2.0, spat, spat), mode="nearest")
            else:
                Tk3 = gaussian_filter(Tk3, (2.0, 0.01, 0.01), mode="nearest")
            T[k] = Tk3
        S, per = score_targets(ho, T)
        print(f"  {name} S={S:+.4f}  " + " ".join(f"{k[:4]}:{per[k]:+.3f}" for k in SUB_COLUMNS[:8]))
