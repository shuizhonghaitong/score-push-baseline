"""Per-variable: GT line SNR, per-cell line R^2 (ceiling), pooled-line agreement."""
import sys, numpy as np
sys.path.insert(0, 'scripts')
import train_v4 as t4
from climx_common import SUB_COLUMNS, lite_file
import netCDF4 as nc
from pathlib import Path

ssps = ["ssp126", "ssp370", "ssp585"]
gt = {s: t4.gt_indices_npz(Path("models/cache"), s) for s in ssps}
co2_ref = np.asarray(nc.Dataset(str(lite_file("ssp126"))).variables["CO2_LBC"][0])
F = {s: np.log(np.asarray(nc.Dataset(str(lite_file(s))).variables["CO2_LBC"][0]) / co2_ref) for s in ssps}
Fh_all = {s: F[s][:85] for s in ssps}

for ho in ssps:
    tr = [s for s in ssps if s != ho]
    Fp = np.concatenate([F[s][:85] for s in tr])
    Fc = Fp - Fp.mean(); varF = (Fc**2).sum()
    Fh = Fh_all[ho]
    fhc = Fh - Fh.mean(); varFh = (fhc**2).sum()
    print(f"== holdout {ho} ==")
    print(f"{'var':8s} {'snr_med':>8s} {'ceilR2_med':>10s} {'poolR2_med':>10s} {'sigfrac':>8s}")
    for k in SUB_COLUMNS:
        rows = []
        for c in range(216):
            i_, j_ = c // 18, c % 18
            I_ho = gt[ho][k][:85, i_, j_].ravel()
            var = I_ho.var()
            if var <= 1e-6:
                continue
            bh = (I_ho * fhc).sum() / varFh
            ah = I_ho.mean() - bh * Fh.mean()
            r2c = 1 - ((I_ho - ah - bh*Fh)**2).sum() / var
            Itot = np.concatenate([gt[s][k][:85, i_, j_].ravel() for s in tr])
            b = (Itot * Fc).sum() / varF
            a = Itot.mean() - b * Fp.mean()
            pred = a + b * Fh
            r2p = 1 - ((I_ho - pred)**2).sum() / var
            noise = np.sqrt(((Itot - (a + b*Fp))**2).mean())
            lineamp = abs(b) * np.sqrt(varF/3)
            rows.append((r2c, r2p, lineamp/noise))
        rows = np.array(rows)
        print(f"{k:8s} {np.median(rows[:,2]):8.3f} {np.median(rows[:,0]):10.4f} "
              f"{np.median(rows[:,1]):10.4f} {(rows[:,0] > 0).mean():8.3f}")
