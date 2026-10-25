"""Full-res linear emulation: per-cell OLS of fields/indices on global LBC forcings.

CV: hold out one SSP, train per-cell models on the other two, predict the
held-out SSP's annual indices at 192x288, score with the official full-res rnNSE.
Test: fit on all three SSPs, predict ssp245 (2015-2014->2100) using the test LBC series.
"""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from climx_common import (ar6_mask_full, SUB_COLUMNS, full_file, make_submission_parquet,
                          _li_lj)

CACHE = Path("models/cache")
N_Y = 86
FIELDS6 = ["tas", "tasmax", "tasmin", "sfcWind", "pr_mean", "pr_tot"]
IDX_NAMES = SUB_COLUMNS
IDX_PREDICTORS = {
    "TXx":  ["tasmax"],
    "TNn":  ["tasmin"],
    "SU":   ["tasmax", "tas"],
    "TR":   ["tasmin", "tas"],
    "FD":   ["tasmin", "tas"],
    "ID":   ["tasmax", "tas"],
    "WSDI": ["sfcWind", "tas"],
    "CSDI": ["tasmin", "tas"],
    "GSL":  ["tas"],
    "Rx5day": ["pr_mean", "pr_tot", "tas"],
    "CDD":  ["pr_tot", "pr_mean", "tas"],
    "CWD":  ["pr_tot", "pr_mean", "tas"],
    "R95p": ["pr_tot", "pr_mean", "tas"],
    "R10mm": ["pr_tot", "pr_mean", "tas"],
    "SDII": ["pr_mean", "pr_tot", "tas"],
}
LBCV = ["CO2_LBC", "CH4_LBC", "N2O_LBC", "CFC11eq_LBC"]
AEROV = ["BC_AX", "BC_N", "SO2", "SO4_PR", "OM_NI"]


def _aero_series(scen: str) -> np.ndarray:
    """Annual-mean land aerosols (+CF2CL2 LBC): (86, 6)."""
    z = np.load(CACHE / f"aero_annual_{scen}.npz")
    a = np.stack([z[v].astype(np.float64) for v in AEROV], axis=1)
    if "cf2cl2" in z:
        a = np.hstack([a, z["cf2cl2"].astype(np.float64)[:, None]])
    z.close()
    return a


_LOAD_CACHE = {}


def load_full(scen: str) -> dict:
    if scen in _LOAD_CACHE:
        return _LOAD_CACHE[scen]
    z = np.load(CACHE / f"full_{scen}.npz")
    out = {
        "idx": {k: z[f"idx_{k}"] for k in IDX_NAMES},
        "mean": {
            "tas": z["mean_tas"], "tasmax": z["mean_tasmax"], "tasmin": z["mean_tasmin"],
            "sfcWind": z["mean_wind"], "pr_mean": z["pr_mean"], "pr_tot": z["pr_tot"],
        },
        "lbc": z["lbc"],
    }
    z.close()
    _LOAD_CACHE[scen] = out
    return out


def fullres_nnse(gt: dict, pred: dict) -> tuple[float, dict]:
    """Official rnNSE, numpy port (AR6 land regions, cos lat weights)."""
    mask = ar6_mask_full()
    lats = np.linspace(-90.0, 90.0, 192)
    cos_w = np.cos(np.deg2rad(lats))
    regions = np.unique(mask[mask >= 0])
    eps = 1e-6
    per_var = {}
    for col in IDX_NAMES:
        g = gt[col].astype(np.float64)
        p = pred[col].astype(np.float64)
        var_cell = g.var(axis=0)
        mse_cell = ((g - p) ** 2).mean(axis=0)
        valid = var_cell > eps
        r2 = np.where(valid, 1.0 - mse_cell / np.maximum(var_cell, eps), np.nan)
        nnse = np.where(valid, r2 / (2.0 - r2), np.nan)
        rs = []
        for rc in regions:
            inr = (mask == rc) & valid
            w = cos_w[:, None] * inr
            tw = w.sum()
            if tw < 1e-12:
                continue
            rs.append(float(np.sum(nnse[inr] * w[inr]) / tw))
        per_var[col] = float(np.mean(rs)) if rs else 0.0
    return float(np.mean(list(per_var.values()))), per_var


def _pinv_solve(X: np.ndarray, Y: np.ndarray, ridge: float):
    """X (n,d) with intercept col, Y (n,C) -> theta (C, d)."""
    d = X.shape[1]
    A = np.linalg.pinv(X.T @ X + ridge * np.eye(d))
    return (A @ X.T @ Y).T


def _land_cells():
    land = ar6_mask_full() >= 0
    r, c = np.where(land)
    return r * 288 + c


def _block_map_full_to_lite():
    li, lj = _li_lj()
    i = np.repeat(li, 288)
    j = np.tile(lj, 192)
    return i * 18 + j  # (55742,) lite cell index per full cell


def _lbc_basis(M: np.ndarray) -> tuple[np.ndarray, int]:
    """Orthonormal PCA basis (4,k) of stacked centered training LBC (n,4);
    k = #components explaining >=98% variance (at least 1)."""
    _, S, Vt = np.linalg.svd(M, full_matrices=False)
    lam = S ** 2
    tot = lam.sum()
    k = 1
    for j in range(1, 4):
        k = j + 1
        if lam[:k].sum() >= 0.98 * tot:
            break
    return Vt[:k].T, k


def emulate(scens_train: list[str], lbc_test: np.ndarray,
            test_field_anchor: dict | None, ridge: float = 1e-3,
            block_pred: dict | None = None, test_block_pred: np.ndarray | None = None,
            block_center: float = 0.0, use_quad: bool = False,
            use_gate: bool = False, use_aero: bool = False) -> dict:
    """Fit per-cell linear emulators on scens_train, predict test-scen indices.

    test_field_anchor: name->(86,192,288) spatial anchor for field means.
      - CV: per-cell mean over the training scenarios (same grid, in-distribution)
      - test: also per-cell mean over training scenarios (best available full-res anchor)
    block_pred: dict scen -> {name -> (86,12,18)} lite per-index predictions (v2 feature).
      Adds per-cell block-anomaly of the same index as an extra predictor.
    test_block_pred: {name -> (86,12,18)} lite predictions for the test/held-out scen.
    Returns: dict name -> (86,192,288) predicted index.
    """
    ba = None
    if block_pred is not None:
        lit = _block_map_full_to_lite()
        ci = _land_cells()
        ba = {}
        for name in IDX_NAMES:
            parts = []
            for s in scens_train:
                A = np.asarray(block_pred[s][name], dtype=np.float64).reshape(N_Y, -1)[:, lit]
                A = A[:, ci]
                parts.append(A - A.mean(axis=0))
            ba[name] = np.stack(parts, axis=1).reshape(-1, ci.size)
        tb = {}
        for name in IDX_NAMES:
            A = np.asarray(test_block_pred[name], dtype=np.float64).reshape(N_Y, -1)[:, lit][:, ci]
            tb[name] = A - A.mean(axis=0)
    else:
        tb = None
    t0 = time.time()
    data = {s: load_full(s) for s in scens_train}
    ci = _land_cells()
    C = ci.size
    nS = len(scens_train)
    n = N_Y * nS
    lbc_train = np.stack([data[s]["lbc"] for s in scens_train], axis=1)  # (86,nS,4)
    lbc_anchor = lbc_train[:, 0, :].mean(axis=0)  # (4,) 2015 common start
    Lc = lbc_train.reshape(-1, 4) - lbc_anchor  # (n,4)
    Vb, kb = _lbc_basis(Lc)
    Z = Lc @ Vb  # (n,k)
    zscale = np.sqrt((Z ** 2).sum(axis=0) / n) + 1e-30
    Z = Z / zscale  # unit-scale basis so the ridge penalty is well-posed
    # extra per-row forcing features: global-land annual aerosols + CF2CL2
    An = None
    if use_aero:
        At = np.vstack([_aero_series(s) for s in scens_train])  # (n,6)
        am, asd = At.mean(axis=0), At.std(axis=0) + 1e-30
        An = (At - am) / asd
        aero_test_n = (_aero_series("test") - am) / asd  # (86,6)
    # quadratic z features: captures curvature of the per-cell response around the per-year anchor
    # (disabled by default: temporal CV showed quadratics hurt badly)
    if use_quad:
        zcols = [Z[:, q] for q in range(Z.shape[1])]
        Q = np.stack([zcols[i] * zcols[j]
                      for i in range(len(zcols)) for j in range(i, len(zcols))], axis=1)
        qscale = np.sqrt((Q ** 2).sum(axis=0) / n) + 1e-30
        Qs = Q / qscale
    else:
        Qs = np.empty((n, 0))
    Xd = np.hstack([np.ones((n, 1)), Z, Qs] + ([An] if An is not None else []))

    # per-cell field train means (anchors) + anomalies
    field_mean3 = {}
    field_anom = {}
    for f in FIELDS6:
        A = np.stack([data[s]["mean"][f].reshape(N_Y, -1)[:, ci].astype(np.float64) for s in scens_train], axis=1).reshape(n, C)
        mA = A.mean(axis=0, keepdims=True)
        field_mean3[f] = mA
        field_anom[f] = A - mA

    theta_field = {f: _pinv_solve(Xd, field_anom[f], ridge) for f in FIELDS6}
    print(f"  fields fit {time.time()-t0:.0f}s C={C}", flush=True)

    zt = ((lbc_test - lbc_anchor) @ Vb) / zscale  # (86,kz)
    if use_quad:
        qt = np.stack([zt[:, i] * zt[:, j]
                       for i in range(zt.shape[1]) for j in range(i, zt.shape[1])], axis=1) / qscale
        xrow = np.hstack([np.ones((N_Y, 1)), zt, qt])  # (86,1+kz+kq)
    else:
        xrow = np.hstack([np.ones((N_Y, 1)), zt])
    if An is not None:
        xrow = np.hstack([xrow, aero_test_n])
    field_pred = {f: (field_mean3[f].T + (xrow @ theta_field[f].T).T) for f in FIELDS6}  # (C,86)

    idx_pred = {}
    for name in IDX_NAMES:
        P = IDX_PREDICTORS[name]
        Y = np.stack([data[s]["idx"][name].reshape(N_Y, -1)[:, ci].astype(np.float64) for s in scens_train], axis=1).reshape(n, C)
        L = len(P)
        has_ba = ba is not None
        # global-PCA orthogonalization of (often collinear) field predictors
        if L > 1:
            Gm = None
            for f in P:
                v = field_anom[f].mean(axis=1)  # (n,)
                Gm = v[:, None] if Gm is None else np.hstack([Gm, v[:, None]])
            Gc = Gm - Gm.mean(axis=0)  # (n, L)
            ev = np.linalg.svd(Gc, compute_uv=False) ** 2
            kp = 1
            for j in range(1, L):
                kp = j + 1
                if ev[:kp].sum() >= 0.98 * ev.sum():
                    break
            _, _, Vt = np.linalg.svd(Gc, full_matrices=False)
            W = Vt[:kp].T  # (L, kp)
        else:
            kp, W = L, np.eye(L)
        # per-cell trend gate: in-sample R2 of Y on [1, z] (trend-like channels are
        # damped where forcing explains little of the cell's index variability)
        # (disabled by default: temporal CV showed gating hurts badly)
        if use_gate:
            Zn = np.hstack([np.ones((n, 1)), Z])
            Azzinv = np.linalg.pinv(Zn.T @ Zn)
            coef = (Azzinv @ (Zn.T @ Y)).T  # (C, 1+kz)
            fitz = Zn @ coef.T  # (n, C)
            ss_res = ((Y - fitz) ** 2).sum(axis=0)
            ss_tot = ((Y - Y.mean(axis=0)) ** 2).sum(axis=0)
            gate = np.clip(1 - ss_res / np.maximum(ss_tot, 1e-12), 0.0, 1.0)
            del fitz, coef
        else:
            gate = None

        parts, parts_pred = [], []
        # direct LBC-z channels (per-cell constant -> always well-conditioned)
        ztest = xrow[:, 1:1 + Z.shape[1]]  # (86, kz)
        for q in range(Z.shape[1]):
            parts.append(Z[:, q][:, None])           # (n,1)
            parts_pred.append(ztest[:, q][:, None])   # (86,1)
        if An is not None:
            for q in range(An.shape[1]):
                parts.append(An[:, q][:, None])
                parts_pred.append(aero_test_n[:, q][:, None])
        for q in range(Qs.shape[1]):
            parts.append(Qs[:, q][:, None])
            parts_pred.append(qt[:, q][:, None])
        for q in range(kp):
            part = partp = None
            for f, w in zip(P, W[:, q]):
                if w == 0.0:
                    continue
                part = field_anom[f] * w if part is None else part + field_anom[f] * w
                partp = (field_pred[f] - field_mean3[f].T) * w if partp is None else partp + (field_pred[f] - field_mean3[f].T) * w
            sc = part.std() + 1e-30
            parts.append(part / sc)
            parts_pred.append(partp / sc)
        if has_ba:
            sc = ba[name].std() + 1e-30
            parts.append(ba[name] / sc)
            parts_pred.append(tb[name] / sc)
        d = 1 + len(parts)
        # per-cell Gram (C, d, d) and rhs (C, d)
        G = np.empty((C, d, d))
        b = np.empty((C, d))
        b[:, 0] = Y.sum(axis=0)
        G[:, 0, 0] = float(n)
        for j in range(d - 1):
            b[:, 1 + j] = (parts[j] * Y).sum(axis=0)
            G[:, 0, 1 + j] = parts[j].sum(axis=0)
            G[:, 1 + j, 0] = G[:, 0, 1 + j]
        for j in range(1, d):
            for kk in range(j, d):
                g = (parts[j - 1] * parts[kk - 1]).sum(axis=0)
                G[:, j, kk] = g
                if j != kk:
                    G[:, kk, j] = g
        ident = np.eye(d)[None, :, :]
        idx_ridge = 1e-2
        try:
            theta = np.linalg.solve(G + idx_ridge * ident, b[:, :, None])[..., 0]  # (C, d)
        except np.linalg.LinAlgError:
            Gflat = (G + idx_ridge * ident).reshape(-1, d, d)
            bflat = b.reshape(C, d)
            sol, res, rank, sv = np.linalg.lstsq(Gflat, bflat, rcond=None)
            theta = sol.reshape(C, d)
        val = np.empty((N_Y, C))
        val[:] = theta[:, 0].T
        trend = np.zeros((N_Y, C))
        for j in range(len(parts_pred)):
            pp = parts_pred[j]
            if pp.shape[-1] == 1:  # z channel: (86,1), per-cell constant
                trend += pp.reshape(-1, 1) * theta[:, 1 + j]
            else:
                trend += pp.T * theta[:, 1 + j][None, :]
        val += trend if gate is None else trend * gate
        full = np.full((N_Y, 192, 288), float(np.nanmean(val)), dtype=np.float64)
        full.reshape(N_Y, -1)[:, ci] = val
        idx_pred[name] = full
    print(f"  total {time.time()-t0:.0f}s", flush=True)
    return idx_pred


def run_cv(ridge=1e-3, use_quad=False, use_gate=False, use_aero=False):
    ssps = ["ssp126", "ssp370", "ssp585"]
    allper = {}
    for ho in ssps:
        tr = [s for s in ssps if s != ho]
        data_ho = load_full(ho)
        pred = emulate(tr, data_ho["lbc"], None, ridge=ridge,
                     use_quad=use_quad, use_gate=use_gate, use_aero=use_aero)
        S, per = fullres_nnse(data_ho["idx"], pred)
        allper[ho] = per
        print(f"\n== CV fold holdout={ho} (train {tr}): S={S:+.5f} ==", flush=True)
        for k in IDX_NAMES:
            print(f"  {k:8s} {per[k]:+.4f}", flush=True)
    avg = {k: float(np.mean([allper[ho][k] for ho in ssps])) for k in IDX_NAMES}
    # 245 is intermediate: weight 126/370 folds higher (0.4/0.4/0.2)
    w126, w370, w585 = 0.4, 0.4, 0.2
    est = {k: w126 * allper["ssp126"][k] + w370 * allper["ssp370"][k] + w585 * allper["ssp585"][k] for k in IDX_NAMES}
    print("\n== 3-fold mean / weighted (0.4,0.4,0.2) ==", flush=True)
    for k in IDX_NAMES:
        print(f"  {k:8s} {avg[k]:+.4f}  {est[k]:+.4f}", flush=True)
    print(f"  MEAN    {float(np.mean(list(avg.values()))):+.4f}  {float(np.mean(list(est.values()))):+.4f}", flush=True)
    return allper


def test_lbc() -> np.ndarray:
    import netCDF4 as nc
    ds = nc.Dataset(str(full_file("test_forcings")))
    lbc = np.stack([np.asarray(ds.variables[v][0, :N_Y], dtype=np.float32) for v in LBCV], axis=1)
    ds.close()
    return lbc


def load_lite_cv(scen: str) -> dict:
    z = np.load(CACHE / f"lite_cv_idx_v9_{scen}.npz")
    out = {k: z[k].astype(np.float32) for k in IDX_NAMES}
    z.close()
    return out


def predict_test(ridge=1e-3, out="submissions/submission_fullres_v1.parquet", ssps=None,
                 use_block=False, use_quad=False, use_gate=False, use_aero=False):
    ssps = ssps or ["ssp126", "ssp370", "ssp585"]
    block_pred = test_block_pred = None
    if use_block:
        block_pred = {s: load_lite_cv(s) for s in ssps}
        z = np.load(CACHE / "lite_test_idx_v9.npz")
        test_block_pred = {k: z[k].astype(np.float32) for k in IDX_NAMES}
        z.close()
    pred = emulate(ssps, test_lbc(), None, ridge=ridge,
                   block_pred=block_pred, test_block_pred=test_block_pred,
                   use_quad=use_quad, use_gate=use_gate, use_aero=use_aero)
    p = make_submission_parquet({k: v.astype(np.float32) for k, v in pred.items()}, Path(out))
    print("wrote", p, flush=True)
    return p


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["cv", "test"], default="cv")
    ap.add_argument("--ridge", type=float, default=1e-3)
    ap.add_argument("--out", default="submissions/submission_fullres_v1.parquet")
    ap.add_argument("--scens", default="", help="comma list of training SSPs, e.g. ssp126,ssp370")
    ap.add_argument("--block", action="store_true", help="add V9 lite block-anomaly feature")
    ap.add_argument("--quad", action="store_true", help="enable quadratic LBC-z terms (off=winning)")
    ap.add_argument("--gate", action="store_true", help="enable per-cell trend gate (off=winning)")
    ap.add_argument("--aero", action="store_true", help="add global-land aerosol + CF2CL2 per-row features")
    a = ap.parse_args()
    if a.mode == "cv":
        run_cv(ridge=a.ridge, use_quad=a.quad, use_gate=a.gate, use_aero=a.aero)
    else:
        predict_test(ridge=a.ridge, out=a.out, use_block=a.block, use_quad=a.quad, use_gate=a.gate, use_aero=a.aero,
                     ssps=[s.strip() for s in a.scens.split(",") if s.strip()] or None)
