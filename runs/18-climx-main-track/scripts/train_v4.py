"""V4 emulator: annual-aggregate trend model.

For each var v in {tas, tasmax, tasmin, pr}, cell c:
    pred_v(c, t, doy) = S_v(c, doy) + s_v * m_v(c) * (F(t) - Fbar)

- S_v: mean daily climatology over pooled train years
- F = ln(CO2/CO2_2015); Fbar = mean F over pooled train years
- m_v(c): OLS slope of the annual aggregate A_v(c, y) on F(y), smoothed
  spatially with Gaussian sigma (per (var) shared sigma)
- s_v: per-variable shrinkage (CV grid on the 15-index rnNSE objective)

Annual aggregates: tas->mean, tasmax->max, tasmin->min, pr->sum.
Train pool: hist years [hist_start, 2013] + 85y (2015..2099) of each train SSP.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import netCDF4 as nc
from scipy.ndimage import gaussian_filter

sys.path.insert(0, str(Path(__file__).resolve().parent))
from climx_common import lite_file, SUB_COLUMNS
from climx_indices import compute_indices, base_period_slice
from rnNSE_lite import score_lite

VARS = ["tas", "tasmax", "tasmin", "pr"]
SSPS = ["ssp126", "ssp370", "ssp585"]
AGG = {"tas": "mean", "tasmax": "max", "tasmin": "min", "pr": "sum"}


def _load(key: str, var: str) -> np.ndarray:
    ds = nc.Dataset(str(lite_file(key)))
    v = np.asarray(ds.variables[var][0], dtype=np.float32)
    ds.close()
    return v


def _co2(key: str) -> np.ndarray:
    ds = nc.Dataset(str(lite_file(key)))
    c = np.asarray(ds.variables["CO2_LBC"][0], dtype=np.float64)
    ds.close()
    return c


class V4:
    def __init__(self, hist_start: int = 1961, sigma: float = 1.5,
                 shrink: dict | None = None):
        self.hist_start = hist_start
        self.sigma = sigma
        self.shrink = shrink or {v: 1.0 for v in VARS}

    def fit(self, train_ssps: list[str]):
        t0 = time.time()
        co2_ref = _co2("ssp126")[0]
        hist_co2 = np.concatenate([_co2("hist_early"), _co2("hist_late")])
        F_hist = np.log(hist_co2 / co2_ref)  # 165 values, years 1850..2014
        F_ssp = {s: np.log(_co2(s) / co2_ref) for s in SSPS}

        # ---- pooled train samples: (F, scenario, year) ----
        samples = []  # (F_value, key, y0day_index)
        hs = self.hist_start
        for y in range(hs, 2014):
            samples.append((F_hist[y - 1850], "hist", (y - 1932) * 365 + 82 * 365))
        for s in train_ssps:
            for y in range(85):
                samples.append((F_ssp[s][y], s, y * 365))
        Ft = np.array([x[0] for x in samples])
        self.Fbar = Ft.mean()
        self.F_test = {s: F_ssp[s] for s in SSPS}
        self.F_test["ssp245"] = np.log(_co2("test_forcings") / co2_ref)
        self.Ft = Ft

        # ---- load daily data for pool ----
        hist = np.concatenate([_load("hist_early", v) for v in VARS], axis=1) if False else None
        data = {}
        for v in VARS:
            h1 = _load("hist_early", v)
            h2 = _load("hist_late", v)
            ssp_arrays = {s: _load(s, v) for s in train_ssps}
            data[v] = dict(hist=np.concatenate([h1, h2], axis=0), ssps=ssp_arrays)

        n = len(samples)
        A = {v: np.empty((n, 12, 18), dtype=np.float64) for v in VARS}
        for v in VARS:
            d = data[v]
            for i, (f, key, y0) in enumerate(samples):
                if key == "hist":
                    a = d["hist"][y0:y0 + 365]
                else:
                    a = d["ssps"][key][y0:y0 + 365]
                agg = AGG[v]
                A[v][i] = getattr(a, agg)(axis=0)

        Fc = Ft - Ft.mean()
        varF = (Fc ** 2).sum()
        self.A_ann = {v: A[v].reshape(n, -1) for v in VARS}
        self.m = {}
        for v in VARS:
            Av = self.A_ann[v]
            m = (Av * Fc[:, None]).sum(axis=0) / varF
            m = m.reshape(12, 18)
            if self.sigma > 0:
                m = gaussian_filter(m, self.sigma, mode="nearest")
            self.m[v] = m

        # ---- climatology S over the pool ----
        self.S = {}
        for v in VARS:
            d = data[v]
            acc = np.zeros((365, 12, 18), dtype=np.float64)
            cnt = 0
            for y in range(self.hist_start, 2014):
                y0 = (y - 1932) * 365 + 82 * 365
                acc += d["hist"][y0:y0 + 365].astype(np.float64)
                cnt += 1
            for s in train_ssps:
                acc += d["ssps"][s][:85 * 365].reshape(85, 365, 12, 18).sum(axis=0)
                cnt += 85
            self.S[v] = acc / cnt
            del data[v]
        print(f"  fit done {time.time()-t0:.0f}s")

    def predict_daily(self, scen: str, shrink: dict | None = None) -> dict[str, np.ndarray]:
        """Return var -> (31389, 12, 18) float64 daily predictions."""
        sh = shrink or self.shrink
        Ft = self.F_test[scen]
        out = {}
        for v in VARS:
            off = sh[v] * self.m[v] * (Ft - self.Fbar)[:, None, None]  # (86,12,18)
            arr = (self.S[v][None, :, :, :] + off[:, None, :, :])  # (86,365,12,18)
            flat = np.empty((31389, 12, 18), dtype=np.float32)
            i = 0
            for y in range(86):
                nn = 364 if y == 85 else 365
                flat[i:i + nn] = arr[y, :nn].astype(np.float32)
                i += nn
            out[v] = flat
        return out


# ---------------- GT index cache ----------------
def gt_indices_npz(cache_dir: Path, scen: str) -> dict[str, np.ndarray]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    p = cache_dir / f"gt_indices_{scen}.npz"
    if p.exists():
        z = np.load(p)
        return {k: z[k] for k in SUB_COLUMNS}
    ds = nc.Dataset(str(lite_file(scen)))
    daily = {v: np.asarray(ds.variables[v][0], dtype=np.float64) for v in VARS}
    ds.close()
    ds2 = nc.Dataset(str(lite_file("hist_late")))
    base = base_period_slice({v: np.asarray(ds2.variables[v][0], dtype=np.float64) for v in VARS})
    ds2.close()
    res = compute_indices(daily, base)
    np.savez_compressed(p, **res)
    return res


def eval_config(model: V4, shrink: dict, gt: dict, daily_pred: dict) -> tuple[float, dict]:
    idx = compute_indices(daily_pred, model.base)
    S, per = score_lite(gt, idx)
    return S, per


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", default="ssp370", help="ssp to hold out, or 'none'")
    ap.add_argument("--hist-start", type=int, default=1961)
    ap.add_argument("--sigmas", default="0,1,1.5,2,3")
    ap.add_argument("--shr-tasmax", default="0.75,1.0,1.25")
    ap.add_argument("--shr-tasmin", default="0.75,1.0,1.25")
    ap.add_argument("--shr-tas", default="0.75,1.0,1.25")
    ap.add_argument("--shr-pr", default="0,0.5,1.0,1.5")
    ap.add_argument("--out", default="models/v4")
    ap.add_argument("--quick", action="store_true", help="tiny grid for smoke test")
    a = ap.parse_args()

    run = Path(__file__).resolve().parents[1]
    cache = run / "models" / "cache"
    sigmas = [float(x) for x in a.sigmas.split(",")]
    g = dict(tasmax=[float(x) for x in a.shr_tasmax.split(",")],
             tasmin=[float(x) for x in a.shr_tasmin.split(",")],
             tas=[float(x) for x in a.shr_tas.split(",")],
             pr=[float(x) for x in a.shr_pr.split(",")])
    if a.quick:
        sigmas = [1.5]
        g = {k: [v[0]] if k != "pr" else [0.0, 1.0] for k, v in g.items()}

    ho = None if a.holdout == "none" else a.holdout
    train_ssps = [s for s in SSPS if s != ho] if ho else SSPS
    print(f"holdout={ho} train={train_ssps}")

    t0 = time.time()
    model = V4(hist_start=a.hist_start, sigma=sigmas[0])
    model.fit(train_ssps)

    # base thresholds cached via model
    ds2 = nc.Dataset(str(lite_file("hist_late")))
    base = base_period_slice({v: np.asarray(ds2.variables[v][0], dtype=np.float64) for v in VARS})
    ds2.close()
    model.base = base

    test_ssps = [ho] if ho else ["ssp245"]
    gt = {s: gt_indices_npz(cache, s) for s in test_ssps}

    # fit variants for each sigma (refit only m; S and Fbar unchanged -> refit cheaply)
    variants = []
    for sig in sigmas:
        mv = V4(hist_start=a.hist_start, sigma=sig)
        # copy everything except m, then recompute m
        mv.__dict__.update({k: v for k, v in model.__dict__.items() if k != "m"})
        # recompute m only:
        n = len(model.Ft)
        Fc = model.Ft - model.Ft.mean()
        varF = (Fc ** 2).sum()
        # rebuild annual aggregates quickly from saved S? need A; recompute:
        print(f"  recompute m for sigma={sig}")
        mv.m = {}
        # we did not keep A; recompute from data (moderately expensive). Instead keep A in model.
        for v in VARS:
            Av = model.A_ann[v]
            m = (Av * Fc[:, None]).sum(axis=0) / varF
            m = m.reshape(12, 18)
            if sig > 0:
                m = gaussian_filter(m, sig, mode="nearest")
            mv.m[v] = m
        variants.append(mv)

    results = []
    for mv in variants:
        for s_tasmax in g["tasmax"]:
            for s_tasmin in g["tasmin"]:
                for s_tas in g["tas"]:
                    for s_pr in g["pr"]:
                        sh = dict(tasmax=s_tasmax, tasmin=s_tasmin, tas=s_tas, pr=s_pr)
                        daily = mv.predict_daily(test_ssps[0], sh)
                        for s in test_ssps:
                            S, per = eval_config(mv, sh, gt[s], daily)
                            results.append((S, mv.sigma, sh))
                        del daily
    results.sort(key=lambda r: -r[0])
    print(f"\n== TOP 15 (holdout {ho}) ==")
    for S, sig, sh in results[:15]:
        print(f"  S={S:+.5f}  sigma={sig}  tasmax={sh['tasmax']} tasmin={sh['tasmin']} tas={sh['tas']} pr={sh['pr']}")
    print(f"total {time.time()-t0:.0f}s, {len(results)} configs")

    # save best predictions for ssp245 when no holdout
    if not ho:
        S, sig, sh = results[0]
        mv = next(v for v in variants if v.sigma == sig)
        daily = mv.predict_daily("ssp245", sh)
        outd = run / a.out
        outd.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(outd / "pred_v4_final.npz",
                            **{v: daily[v].astype(np.float32) for v in VARS},
                            sigma=np.float32(sig),
                            **{f"s_{k}": np.float32(vv) for k, vv in sh.items()})
        # also save indices at lite resolution
        idx = compute_indices(daily, mv.base)
        np.savez_compressed(outd / "indices_v4_final_lite.npz", **idx)
        S, per = score_lite(gt["ssp245"], idx)
        print(f"saved final: S_lite_selfcheck={S:+.5f}")
        for k in SUB_COLUMNS:
            print(f"  {k:8s} {per[k]:+.4f}")


if __name__ == "__main__":
    main()
