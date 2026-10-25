"""V9: V8 + precipitation blend stage (two template years, weight w, offset L0)
+ optional constant targets for unreliable index series (CSDI by default)
+ tunable CAP for the capped per-index losses."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_v8 import (
    V8, build_templates_cached, indices_temp2, C, HE_LEN,
)
from train_v7 import (
    _quant, _med3, VALID_VAR, CAP, ABSFLOOR_TEMP, D_GSL,
    noise_scales, build_model_cached,
)
from train_v6 import fit_targets, load_F, N, N_DAYS, indices_gsl, indices_pr
from climx_common import (
    SUB_COLUMNS, make_submission_parquet, nearest_upsample_lite_to_full,
)
from climx_indices import compute_indices, KG_M2_S_TO_MM_DAY
from rnNSE_lite import score_lite
import train_v4 as t4


class V9(V8):
    def __init__(self, S, wob_max, wob_min, pr_years, base, th_wsdi, th_csdi,
                 th_r95p, noise=None, templates=None, cap=2.0,
                 const_temp_keys=(), const_pr_keys=(), sec_scale=None):
        super().__init__(S, wob_max, wob_min, pr_years, base, th_wsdi, th_csdi,
                         th_r95p, noise=noise, templates=templates)
        self.cap = cap
        self.const_temp_keys = set(const_temp_keys)
        self.const_pr_keys = set(const_pr_keys)
        self.sec_scale = sec_scale

    # ---------------- temp (from V8._temp_field_t, with const targets) ----
    def _temp_field_t(self, T, var, quant=True):
        is_max = var == "tasmax"
        keys = ["TXx", "SU", "ID", "WSDI"] if is_max else ["TNn", "TR", "FD", "CSDI"]
        th = self.th_wsdi if is_max else self.th_csdi
        tpl = self.templates[var]
        ny = tpl.shape[0]
        thc = th.reshape(365, C).T.copy()
        thc_rep = np.repeat(thc, ny, axis=0)
        tpl_tab = tpl.transpose(0, 2, 3, 1).reshape(ny, C, 365).astype(np.float64)
        base_tab = tpl_tab.reshape(ny * C, 365)
        main_c = (base_tab.max(axis=1) if is_max else base_tab.min(axis=1)).reshape(ny, C)
        tab0 = indices_temp2(base_tab, thc_rep, is_max).reshape(ny, C, 4)
        poolmad = 1.4826 * np.median(np.abs(tab0 - tab0.mean(axis=0)[None]), axis=0)
        if self.sec_scale is None:
            sig2 = np.maximum(poolmad[:, 1:] ** 2, 1e-9)
        else:
            s = np.atleast_1d(np.asarray(self.sec_scale, dtype=np.float64))
            if s.size == 1:
                sig2 = np.full((C, 3), float(s[0]) ** 2)
            else:
                sig2 = np.broadcast_to(s[None, :], (C, 3)).copy() ** 2
            sig2 = np.maximum(sig2, 1e-9)
        Tq = np.stack([(self._target_k(T[k], k, quant)).reshape(N, C) for k in keys],
                      axis=-1)  # (86,216,4)
        b = np.empty((N, C), dtype=np.int64)
        t0 = time.time()
        for t in range(N):
            d = (Tq[t, :, 0][None, :] - main_c).reshape(ny * C)
            tab = indices_temp2(base_tab + d[:, None], thc_rep, is_max).reshape(ny, C, 4)
            e2 = (tab[:, :, 1:] - Tq[t][None, :, 1:]) ** 2
            loss = np.minimum(e2 / sig2[None, :, :], self.cap).sum(axis=2)
            b[t] = np.argmin(loss, axis=0)
            if t % 20 == 0:
                print(f"      {var} t={t}/{N} {time.time()-t0:.0f}s", flush=True)
        bc = b.copy()
        for c in range(C):
            bc[:, c] = _med3(b[:, c])
        out = np.empty((N_DAYS, 12, 18), np.float32)
        t0 = time.time()
        ar = np.arange(C)
        for t in range(N):
            nn = 364 if t == N - 1 else 365
            selc = bc[t]
            d = (Tq[t, :, 0] - main_c[selc, ar])
            sel = tpl_tab[selc, ar][:, :nn] + d[:, None]
            out[t * 365:t * 365 + nn] = sel.T.reshape(nn, 12, 18).astype(np.float32)
        print(f"      {var} materialize {time.time()-t0:.0f}s", flush=True)
        return out

    def _target_k(self, Tk, k, quant=True):
        """Per-cell target curve; constant (quantized mean) for const keys."""
        if k in self.const_temp_keys:
            c = np.broadcast_to(Tk.mean(axis=0)[None, :, :], (N,) + Tk.shape[1:])
            return _quant(c, ABSFLOOR_TEMP[k]) if quant else c
        return _quant(Tk, ABSFLOOR_TEMP[k]) if quant else Tk

    # ---------------- single-year (a, L0) pr fit (V7 logic) ---------------
    def _single_year_f(self, T, tab, sig2, mask, Tbar, active):
        ny = self.pr_years.shape[0]
        keys_n = 6
        e2 = (tab - Tbar[None]) ** 2
        lossbar = np.minimum(e2 / sig2[None], self.cap).sum(axis=2)
        lossbar = np.where(mask, lossbar, np.inf)
        b0 = lossbar.argmin(axis=0)
        ia0 = (b0 // ny).astype(int)
        iy0 = (b0 % ny).astype(int)
        ia0 = np.where(active, ia0, 3)
        iy0 = np.where(active, iy0, 0)
        a0 = np.asarray(self.PR_A_GRID)[ia0]
        ci = np.arange(12)[:, None].repeat(18, axis=1).reshape(216)
        cj = np.arange(18)[None, :].repeat(12, axis=0).reshape(216)
        flat = (np.arange(ny)[:, None] * 365 + np.arange(365)[None, :]) * 216
        flat = flat[:, :, None] + ci[None, None, :] * 18 + cj[None, None, :]
        ar = np.arange(365)[None, :]
        cc = np.arange(216)[:, None]
        fbase = self.pr_years.reshape(-1)[flat[iy0[:, None], ar, cc]] * a0[:, None]
        ac = np.arange(0.6, 1.6 + 1e-9, 0.1)
        Lc = np.arange(-4.0, 4.0 + 1e-9, 1.0) / KG_M2_S_TO_MM_DAY
        nAC = len(ac) * len(Lc)
        thc = self.th_r95p.reshape(216, 365)
        fields = (fbase[:, None, None, :] * ac[None, :, None, None]
                  + Lc[None, None, :, None])
        fields = np.maximum(fields.reshape(216 * nAC, 365), 0.0)
        r = indices_pr(fields, np.repeat(thc, nAC, axis=0)).reshape(216, nAC, keys_n)
        e2 = (r - Tbar[:, None, :]) ** 2
        loss = np.minimum(e2 / sig2[:, None, :], self.cap).sum(axis=2)
        b = loss.argmin(axis=1)
        ia_c = (b // len(Lc)).astype(int)
        iL_c = (b % len(Lc)).astype(int)
        af = np.arange(-0.10, 0.10 + 1e-9, 0.02)
        Lf = np.arange(-1.0, 1.0 + 1e-9, 0.2) / KG_M2_S_TO_MM_DAY
        a_f = ac[ia_c][:, None] + af[None, :]
        L_f = Lc[iL_c][:, None] + Lf[None, :]
        nAF = a_f.shape[1] * L_f.shape[1]
        fields = (fbase[:, None, None, :] * a_f[:, :, None, None]
                  + L_f[:, None, :, None])
        fields = np.maximum(fields.reshape(216 * nAF, 365), 0.0)
        r = indices_pr(fields, np.repeat(thc, nAF, axis=0)).reshape(216, nAF, keys_n)
        e2 = (r - Tbar[:, None, :]) ** 2
        loss = np.minimum(e2 / sig2[:, None, :], self.cap).sum(axis=2)
        b2 = loss.argmin(axis=1)
        ia2 = (b2 // L_f.shape[1]).astype(int)
        iL2 = (b2 % L_f.shape[1]).astype(int)
        a_sel = a_f[np.arange(216), ia2]
        L0_sel = L_f[np.arange(216), iL2]
        a_sel = np.where(active, a_sel, 1.0)
        L0_sel = np.where(active, L0_sel, 0.0)
        return np.maximum(fbase * a_sel[:, None] + L0_sel[:, None], 0.0)

    # ---------------- pr blend stage --------------------------------------
    def _pr_field(self, T, quant=True):
        keys = ["R95p", "CDD", "CWD", "Rx5day", "R10mm", "SDII"]
        ny = self.pr_years.shape[0]
        tab = np.empty((len(self.PR_A_GRID) * ny, 216, 6))
        t0 = time.time()
        for x, a in enumerate(self.PR_A_GRID):
            for y0 in range(ny):
                f = np.maximum(self.pr_years[y0].astype(np.float64) * a, 0.0)
                r = indices_pr(f.transpose(2, 0, 1).reshape(216, 365),
                               self.th_r95p.reshape(216, 365))
                tab[x * ny + y0] = r[:, [3, 1, 2, 0, 4, 5]]
        print(f"    pr template table {time.time()-t0:.0f}s", flush=True)

        poolmad = 1.4826 * np.median(np.abs(tab - tab.mean(axis=0)[None]), axis=0)
        poolmad = np.maximum(poolmad, 1e-3)
        sig2 = poolmad ** 2
        Tbar = np.stack([T[k].mean(axis=0).reshape(-1) for k in keys],
                        axis=-1).reshape(216, 6)
        varT = np.stack([T[k].var(axis=0).reshape(-1) for k in keys], axis=-1).reshape(216, 6)
        active = (varT > VALID_VAR).any(axis=1)
        active = np.where(active, 1.0, 0.0).astype(bool)

        T95 = T["R95p"].mean(axis=0).reshape(216)
        r95 = tab[:, :, 0]
        mask = (r95 >= 0.5 * T95[None]) & (r95 <= 2.0 * T95[None])
        nkeep = mask.sum(axis=0)
        loose = (r95 >= 0.25 * T95[None]) & (r95 <= 4.0 * T95[None])
        mask = np.where(nkeep[None] >= 16, mask, loose)
        mask = np.where((T95 > 5.0)[None], mask, True)
        print(f"    pr band filter kept median {int(np.median(nkeep))}/{tab.shape[0]}", flush=True)

        ci = np.arange(12)[:, None].repeat(18, axis=1).reshape(216)
        cj = np.arange(18)[None, :].repeat(12, axis=0).reshape(216)
        flat = (np.arange(ny)[:, None] * 365 + np.arange(365)[None, :]) * 216
        flat = flat[:, :, None] + ci[None, None, :] * 18 + cj[None, None, :]
        py_flat = self.pr_years.reshape(-1)
        thc = self.th_r95p.reshape(216, 365)
        ar = np.arange(365)[None, :]
        cc = np.arange(216)[:, None]

        # stage 1: (a, y1) per cell
        e2 = (tab - Tbar[None]) ** 2
        lossbar_raw = np.minimum(e2 / sig2[None], self.cap).sum(axis=2)
        lossbar = np.where(mask, lossbar_raw, np.inf)
        b0 = lossbar.argmin(axis=0)
        ia0 = (b0 // ny).astype(int)
        iy1 = (b0 % ny).astype(int)
        ia0 = np.where(active, ia0, 3)
        iy1 = np.where(active, iy1, 0)
        a1 = np.asarray(self.PR_A_GRID)[ia0]

        # stage 2: pick y2 (dissimilar wetness, near-optimal)
        loss_year = lossbar.reshape(len(self.PR_A_GRID), ny, 216).min(axis=0)  # (ny,216)
        L1 = loss_year[iy1, np.arange(216)]
        r95_3 = tab[:, :, 0].reshape(len(self.PR_A_GRID), ny, 216)
        r95_y1 = r95_3[ia0, iy1, np.arange(216)]
        diss_yr = (np.abs(r95_3 - r95_y1[None, None, :])
                   >= 0.15 * np.maximum(1.0, r95_y1)[None, None, :]).any(axis=0)  # (ny,216)
        not_y1 = np.arange(ny)[:, None] != iy1[None, :]
        cand = (loss_year <= L1[None] + np.maximum(0.25 * L1, 0.5)[None] + 1.0) & diss_yr & not_y1
        cand_all = diss_yr & not_y1
        cand = np.where(cand.any(axis=0)[None], cand, cand_all)
        yy2 = np.where(cand, loss_year, np.inf).argmin(axis=0)
        ia2c = ia0.copy()
        iy2 = yy2
        # best a for y2 (unmasked loss)
        for c in range(216):
            if iy2[c] == iy1[c] or not active[c]:
                continue
            seg = lossbar_raw.reshape(len(self.PR_A_GRID), ny, 216)[:, iy2[c], c]
            ia2c[c] = int(np.argmin(seg))
        a2 = np.asarray(self.PR_A_GRID)[ia2c]
        nblend = int((iy2 != iy1).sum())
        print(f"    pr y2 differs on {nblend}/216 cells", flush=True)

        f1 = py_flat[flat[iy1[:, None], ar, cc]] * a1[:, None]
        f2 = py_flat[flat[iy2[:, None], ar, cc]] * a2[:, None]

        # stage 3: (w, L0) fit on the blend
        wc = np.linspace(0.0, 1.0, 11)
        Lc = np.arange(-4.0, 4.0 + 1e-9, 0.5) / KG_M2_S_TO_MM_DAY
        nW, nL = len(wc), len(Lc)
        tgtv = Tbar
        thc_w = np.repeat(thc, nW * nL, axis=0)
        best_w = np.zeros(216); best_L = np.zeros(216)
        fields = (f1[:, None, None, :] * wc[None, :, None, None]
                  + f2[:, None, None, :] * (1.0 - wc)[None, :, None, None]
                  + Lc[None, None, :, None])
        fields = np.maximum(fields.reshape(216 * nW * nL, 365), 0.0)
        r = indices_pr(fields, thc_w).reshape(216, nW * nL, 6)
        e2 = (r - tgtv[:, None, :]) ** 2
        loss = np.minimum(e2 / sig2[:, None, :], self.cap).sum(axis=2)
        b = loss.argmin(axis=1)
        iw = (b // nL).astype(int)
        iL = (b % nL).astype(int)
        wf = wc[iw][:, None] + np.arange(-0.1, 0.1 + 1e-9, 0.02)[None, :]
        wf = np.clip(wf, 0.0, 1.0)
        Lf = Lc[iL][:, None] + np.arange(-1.0, 1.0 + 1e-9, 0.1) / KG_M2_S_TO_MM_DAY
        nWF = wf.shape[1] * Lf.shape[1]  # full (w, L0) grid around coarse pick
        fields = (f1[:, None, None, :] * wf[:, :, None, None]
                  + f2[:, None, None, :] * (1.0 - wf)[:, :, None, None]
                  + Lf[:, None, :, None])
        fields = np.maximum(fields.reshape(216 * nWF, 365), 0.0)
        r = indices_pr(fields, np.repeat(thc, nWF, axis=0)).reshape(216, nWF, 6)
        e2 = (r - tgtv[:, None, :]) ** 2
        loss = np.minimum(e2 / sig2[:, None, :], self.cap).sum(axis=2)
        b2 = loss.argmin(axis=1)
        w_sel = np.clip(wf[np.arange(216), (b2 // Lf.shape[1]).astype(int)], 0.0, 1.0)
        L_sel = Lf[np.arange(216), (b2 % Lf.shape[1]).astype(int)]
        year_f = np.maximum(f1 * w_sel[:, None] + f2 * (1.0 - w_sel)[:, None]
                            + L_sel[:, None], 0.0)
        print(f"    pr blend fit {time.time()-t0:.0f}s", flush=True)
        t0 = time.time()
        year_s = self._single_year_f(T, tab, sig2, mask, Tbar, active)
        rb = indices_pr(year_f, thc)
        rs = indices_pr(year_s, thc)
        eb = np.minimum((rb - Tbar) ** 2 / sig2, self.cap).sum(axis=1)
        es = np.minimum((rs - Tbar) ** 2 / sig2, self.cap).sum(axis=1)
        use_blend = eb < es
        year_f = np.where(use_blend[:, None], year_f, year_s)
        print(f"    pr blend used on {int(use_blend.sum())}/216 cells "
              f"({time.time()-t0:.0f}s)", flush=True)
        out = np.empty((N_DAYS, 12, 18), dtype=np.float32)
        t0 = time.time()
        for t in range(N):
            nn = 364 if t == N - 1 else 365
            out[t * 365:t * 365 + nn] = year_f[:, :nn].T.reshape(nn, 12, 18).astype(np.float32)
        print(f"    pr refinement {time.time()-t0:.0f}s", flush=True)
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holdout", default="ssp370")
    ap.add_argument("--h", type=float, default=0.15)
    ap.add_argument("--min-n", type=int, default=8)
    ap.add_argument("--smooth", type=float, default=2.0)
    ap.add_argument("--tau-age", type=float, default=0.0)
    ap.add_argument("--hist-start", type=int, default=1961)
    ap.add_argument("--no-quant", action="store_true")
    ap.add_argument("--cap", type=float, default=2.0)
    ap.add_argument("--sec-scale", default="none",
                    help="fixed per-secondary-index scale (days) for template-selection loss; "
                         "'none' = pool-mad; scalar (e.g. 20) or comma-3 (e.g. 30,30,30 for SU,ID,WSDI)")
    ap.add_argument("--use-t-for", default="",
                    help="comma-separated indices to take from T (forcing->index median) instead of "
                         "the emulator; default '' = pure V9 (emulator for all). e.g. TXx,TNn,WSDI,CSDI")
    ap.add_argument("--const", default="CSDI",
                    help="comma-separated index names to target as constants (e.g. CSDI)")
    a = ap.parse_args()

    ho = None if a.holdout == "none" else a.holdout
    train_ssps = [s for s in t4.SSPS if s != ho] if ho else t4.SSPS
    print(f"holdout={ho} train={train_ssps}")
    F = load_F()
    m6 = build_model_cached(train_ssps, hist_start=a.hist_start)
    t0 = time.time()
    ns = noise_scales(train_ssps)
    t0 = time.time()
    tpl = build_templates_cached(train_ssps, hist_start=a.hist_start)
    const = {k for k in a.const.split(",") if k}
    if a.sec_scale == "none":
        sec_scale = None
    else:
        parts = [float(x) for x in a.sec_scale.split(",")]
        sec_scale = parts[0] if len(parts) == 1 else parts
    print(f"sec_scale={sec_scale}", flush=True)
    m = V9(m6.S, m6.wob_max, m6.wob_min, m6.pr_years, m6.base,
           m6.th_wsdi, m6.th_csdi, m6.th_r95p, noise=ns, templates=tpl,
           cap=a.cap, const_temp_keys=const, const_pr_keys=const, sec_scale=sec_scale)
    scen = ho if ho else "test_forcings"
    F_test = F[scen]
    t0 = time.time()
    T = fit_targets(train_ssps, F, F_test, h=a.h, min_n=a.min_n, smooth=a.smooth,
                    tau_age=a.tau_age)
    print(f"  targets {time.time()-t0:.0f}s")
    t0 = time.time()
    daily = m.predict_daily(T, quant=not a.no_quant)
    print(f"  fields {time.time()-t0:.0f}s")
    idx = compute_indices(daily, m.base)
    if ho:
        np.savez_compressed(Path("models/cache") / f"lite_cv_idx_v9_{ho}.npz",
                            **{k: idx[k].astype(np.float32) for k in SUB_COLUMNS})
        print(f"wrote models/cache/lite_cv_idx_v9_{ho}.npz", flush=True)
        gt = t4.gt_indices_npz(Path("models/cache"), ho)
        S, per = score_lite(gt, idx)
        print(f"\n== V9 CV holdout={ho}: S={S:+.5f} ==")
        for k in SUB_COLUMNS:
            print(f"  {k:8s} {per[k]:+.4f}")
    else:
        outd = Path("submissions")
        outd.mkdir(exist_ok=True)
        use_t = {k for k in a.use_t_for.split(",") if k}
        assert use_t <= set(SUB_COLUMNS), f"unknown index in --use-t-for: {use_t - set(SUB_COLUMNS)}"
        combined = {k: (T[k] if k in use_t else idx[k]) for k in SUB_COLUMNS}
        lite_cache = Path("models/cache")
        np.savez_compressed(lite_cache / "lite_test_idx_v9.npz",
                            **{k: combined[k].astype(np.float32) for k in SUB_COLUMNS})
        print("wrote lite cache models/cache/lite_test_idx_v9.npz", flush=True)
        idx_full = {k: nearest_upsample_lite_to_full(combined[k].astype(np.float32))
                    for k in SUB_COLUMNS}
        p = make_submission_parquet(idx_full, outd / "submission_v9.parquet")
        print(f"  blend: emulator for {[k for k in SUB_COLUMNS if k not in use_t]}, T for {sorted(use_t)}")
        print("wrote", p)


if __name__ == "__main__":
    main()
