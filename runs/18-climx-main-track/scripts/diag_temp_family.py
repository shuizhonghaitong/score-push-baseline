"""Diagnose temp secondary-index gaps on a CV fold:
- family range vs target range per index
- single-index (ep,ec) fit score vs joint capped-loss fit vs no-wobble baseline
- T-only series score (ceiling reference)
"""
import sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_v7 import (V7, EPS_T, _quant, _med3, VALID_VAR, CAP, ABSFLOOR_TEMP,
                      build_model_cached, noise_scales, load_F)
from train_v6 import fit_targets, indices_temp
import train_v4 as t4
from rnNSE_lite import score_lite
from climx_indices import compute_indices
from climx_common import SUB_COLUMNS

HO = sys.argv[1] if len(sys.argv) > 1 else "ssp370"
train_ssps = [s for s in t4.SSPS if s != HO]
print(f"fold={HO} train={train_ssps}", flush=True)
m6 = build_model_cached(train_ssps, hist_start=1961)
ns = noise_scales(train_ssps)
m = V7(m6.S, m6.wob_max, m6.wob_min, m6.pr_years, m6.base,
       m6.th_wsdi, m6.th_csdi, m6.th_r95p, noise=ns)
F = load_F()
T = fit_targets(train_ssps, F, F[HO], h=0.15, min_n=8, smooth=2.0, tau_age=0.0)
N, N_DAYS = 86, 31389
gt = t4.gt_indices_npz(Path("models/cache"), HO)

def score4(field, other, keys, is_max):
    daily = dict(other)
    daily["tasmax" if is_max else "tasmin"] = field
    idx = compute_indices(daily, m.base)
    out = {}
    for k in keys:
        out[k] = score_lite(gt, {kk: idx[kk] if kk in keys else np.zeros((86,12,18)) for kk in SUB_COLUMNS})[1][k]
    return out

def case(var, is_max, keys, th, other):
    S_v = m.S[var]
    wob = m.wob_max if is_max else m.wob_min
    E = EPS_T; ne = len(E)
    outs = {k: np.empty((N_DAYS, 12, 18), np.float32) for k in keys[1:]}
    out_joint = np.empty((N_DAYS, 12, 18), np.float32)
    out_base = np.empty((N_DAYS, 12, 18), np.float32)
    ratio = {k: [] for k in keys[1:]}
    t0 = time.time()
    for i in range(12):
        for j in range(18):
            Sc = S_v[:, i, j]; wc = wob[:, i, j]; th_ = th[:, i, j]
            wp = np.maximum(wc, 0.0); wn = np.minimum(wc, 0.0)
            Tq = np.stack([_quant(T[k][:, i, j], ABSFLOOR_TEMP[k]) for k in keys], axis=-1)
            Tm = Tq[:, 0]
            base_c = (Sc[None, None, :] + E[:, None, None]*wp[None, None, :]
                      + E[None, :, None]*wn[None, None, :])
            M = base_c.max(axis=2) if is_max else base_c.min(axis=2)
            tab0 = indices_temp(base_c.reshape(ne*ne, 365),
                                th_ if is_max else None, None if is_max else th_)
            fam = tab0[:, 1:].max(axis=0) - tab0[:, 1:].min(axis=0)
            trange = Tq[:, 1:].max(axis=0) - Tq[:, 1:].min(axis=0)
            for a, k in enumerate(keys[1:]):
                ratio[k].append(float(trange[a]) / max(fam[a], 1e-9))
            bs = {k: np.empty(N, dtype=int) for k in ["__joint__"] + keys[1:]}
            for t in range(N):
                fields = base_c + (Tm[t] - M)[:, :, None]
                tab = indices_temp(fields.reshape(ne*ne, 365),
                                   th_ if is_max else None, None if is_max else th_)
                e2 = (tab[:, 1:] - Tq[t, 1:][None, :]) ** 2
                sig2 = np.maximum(tab0[:, 1:].var(axis=0), 1e-12)
                loss = np.minimum(e2 / sig2[None, :], CAP).sum(axis=1)
                bs["__joint__"][t] = int(np.argmin(loss))
                for a, k in enumerate(keys[1:]):
                    bs[k][t] = int(np.argmin(e2[:, a]))
            for name in list(bs):
                b = _med3(bs[name])
                epi = (b // ne).astype(int); eci = (b % ne).astype(int)
                tgt = out_joint if name == "__joint__" else (out_base if name == "__base__" else outs[name])
                for t in range(N):
                    nn = 364 if t == N-1 else 365
                    d = Tm[t] - M[epi[t], eci[t]]
                    field = Sc + d + E[epi[t]]*wp + E[eci[t]]*wn
                    tgt[t*365:t*365+nn, i, j] = field[:nn].astype(np.float32)
            for t in range(N):
                nn = 364 if t == N-1 else 365
                out_base[t*365:t*365+nn, i, j] = (Sc + (Tm[t] - M[0, 0]))[:nn].astype(np.float32)
    print(f"  {var} fields {time.time()-t0:.0f}s  median trange/family_range: "
          + " ".join(f"{k}={np.median(ratio[k]):.2f}" for k in keys[1:]), flush=True)
    res = {}
    for k in keys[1:]:
        res[k] = score4(outs[k], other, keys, is_max)[k]
    res["JOINT"] = {k: v for k, v in score4(out_joint, other, keys, is_max).items()}
    res["BASE"] = {k: v for k, v in score4(out_base, other, keys, is_max).items()}
    print(f"  {var} single/joint/base: " + " ".join(f"{k}: s={res[k]:+.3f} j={res['JOINT'][k]:+.3f} b={res['BASE'][k]:+.3f}" for k in keys[1:]), flush=True)
    return res

# simpler neutral fields: constant S over all days
other_max = {v: np.repeat(m.S[v], 86, axis=0)[:N_DAYS].astype(np.float32) for v in ["tas", "tasmin", "pr"]}
res_max = case("tasmax", True, ["TXx", "SU", "ID", "WSDI"], m.th_wsdi, other_max)
other_min = {v: np.repeat(m.S[v], 86, axis=0)[:N_DAYS].astype(np.float32) for v in ["tas", "tasmax", "pr"]}
res_min = case("tasmin", False, ["TNn", "TR", "FD", "CSDI"], m.th_csdi, other_min)

# T-only ceiling
to = {}
for k in ["TXx", "SU", "ID", "WSDI", "TNn", "TR", "FD", "CSDI"]:
    idx = {kk: (np.broadcast_to(T[k][:, None, None], (86, 12, 18)).astype(np.float32) if kk == k else np.zeros((86, 12, 18))) for kk in SUB_COLUMNS}
    to[k] = score_lite(gt, idx)[1][k]

print(f"\nfold {HO}:")
print(f"{'index':8s} {'T-only':>8s} {'no-wobble':>9s} {'single':>8s} {'joint':>8s}")
for k in ["TXx", "SU", "ID", "WSDI", "TNn", "TR", "FD", "CSDI"]:
    row = res_max if k in res_max else res_min
    print(f"{k:8s} {to[k]:+8.3f} {row['BASE'][k]:+9.3f} {row[k]:+8.3f} {row['JOINT'][k]:+8.3f}")
