"""Temporal CV: fit on years 2015-2057 of {126,370}, predict 2058-2100 of each.
Replicates emulate()'s math on custom row sets (out-of-time, in-domain)."""
import sys, time
import numpy as np
sys.path.insert(0, "scripts")
import fullres_linear as fl

t0 = time.time()
USE_GATE = "nogate" not in sys.argv
USE_QUAD = "noquad" not in sys.argv
USE_AERO = "aero" in sys.argv
SHRINK = 0.0
for a in sys.argv:
    if a.startswith("shrink="):
        SHRINK = float(a.split("=")[1])
scens = ["ssp126", "ssp370"]

_BLK = {}

def shrink_cols(theta):
    """theta (C,d): shrink trend cols (1..) toward lite-block means; keep intercept."""
    if SHRINK <= 0:
        return theta
    if "blk" not in _BLK:
        _BLK["blk"] = fl._block_map_full_to_lite()[ci]
    blk = _BLK["blk"]
    for j in range(1, theta.shape[1]):
        col = theta[:, j]
        bm = np.zeros(576); cnt = np.zeros(576)
        np.add.at(bm, blk, col); np.add.at(cnt, blk, 1.0)
        bm /= np.maximum(cnt, 1.0)
        theta[:, j] = (1 - SHRINK) * col + SHRINK * bm[blk]
    return theta
data = {s: fl.load_full(s) for s in scens}
ci = fl._land_cells(); C = ci.size
N = 86
T = 43
train_years = np.arange(T)
test_years = np.arange(T, N)
ridge = 1e-3
IDX_RIDGE = 1e-2
for a in sys.argv:
    if a.startswith("idxridge="):
        IDX_RIDGE = float(a.split("=")[1])

# per-year anchor across the two scenarios (same as emulate)
lbc_all = np.stack([data[s]["lbc"] for s in scens], axis=1)  # (86,2,4)
anchor = lbc_all[:, 0, :].mean(axis=0)  # (4,) 2015 common start

rows = []
for s in scens:
    rows += [(s, t) for t in train_years]
n = len(rows)
yr = np.array([t for _, t in rows])
Lc = np.stack([data[s]["lbc"][t] for s, t in rows], axis=0) - anchor  # (n,4)
Vb, kb = fl._lbc_basis(Lc)
Z = Lc @ Vb
zs = np.sqrt((Z ** 2).sum(0) / n) + 1e-30
Z = Z / zs
if USE_AERO:
    At = np.vstack([fl._aero_series(s)[t] for s, t in rows])  # (n,6)
    am, asd = At.mean(0), At.std(0) + 1e-30
    An = (At - am) / asd
    aero_test = {s: (fl._aero_series(s)[test_years] - am) / asd for s in scens}
zcols = [Z[:, q] for q in range(kb)]
if USE_QUAD:
    Q = np.stack([zcols[i] * zcols[j] for i in range(kb) for j in range(i, kb)], axis=1)
    qs = np.sqrt((Q ** 2).sum(0) / n) + 1e-30
    Qs = Q / qs
else:
    Qs = np.empty((n, 0)); qs = np.empty(0)
Xd = np.hstack([np.ones((n, 1)), Z, Qs] + ([An] if USE_AERO else []))

# field means / anomalies at train rows
field_mean, field_anom = {}, {}
for f in fl.FIELDS6:
    A = np.stack([data[s]["mean"][f][t].ravel()[ci] for s, t in rows], axis=0).astype(np.float64)
    mA = A.mean(0, keepdims=True)
    field_mean[f] = mA
    field_anom[f] = A - mA
theta_field = {f: shrink_cols(fl._pinv_solve(Xd, field_anom[f], ridge)) for f in fl.FIELDS6}
print(f"[t] fit fields {time.time()-t0:.0f}s n={n} shrink={SHRINK}", flush=True)

# test xrow
zr = np.stack([data["ssp370"]["lbc"][t] for t in test_years], axis=0)  # placeholder
def xrow_for(scen):
    zt = (data[scen]["lbc"][test_years] - anchor) @ Vb / zs
    if USE_QUAD:
        qt = np.stack([zt[:, i] * zt[:, j] for i in range(kb) for j in range(i, kb)], axis=1) / qs
        return np.hstack([np.ones((len(test_years), 1)), zt, qt])
    return np.hstack([np.ones((len(test_years), 1)), zt])

xrow = {s: xrow_for(s) for s in scens}
if USE_AERO:
    xrow = {s: np.hstack([xrow[s], aero_test[s]]) for s in scens}
field_pred = {}
for f in fl.FIELDS6:
    field_pred[f] = {}
    for s in scens:
        field_pred[f][s] = field_mean[f].T + (xrow[s] @ theta_field[f].T).T  # (C, 43)

mask = fl.ar6_mask_full()
lats = np.linspace(-90.0, 90.0, 192)
cos_w = np.cos(np.deg2rad(lats))
regions = np.unique(mask[mask >= 0])
eps = 1e-6

def score_years(gt, pred):
    """gt/pred (86,192,288); score over test_years only."""
    per = {}
    for col in fl.IDX_NAMES:
        g = gt[col][test_years].astype(np.float64)
        p = pred[col][test_years].astype(np.float64)
        var = g.var(0); mse = ((g - p) ** 2).mean(0)
        valid = var > eps
        r2 = np.where(valid, 1 - mse / np.maximum(var, eps), np.nan)
        nnse = np.where(valid, r2 / (2 - r2), np.nan)
        rs = []
        for rc in regions:
            inr = (mask == rc) & valid
            w = cos_w[:, None] * inr
            tw = w.sum()
            if tw < 1e-12:
                continue
            rs.append(float(np.sum(nnse[inr] * w[inr]) / tw))
        per[col] = float(np.mean(rs)) if rs else 0.0
    return float(np.mean(list(per.values()))), per

preds = {}
for s in scens:
    preds[s] = None
for name in fl.IDX_NAMES:
    P = fl.IDX_PREDICTORS[name]
    Y = np.stack([data[sc]["idx"][name][t].ravel()[ci] for sc, t in rows], axis=0).astype(np.float64)
    L = len(P)
    if L > 1:
        Gm = None
        for f in P:
            v = field_anom[f].mean(axis=1)
            Gm = v[:, None] if Gm is None else np.hstack([Gm, v[:, None]])
        Gc = Gm - Gm.mean(0)
        ev = np.linalg.svd(Gc, compute_uv=False) ** 2
        kp = 1
        for j in range(1, L):
            kp = j + 1
            if ev[:kp].sum() >= 0.98 * ev.sum():
                break
        _, _, Vt = np.linalg.svd(Gc, full_matrices=False)
        W = Vt[:kp].T
    else:
        kp, W = L, np.eye(L)
    Zn = np.hstack([np.ones((n, 1)), Z])
    Azzinv = np.linalg.pinv(Zn.T @ Zn)
    coef = (Azzinv @ (Zn.T @ Y)).T
    fitz = Zn @ coef.T
    ss_res = ((Y - fitz) ** 2).sum(0)
    ss_tot = ((Y - Y.mean(0)) ** 2).sum(0)
    gate = np.clip(1 - ss_res / np.maximum(ss_tot, 1e-12), 0.0, 1.0) if USE_GATE else np.ones(C)
    del fitz, coef

    parts, parts_pred = [], {}
    ztest = {s: xrow[s][:, 1:1 + kb] for s in scens}
    for q in range(kb):
        parts.append(Z[:, q][:, None])
        for s in scens:
            parts_pred.setdefault(s, []).append(ztest[s][:, q][:, None])
    if USE_AERO:
        for q in range(An.shape[1]):
            parts.append(An[:, q][:, None])
            for s in scens:
                parts_pred[s].append(aero_test[s][:, q][:, None])
    qtest = {s: xrow[s][:, 1 + kb:] for s in scens}
    for q in range(Qs.shape[1]):
        parts.append(Qs[:, q][:, None])
        for s in scens:
            parts_pred[s].append(qtest[s][:, q][:, None])
    for q in range(kp):
        part = None
        for f, w in zip(P, W[:, q]):
            if w == 0.0:
                continue
            part = field_anom[f] * w if part is None else part + field_anom[f] * w
        sc = part.std() + 1e-30
        parts.append(part / sc)
        for s in scens:
            partp = None
            for f, w in zip(P, W[:, q]):
                if w == 0.0:
                    continue
                partp = (field_pred[f][s] - field_mean[f].T) * w if partp is None else partp + (field_pred[f][s] - field_mean[f].T) * w
            parts_pred[s].append(partp / sc)
    d = 1 + len(parts)
    G = np.empty((C, d, d)); b = np.empty((C, d))
    b[:, 0] = Y.sum(0); G[:, 0, 0] = float(n)
    for j in range(d - 1):
        b[:, 1 + j] = (parts[j] * Y).sum(0)
        G[:, 0, 1 + j] = parts[j].sum(0)
        G[:, 1 + j, 0] = G[:, 0, 1 + j]
    for j in range(1, d):
        for kk in range(j, d):
            g = (parts[j - 1] * parts[kk - 1]).sum(0)
            G[:, j, kk] = g
            if j != kk:
                G[:, kk, j] = g
    ident = np.eye(d)[None]
    try:
        theta = np.linalg.solve(G + IDX_RIDGE * ident, b[:, :, None])[..., 0]
    except np.linalg.LinAlgError:
        theta = np.linalg.lstsq((G + IDX_RIDGE * ident).reshape(-1, d, d), b.reshape(C, d), rcond=None)[0].reshape(C, d)
    theta = shrink_cols(theta)
    for s in scens:
        val = np.empty((len(test_years), C))
        val[:] = theta[:, 0]
        trend = np.zeros_like(val)
        for j, pp in enumerate(parts_pred[s]):
            if pp.shape[-1] == 1:
                trend += pp.reshape(-1, 1) * theta[:, 1 + j]
            else:
                trend += pp.T * theta[:, 1 + j][None, :]
        val += trend * gate
        full = np.zeros((N, 192, 288))
        full[T:].reshape(T, -1)[:, ci] = val
        if preds[s] is None:
            preds[s] = {}
        preds[s][name] = full

tag = f"gate={USE_GATE} quad={USE_QUAD} aero={USE_AERO} shrink={SHRINK} idxridge={IDX_RIDGE}"
# debug probe (recompute TXx theta by re-running the last-index solve is not possible; use SDII theta as sanity + correct slicing)
name = "TXx"
c = ci[0]
gt = data["ssp370"]["idx"][name][test_years].reshape(T, -1)[:, c]
pv = preds["ssp370"][name][test_years].reshape(T, -1)[:, c]
print(f"[dbg] cell {c} 370 holdout: GT {gt[0]:.2f}->{gt[-1]:.2f} (std {gt.std():.2f})  P {pv[0]:.4f}->{pv[-1]:.4f} (std {pv.std():.4f})", flush=True)
gt2 = data["ssp370"]["idx"][name][:T].reshape(T, -1)[:, c]
print(f"[dbg] same cell 370 TRAIN: GT {gt2[0]:.2f}->{gt2[-1]:.2f} (std {gt2.std():.2f})", flush=True)
for s in scens:
    S, per = score_years(data[s]["idx"], preds[s])
    print(f"\n[t] temporal holdout 2058-2100 [{tag}] scen={s} S={S:+.4f}", flush=True)
    for k in fl.IDX_NAMES:
        print(f"    {k:8s} {per[k]:+.4f}", flush=True)
print(f"[t] done {time.time()-t0:.0f}s", flush=True)
