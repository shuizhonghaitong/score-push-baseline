# Use the Dependency Manager
# pip install rdkit==2026.3.3
# pip install numba
"""CASMI 2026 V2: entropy (Jensen-Shannon) spectral engine.

- Direct channel: JS-entropy similarity over a tight neutral-mass window,
  L1-normalized + entropy-flattened peaks, top-256, MZ_TOL matching.
- Analog channel: mass-shifted entropy similarity over +/-200 Da on the
  per-structure representative spectra; candidates scored by
  Tanimoto(fp, hit_fp) * sim^P.
Pre-cleaned flat library store + numba. CPU-only.
"""
import os, sys, time, glob, hashlib, re
from multiprocessing import Pool as MPool

import numpy as np
import pandas as pd
from numba import njit, prange

# ----------------------------------------------------------------------------
# constants
# ----------------------------------------------------------------------------
INT_FLOOR = 0.002
MAX_PEAKS = 256
MZ_TOL = 0.01
INT_POWER = 1.0
ENT_WEIGHT = True
PPM_WIN = 8.5
PPM_FALLBACK = (30.0, 100.0, 1000.0)
LIB_PPM_WIN = 8.5
ANALOG_DA_WIN = 200.0
N_ANALOG = 80
SIM_POWER = 4.0
ANALOG_MIN_SIM = 0.40
DIRECT_GATE = 0.60
W_ANALOG = 0.5
TOPN = 25

_H = 1.00782503207
_E = 0.00054857990
PROTON = _H - _E
_H2O = 18.01056466
_NH4 = 14.0030740048 + 4 * _H
_NA = 22.9897692809
_K = 38.96370668
_CL = 34.96885268
_CH2O2 = 12.000000 + 2 * _H + 2 * 15.9949146196
ADDUCTS = {
    "[M+H]+": (PROTON, +1),
    "[M+NH4]+": (_NH4 - _E, +1),
    "[M-H2O+H]+": (_H2O + PROTON, +1),
    "[M-2H2O+H]+": (2 * _H2O + PROTON, +1),
    "[M+Na]+": (_NA - _E, +1),
    "[M+K]+": (_K - _E, +1),
    "[M-H]-": (PROTON, -1),
    "[M-H2O-H]-": (_H2O + PROTON, -1),
    "[M+CH2O2-H]-": (_CH2O2 - _E, -1),
    "[M+Cl]-": (_CL - _E, -1),
}
_AMU = {"C": 12.0, "H": _H, "N": 14.0030740048, "O": 15.9949146196, "F": 18.99840322,
        "P": 30.97376163, "S": 31.97207100, "Cl": 34.96885268, "Br": 78.9183371,
        "I": 126.904473, "B": 11.0093054, "Si": 27.9769265325, "Se": 79.9165213,
        "Na": _NA, "K": _K, "Mg": 23.9850417, "Ca": 39.9625908, "Li": 7.0160034}
_FMLA_RE = re.compile(r"([A-Z][a-z]?)(\d*)")

def formula_mass(formula):
    if not formula:
        return np.nan
    try:
        m = 0.0
        for el, num in _FMLA_RE.findall(formula):
            m += (int(num) if num else 1) * _AMU[el]
        return m
    except KeyError:
        return np.nan

def neutral_mass(prec, adduct):
    car = ADDUCTS.get(adduct)
    if car is None:
        return np.nan
    try:
        prec = float(prec)
    except (TypeError, ValueError):
        return np.nan
    if not np.isfinite(prec) or prec <= 0:
        return np.nan
    return prec - car[1] * car[0]

# ----------------------------------------------------------------------------
# numba kernels (ported entropy-similarity engine)
# ----------------------------------------------------------------------------
@njit(cache=True, fastmath=True)
def _clean(mz, it, floor, topk, power, ent_weight):
    n = len(mz)
    if n == 0:
        return np.empty(0, np.float32), np.empty(0, np.float32)
    mx = 0.0
    for i in range(n):
        if it[i] > mx:
            mx = it[i]
    if mx <= 0 or not np.isfinite(mx):
        return np.empty(0, np.float32), np.empty(0, np.float32)
    thr = floor * mx
    c = 0
    for i in range(n):
        if it[i] >= thr:
            c += 1
    idx = np.empty(c, np.int64)
    j = 0
    for i in range(n):
        if it[i] >= thr:
            idx[j] = i
            j += 1
    if c > topk:
        v = np.empty(c, np.float32)
        for i in range(c):
            v[i] = it[idx[i]]
        o = np.argsort(v)[c - topk:]
        k2 = np.empty(topk, np.int64)
        for i in range(topk):
            k2[i] = idx[o[i]]
        k2.sort()
        idx = k2
        c = topk
    om = np.empty(c, np.float32)
    oi = np.empty(c, np.float32)
    s = 0.0
    for i in range(c):
        om[i] = mz[idx[i]]
        v = it[idx[i]] ** power
        oi[i] = v
        s += v
    if s > 0:
        for i in range(c):
            oi[i] /= s
    if ent_weight:
        S = 0.0
        for i in range(c):
            if oi[i] > 0:
                S -= oi[i] * np.log(oi[i])
        if S < 3.0:
            w = 0.25 + 0.25 * S
            s2 = 0.0
            for i in range(c):
                oi[i] = oi[i] ** w
                s2 += oi[i]
            if s2 > 0:
                for i in range(c):
                    oi[i] /= s2
    return om, oi

@njit(cache=True, fastmath=True)
def entropy_sim(qmz, qp, cmz, cp, tol):
    i = 0
    j = 0
    n = len(qmz)
    m = len(cmz)
    SA = 0.0
    for x in range(n):
        if qp[x] > 0:
            SA -= qp[x] * np.log(qp[x])
    SB = 0.0
    for x in range(m):
        if cp[x] > 0:
            SB -= cp[x] * np.log(cp[x])
    buf = np.empty(n + m, np.float64)
    b = 0
    while i < n and j < m:
        d = qmz[i] - cmz[j]
        if d < -tol:
            buf[b] = qp[i]
            i += 1
            b += 1
        elif d > tol:
            buf[b] = cp[j]
            j += 1
            b += 1
        else:
            buf[b] = qp[i] + cp[j]
            i += 1
            j += 1
            b += 1
    while i < n:
        buf[b] = qp[i]
        i += 1
        b += 1
    while j < m:
        buf[b] = cp[j]
        j += 1
        b += 1
    tot = 0.0
    for x in range(b):
        tot += buf[x]
    if tot <= 0:
        return 0.0
    SAB = 0.0
    for x in range(b):
        v = buf[x] / tot
        if v > 0:
            SAB -= v * np.log(v)
    return 1.0 - (2.0 * SAB - SA - SB) / np.log(4.0)

@njit(cache=True, fastmath=True, parallel=True)
def entropy_sim_shift(qmz, qp, cmz, cp, tol, shift):
    a = entropy_sim(qmz, qp, cmz, cp, tol)
    if shift > -0.001 and shift < 0.001:
        return a
    sm = np.empty(len(cmz), np.float32)
    for i in range(len(cmz)):
        sm[i] = cmz[i] + shift
    b = entropy_sim(qmz, qp, sm, cp, tol)
    return a if a > b else b

@njit(cache=True, fastmath=True, parallel=True)
def search_pre(qmz, qp, lo, hi, roff, rmz, rit, tol):
    out = np.zeros(hi - lo, np.float32)
    for k in prange(hi - lo):
        a = roff[lo + k]
        b = roff[lo + k + 1]
        if b <= a:
            continue
        out[k] = entropy_sim(qmz, qp, rmz[a:b], rit[a:b], tol)
    return out

@njit(cache=True, fastmath=True, parallel=True)
def search_shift_pre(qmz, qp, lo, hi, roff, rmz, rit, tol, shift):
    out = np.zeros(hi - lo, np.float32)
    for k in prange(hi - lo):
        a = roff[lo + k]
        b = roff[lo + k + 1]
        if b <= a:
            continue
        out[k] = entropy_sim_shift(qmz, qp, rmz[a:b], rit[a:b], tol, shift[k])
    return out

@njit(cache=True)
def clean_store(rows, off, allmz, allin, floor, topk, power, ent_weight):
    n = len(rows)
    lens = np.zeros(n + 1, np.int64)
    tmp_m = []
    tmp_i = []
    for k in range(n):
        r = rows[k]
        cm, cp = _clean(allmz[off[r]:off[r + 1]], allin[off[r]:off[r + 1]], floor, topk, power, ent_weight)
        tmp_m.append(cm)
        tmp_i.append(cp)
        lens[k + 1] = lens[k] + len(cm)
    om = np.empty(lens[n], np.float32)
    oi = np.empty(lens[n], np.float32)
    for k in range(n):
        om[lens[k]:lens[k + 1]] = tmp_m[k]
        oi[lens[k]:lens[k + 1]] = tmp_i[k]
    return lens, om, oi

# ----------------------------------------------------------------------------
# pool / fingerprints
# ----------------------------------------------------------------------------
_GEN = None
def _get_gen():
    global _GEN
    if _GEN is None:
        from rdkit.Chem import rdFingerprintGenerator
        _GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    return _GEN

def _fp_one(smi):
    from rdkit import Chem
    m = Chem.MolFromSmiles(str(smi))
    if m is None:
        return None
    try:
        bv = _get_gen().GetFingerprint(m).GetExplicitBitVector()
        w = np.zeros(32, np.uint32)
        for b in bv.GetOnBits():
            w[b // 32] |= np.uint32(1) << (b % 32)
        return w
    except Exception:
        return None

def _clean_chunk(args):
    c, off, allmz, allin = args
    return clean_store(c, off, allmz, allin, INT_FLOOR, MAX_PEAKS, INT_POWER, ENT_WEIGHT)

def tanimoto_words(a, b):
    inter = np.bitwise_and(a[:, None, :].astype(np.uint64),
                           b[None, :, :].astype(np.uint64)).sum(axis=2).astype(np.float64)
    ua = a.sum(axis=1).astype(np.float64)[:, None]
    ub = b.sum(axis=1).astype(np.float64)[None, :]
    union = ua + ub - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0).astype(np.float32)

def _pool_window(pool_mass, target):
    for p in (PPM_WIN,) + tuple(PPM_FALLBACK):
        tol = target * p * 1e-6
        lo = int(np.searchsorted(pool_mass, target - tol, 'left'))
        hi = int(np.searchsorted(pool_mass, target + tol, 'right'))
        if hi - lo >= TOPN:
            return np.arange(lo, hi)
    tol = target * 1000.0 * 1e-6
    return np.arange(int(np.searchsorted(pool_mass, target - tol, 'left')),
                     int(np.searchsorted(pool_mass, target + tol, 'right')))

# ----------------------------------------------------------------------------
# core
# ----------------------------------------------------------------------------
def extract_flat(tbl, mask=None):
    mzc = tbl.column('ms2_mzs').combine_chunks()
    itc = tbl.column('ms2_normalized_intensities').combine_chunks()
    off = mzc.offsets.to_numpy(zero_copy_only=False).astype(np.int64)
    allmz = mzc.values.to_numpy(zero_copy_only=False).astype(np.float32)
    allin = itc.values.to_numpy(zero_copy_only=False).astype(np.float32)
    if mask is None:
        return {'off': off, 'allmz': allmz, 'allin': allin}
    a = off[:-1][mask]
    b = off[1:][mask]
    if len(a) < len(off) - 1:
        allmz = np.concatenate([allmz[ai:bi] for ai, bi in zip(a, b)])
        allin = np.concatenate([allin[ai:bi] for ai, bi in zip(a, b)])
    return {'off': np.concatenate([[0], np.cumsum(b - a)]), 'allmz': allmz, 'allin': allin}

def pa_mask(mask):
    import pyarrow as pa
    return pa.array(mask)

def neutral_mass_array(prec, adducts):
    out = np.full(len(prec), np.nan, np.float64)
    for i, (p, a) in enumerate(zip(prec, adducts)):
        out[i] = neutral_mass(p, a)
    return out

def run_core(tr, te, out_path, n_mols=None, max_lib=None, pool_extra=None, logf=print,
             flat=None):
    """flat: optional dict with pre-extracted (off, allmz, allin) from pyarrow
    (much lighter than going through pandas object arrays)."""
    T0 = time.time()
    def log(msg):
        logf(f"[{time.time()-T0:7.1f}s] {msg}")

    if out_path is None:
        out_path = os.path.join('/kaggle/working' if os.path.isdir('/kaggle/working')
                                else os.path.dirname(os.path.abspath(__file__)), 'submission.csv')
    log(f"cpus={os.cpu_count()} train={len(tr):,}")

    nms = np.array([neutral_mass(p, a) for p, a in zip(tr.precursor_mz.to_numpy(), tr.adduct.to_numpy())], np.float64)
    valid = np.isfinite(nms)
    tr = tr[valid].reset_index(drop=True)
    nms = nms[valid]
    if max_lib and len(tr) > max_lib:
        tr = tr.iloc[:max_lib].reset_index(drop=True)
        nms = nms[:max_lib]
    log(f"library: {len(tr):,} spectra with finite neutral mass")

    # --- flat raw store (unsorted order = tr index)
    if flat is not None:
        off, allmz, allin = flat['off'], flat['allmz'], flat['allin']
    else:
        mzc = tr.ms2_mzs.to_numpy()
        itc = tr.ms2_normalized_intensities.to_numpy()
        off = np.zeros(len(tr) + 1, np.int64)
        allmz = np.concatenate([np.asarray(m, np.float32) for m in mzc]) if len(mzc) else np.empty(0, np.float32)
        allin = np.concatenate([np.asarray(v, np.float32) for v in itc]) if len(itc) else np.empty(0, np.float32)
        lens = np.array([len(np.asarray(m)) for m in mzc], np.int64)
        off = np.concatenate([[0], np.cumsum(lens)])
    keystr = tr.inchikey14.to_numpy()
    if 'ms2_mzs' in tr.columns:
        tr = tr.drop(columns=['ms2_mzs', 'ms2_normalized_intensities'])

    # --- order by neutral mass
    order = np.argsort(nms, kind='stable')
    snm = nms[order]

    # --- pre-clean into flat store (parallel chunks via numba single call is serial;
    #     use chunked multiprocessing over row ranges)
    log("pre-cleaning library (numba)")
    t0 = time.time()
    rows = order
    ncpu = min(os.cpu_count() or 1, 16)
    # warmup the njit with a tiny store (in main process, once)
    _ = clean_store(np.array([0], np.int64), off[:2], allmz[:off[1]], allin[:off[1]],
                     INT_FLOOR, MAX_PEAKS, INT_POWER, ENT_WEIGHT)
    if ncpu > 1 and len(rows) > 10000:
        chunks = np.array_split(rows, ncpu)
        with MPool(ncpu) as mp:
            res = mp.map(_clean_chunk, [ (c, off, allmz, allin) for c in chunks ], chunksize=1)
        res = [r for r in res]
        parts = [r[1] for r in res]
        partsl = [r[2] for r in res]
        rmz = np.concatenate(parts) if parts else np.empty(0, np.float32)
        rit = np.concatenate(partsl) if partsl else np.empty(0, np.float32)
        roff = np.zeros(len(rows) + 1, np.int64)
        base = 0
        for r in res:
            seg = r[0]  # local offsets, len = chunk_len+1
            roff[base:base + len(seg)] = seg + base
            base += len(seg) - 1
    else:
        roff, rmz, rit = clean_store(rows, off, allmz, allin, INT_FLOOR, MAX_PEAKS, INT_POWER, ENT_WEIGHT)
    log(f"pre-clean done ({time.time()-t0:.0f}s), {rmz.size/1e6:.1f}M peaks")
    del allmz, allin, off

    lib = {'snm': snm, 'order': order, 'roff': roff, 'rmz': rmz, 'rit': rit, 'keystr': keystr[order]}

    # --- representative per structure (richest base peak), stored sorted by nm
    bpi = tr.base_peak_intensity.to_numpy(np.float64)
    bpi = np.where(np.isfinite(bpi), bpi, 0.0)
    best = {}
    for i in range(len(tr)):
        k = str(keystr[i])
        if k not in best or bpi[i] > bpi[best[k]]:
            best[k] = i
    rep_rows = np.array(sorted(best.values()))
    rep_pos = np.searchsorted(order, rep_rows)  # positions in sorted lib (order is a permutation; searchsorted on sorted values not valid -> build map)
    posmap = np.empty(len(order), np.int64)
    posmap[order] = np.arange(len(order))
    rep_sorted = posmap[rep_rows]
    rep_nm = snm[rep_sorted]
    ok = np.isfinite(rep_nm)
    rep_sorted, rep_nm, rep_key = rep_sorted[ok], rep_nm[ok], lib['keystr'][rep_sorted[ok]]
    o = np.argsort(rep_nm, kind='stable')
    rep_sorted, rep_nm, rep_key = rep_sorted[o], rep_nm[o], rep_key[o]
    # extract per-rep cleaned peaks
    # slice cleaned peaks for rep rows (no re-cleaning: store is already flattened)
    rep_starts = roff[rep_sorted]
    rep_ends = roff[rep_sorted + 1]
    rmz2 = np.concatenate([rmz[a:b] for a, b in zip(rep_starts, rep_ends)]) if len(rep_sorted) else np.empty(0, np.float32)
    rit2 = np.concatenate([rit[a:b] for a, b in zip(rep_starts, rep_ends)]) if len(rep_sorted) else np.empty(0, np.float32)
    r2off = np.zeros(len(rep_sorted) + 1, np.int64)
    r2off[1:] = np.cumsum(rep_ends - rep_starts)
    rep = {'snm': rep_nm, 'rmz': rmz2, 'rit': rit2, 'roff': r2off, 'keystr': rep_key}
    log(f"rep library {len(rep_sorted):,}")

    # --- candidate pool
    log("building candidate pool")
    psrc = tr[['inchikey14', 'normalized_smiles', 'molecular_formula']].drop_duplicates('inchikey14').reset_index(drop=True)
    if pool_extra is not None:
        psrc = pd.concat([psrc, pool_extra], ignore_index=True).drop_duplicates('inchikey14')
    pkeys = psrc.inchikey14.to_numpy()
    psmi = psrc.normalized_smiles.to_numpy()
    pmass = np.array([formula_mass(f) for f in psrc.molecular_formula.to_numpy()], np.float64)
    okp = np.isfinite(pmass)
    pkeys, psmi, pmass = pkeys[okp], psmi[okp], pmass[okp]
    log(f"pool: {len(pkeys):,} structures")

    from rdkit import RDLogger
    RDLogger.DisableLog('rdApp.*')
    _get_gen()  # warm generator in parent
    FW = 32
    t0 = time.time()
    if ncpu > 1:
        with MPool(ncpu) as mp:
            fpr = mp.map(_fp_one, psmi, chunksize=256)
    else:
        fpr = [_fp_one(s) for s in psmi]
    log(f"fingerprints ({time.time()-t0:.0f}s)")
    pool_fp = np.zeros((len(psmi), FW), np.uint32)
    for i, x in enumerate(fpr):
        if x is not None:
            pool_fp[i] = x
    del fpr

    po = np.argsort(pmass, kind='stable')
    pool = {'mass': pmass[po], 'smiles': psmi[po], 'keystr': pkeys[po], 'fp': pool_fp[po]}
    pool_k2i = {str(k): i for i, k in enumerate(pool['keystr'])}
    lib['poolpos'] = np.array([pool_k2i.get(str(k), -1) for k in lib['keystr']], np.int64)
    rep['poolpos'] = np.array([pool_k2i.get(str(k), -1) for k in rep['keystr']], np.int64)
    log("pool ready")

    # --- inference
    def predict_mid(g):
        specs = []
        for r in g.itertuples(index=False):
            nm = neutral_mass(r.precursor_mz, r.adduct)
            if not np.isfinite(nm):
                continue
            qm, qp = _clean(np.asarray(r.ms2_mzs, np.float32), np.asarray(r.ms2_normalized_intensities, np.float32),
                            INT_FLOOR, MAX_PEAKS, INT_POWER, ENT_WEIGHT)
            if len(qm) < 3:
                continue
            specs.append((nm, qm, qp))
        if not specs:
            return ['CCO'] * TOPN
        target = float(np.median([s[0] for s in specs]))

        cand = _pool_window(pool['mass'], target)
        if len(cand) < TOPN * 2:
            midp = int(np.searchsorted(pool['mass'], target, 'left'))
            extra = []
            d = 1
            while len(cand) + len(extra) < TOPN * 2 and d < len(pool['mass']):
                if midp - d >= 0:
                    extra.append(midp - d)
                if midp + d < len(pool['mass']):
                    extra.append(midp + d)
                d += 1
            if extra:
                cand = np.unique(np.concatenate([cand, np.array(extra, np.int64)]))
        if len(cand) == 0:
            return ['CCO'] * TOPN
        inv = {int(c): i for i, c in enumerate(cand)}

        # direct
        score = np.zeros(len(cand), np.float32)
        best_direct = 0.0
        for nm, qm, qp in specs:
            tol = target * LIB_PPM_WIN * 1e-6
            lo = int(np.searchsorted(lib['snm'], target - tol, 'left'))
            hi = int(np.searchsorted(lib['snm'], target + tol, 'right'))
            if hi <= lo:
                continue
            sc = search_pre(qm, qp, lo, hi, lib['roff'], lib['rmz'], lib['rit'], MZ_TOL)
            best_direct = max(best_direct, float(sc.max()))
            pos = lib['poolpos'][lo:hi]
            for i in range(len(pos)):
                p = int(pos[i])
                if p < 0:
                    continue
                pp = inv.get(p, -1)
                if pp >= 0 and sc[i] > score[pp]:
                    score[pp] = sc[i]

        # analog (mass-shifted), gated
        if best_direct < DIRECT_GATE:
            an_best = {}
            for nm, qm, qp in specs:
                lo = int(np.searchsorted(rep['snm'], target - ANALOG_DA_WIN, 'left'))
                hi = int(np.searchsorted(rep['snm'], target + ANALOG_DA_WIN, 'right'))
                if hi <= lo:
                    continue
                shift = (target - rep['snm'][lo:hi]).astype(np.float32)
                sc = search_shift_pre(qm, qp, lo, hi, rep['roff'], rep['rmz'], rep['rit'], MZ_TOL, shift)
                mask = sc >= ANALOG_MIN_SIM
                if not mask.any():
                    continue
                if hi - lo > N_ANALOG:
                    to = np.argpartition(-sc, N_ANALOG - 1)[:N_ANALOG]
                else:
                    to = np.nonzero(mask)[0]
                for i in to:
                    if sc[i] >= ANALOG_MIN_SIM:
                        p = int(rep['poolpos'][lo + i])
                        if p >= 0 and (p not in an_best or sc[i] > an_best[p]):
                            an_best[p] = float(sc[i])
            if an_best:
                items = sorted(an_best.items(), key=lambda t: -t[1])[:2 * N_ANALOG]
                ak = np.array([k for k, _ in items], np.int64)
                as_ = np.array([s for _, s in items], np.float32)
                ws = np.array([an_best[int(p)] for p in ak], np.float32)
                inside = ak[np.isin(ak, cand)]
                if len(inside):
                    tan = tanimoto_words(pool['fp'][cand], pool['fp'][inside])
                    ws2 = np.array([ws[int(np.where(ak == p)[0][0])] for p in inside], np.float32)
                    score = score + W_ANALOG * (tan * (ws2 ** SIM_POWER)[None, :]).max(axis=1)

        ord_ = np.argsort(-score, kind='stable')
        out, seen = [], set()
        for p in cand[ord_]:
            if len(out) >= TOPN:
                break
            p = int(p)
            if p in seen:
                continue
            seen.add(p)
            s = str(pool['smiles'][p])
            if s and s.lower() != 'nan':
                out.append(s)
        if len(out) < TOPN:
            midp = int(np.searchsorted(pool['mass'], target, 'left'))
            d = 1
            while len(out) < TOPN and d < len(pool['mass']):
                for q in (midp - d, midp + d):
                    if len(out) >= TOPN:
                        break
                    if 0 <= q < len(pool['mass']) and q not in seen:
                        seen.add(q)
                        s = str(pool['smiles'][q])
                        if s and s.lower() != 'nan':
                            out.append(s)
                d += 1
        while len(out) < TOPN:
            out.append('CCO')
        return out[:TOPN]

    mols = list(te.groupby('molecule_id'))
    if n_mols:
        mols = mols[:n_mols]
    log(f"inferencing {len(mols)} molecules")
    t0 = time.time()
    rows = []
    for gi, (mid, g) in enumerate(mols):
        rows.append((str(mid), ';'.join(predict_mid(g))))
        if gi % 50 == 0:
            log(f"  {gi}/{len(mols)} ({time.time()-t0:.0f}s)")
    sub = pd.DataFrame(rows, columns=['molecule_id', 'smiles'])
    allm = set(te.molecule_id.astype(str))
    have = set(sub.molecule_id)
    if allm - have:
        sub = pd.concat([sub, pd.DataFrame(
            [{'molecule_id': m, 'smiles': 'CCO'} for m in sorted(allm - have)])], ignore_index=True)
    sub = sub.drop_duplicates('molecule_id')
    sub.to_csv(out_path, index=False)
    log(f"DONE -> {out_path} ({time.time()-T0:.0f}s)")
    return sub

def run(comp_dir, out_path=None, n_mols=None, max_lib=None, pool_extra=None, logf=print):
    TRAIN = os.path.join(comp_dir, 'train.parquet')
    TEST = os.path.join(comp_dir, 'test.parquet')
    import pyarrow.parquet as pq
    tbl = pq.read_table(TRAIN, columns=['ms2_mzs', 'ms2_normalized_intensities', 'adduct',
                                        'precursor_mz', 'inchikey14', 'base_peak_intensity',
                                        'normalized_smiles', 'molecular_formula'])
    nms = neutral_mass_array(tbl.column('precursor_mz').to_numpy(),
                             np.asarray(tbl.column('adduct').to_pylist(), dtype=object))
    mask = np.isfinite(nms)
    flat = extract_flat(tbl, mask)
    tr = tbl.filter(pa_mask(mask)).drop(['ms2_mzs', 'ms2_normalized_intensities']).to_pandas()
    te = pd.read_parquet(TEST)
    return run_core(tr, te, out_path, n_mols=n_mols, max_lib=max_lib,
                    pool_extra=pool_extra, logf=logf, flat=flat)

def kaggle_main():
    hits = glob.glob('/kaggle/input/**/test.parquet', recursive=True)
    comp = os.path.dirname(sorted(hits, key=len)[0])
    te_ids = sorted(set(pd.read_parquet(os.path.join(comp, 'test.parquet'),
                                         columns=['spectrum_id']).spectrum_id.astype(str)))
    sig = hashlib.md5(','.join(te_ids).encode()).hexdigest()
    is_rerun = bool(os.environ.get('KAGGLE_IS_COMPETITION_RERUN')) or sig != '654e030e4173780e945252b4a5adf70b'
    print(f"TEST_SIG={sig} IS_RERUN={is_rerun}", flush=True)
    import pyarrow.parquet as pq
    tbl = pq.read_table(os.path.join(comp, 'train.parquet'), columns=['ms2_mzs', 'ms2_normalized_intensities',
        'adduct', 'precursor_mz', 'inchikey14', 'base_peak_intensity', 'normalized_smiles',
        'molecular_formula'])
    nms = neutral_mass_array(tbl.column('precursor_mz').to_numpy(),
                             np.asarray(tbl.column('adduct').to_pylist(), dtype=object))
    mask = np.isfinite(nms)
    flat = extract_flat(tbl, mask)
    tr = tbl.filter(pa_mask(mask)).drop(['ms2_mzs', 'ms2_normalized_intensities']).to_pandas()
    te = pd.read_parquet(os.path.join(comp, 'test.parquet'))
    if not is_rerun:
        run_core(tr, te, out_path='/kaggle/working/submission.csv', n_mols=12, max_lib=300_000, flat=flat)
    else:
        run_core(tr, te, out_path='/kaggle/working/submission.csv', flat=flat)

# (module entry point disabled in combined validation script)


# %%

# Use the Dependency Manager
# pip install rdkit==2026.3.3
"""Kaggle-side validation: P1 (hold one spectrum/mol) + P2 (hold all spectra/mol) proxies.
Outputs /kaggle/working/eval_metrics.json. Not a submission."""
import os, glob, time, json
import numpy as np
import pandas as pd

T0 = time.time()
def log(m): print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)

hits = glob.glob('/kaggle/input/**/train.parquet', recursive=True)
COMP = os.path.dirname(sorted(hits, key=len)[0])
from casmi_v2 import run_core

log("reading train")
tr = pd.read_parquet(os.path.join(COMP, "train.parquet"),
                     columns=['ms2_mzs', 'ms2_normalized_intensities', 'adduct', 'precursor_mz',
                              'inchikey14', 'base_peak_intensity', 'normalized_smiles',
                              'molecular_formula', 'molecule_id', 'spectrum_id'])
log(f"train {len(tr):,}")

smi_to_ik = {}
for s, k in zip(tr.normalized_smiles, tr.inchikey14):
    s = str(s)
    if s not in smi_to_ik:
        smi_to_ik[s] = str(k)
log("smi map built")

def mrr25(smis, truth):
    for i, s in enumerate(smis[:25], start=1):
        if smi_to_ik.get(s) == truth:
            return 1.0 / i
    return 0.0

def evaluate(sub, truth_map):
    tot, h1, h5 = 0.0, 0, 0
    for r in sub.itertuples(index=False):
        smis = [s for s in str(r.smiles).split(";") if s]
        r_ = mrr25(smis, truth_map.get(r.molecule_id))
        tot += r_
        if r_ >= 1.0: h1 += 1
        if r_ >= 0.2: h5 += 1
    n = len(sub)
    res = {"n": n, "mrr25": round(tot / n, 4), "top1": round(h1 / n, 4), "top5": round(h5 / n, 4)}
    log(json.dumps(res))
    return res

N_MOLS = 400
rng = np.random.default_rng(0)
mols = tr.molecule_id.unique()
hold_mols = set(rng.choice(mols, N_MOLS, replace=False))
hold = tr.molecule_id.isin(hold_mols)
t_hold = tr[hold].reset_index(drop=True)
t_lib = tr[~hold].reset_index(drop=True)
truth_map = t_hold.groupby("molecule_id").inchikey14.first().to_dict()

# P2: all spectra held out; pool gets the structures back
log("=== P2 (strict) ===")
pool_extra = t_hold[["inchikey14", "normalized_smiles", "molecular_formula"]].drop_duplicates("inchikey14")
sub2 = run_core(t_lib, t_hold, out_path='/kaggle/working/sub_p2.csv', pool_extra=pool_extra, logf=log)
res_p2 = evaluate(sub2, truth_map)

# P1: one spectrum held out, others stay in library
log("=== P1 (self-match) ===")
drop_idx = t_hold.groupby("molecule_id").head(1).index
lib1 = pd.concat([t_lib, t_hold.loc[~t_hold.index.isin(drop_idx)]], ignore_index=True)
test1 = t_hold.loc[t_hold.index.isin(drop_idx)].reset_index(drop=True)
sub1 = run_core(lib1, test1, out_path='/kaggle/working/sub_p1.csv', logf=log)
res_p1 = evaluate(sub1, truth_map)

with open('/kaggle/working/eval_metrics.json', 'w') as f:
    json.dump({"p1": res_p1, "p2": res_p2, "n_mols": N_MOLS, "seed": 0}, f, indent=1)
log("DONE")
