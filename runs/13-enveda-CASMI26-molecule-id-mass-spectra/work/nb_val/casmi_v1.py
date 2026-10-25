# Use the Dependency Manager
# pip install rdkit==2026.3.3
"""CASMI 2026 V1: mass-window candidate pool + direct library cosine search
+ fingerprint analog channel. CPU-only, self-contained.

Local entry:  python casmi_v1.py <comp_dir> [n_mols]
Kaggle entry: writes /kaggle/working/submission.csv (auto-detected).
"""
import os, sys, time, glob, hashlib, re
from multiprocessing import Pool as MPool

import numpy as np
import pandas as pd

# ============================================================================
# constants
# ============================================================================
INT_FLOOR = 0.002
MAX_PEAKS = 64
ANALOG_PEAKS = 32
MZ_TOL = 0.01
PPM_WIN = 10.0
PPM_FALLBACK = (30.0, 100.0, 1000.0)
LIB_PPM_WIN = 15.0
ANALOG_DA_WIN = 200.0
N_ANALOG = 60
SIM_POWER = 3.0
ANALOG_MIN_SIM = 0.30
DIRECT_GATE = 0.55
W_DIRECT = 1.0
W_ANALOG = 0.5
TOPN = 25
COS_CHUNK = 4096

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

def clean_q(mz, it, prec, topk):
    mz = np.asarray(mz, np.float64)
    it = np.asarray(it, np.float64)
    n = len(mz)
    if n == 0:
        return (np.empty(0, np.float32), np.empty(0, np.float32))
    mx = it.max()
    if mx <= 0 or not np.isfinite(mx):
        return (np.empty(0, np.float32), np.empty(0, np.float32))
    idx = np.flatnonzero((it >= INT_FLOOR * mx) & (mz <= float(prec) + 2.0))
    if idx.size > topk:
        idx = idx[np.argpartition(-it[idx], topk)[:topk]]
    idx = idx[np.argsort(mz[idx], kind='stable')]
    return mz[idx].astype(np.float32), it[idx].astype(np.float32)

# ============================================================================
# similarity
# ============================================================================
def cosine_range(lib, lo, hi, q_mz, q_it):
    n_c = hi - lo
    if n_c == 0 or len(q_mz) == 0:
        return np.empty(0, np.float32)
    qmz = q_mz[:, None, None].astype(np.float32)
    qit = q_it[None, None, :].astype(np.float32)
    cmz = lib['mz'][:, lo:hi]
    cit = lib['it'][:, lo:hi]
    matched = np.where(np.abs(cmz[:, :, None] - qmz) <= MZ_TOL,
                       cit[:, :, None] * qit, 0.0).sum(axis=2)
    nq = float(np.dot(q_it, q_it))
    sims = np.zeros(n_c, np.float32)
    okc = lib['norm'][lo:hi] > 0
    sims[okc] = matched[okc] / (np.sqrt(nq) * lib['norm'][lo:hi][okc])
    return sims

def cosine_chunks(lib, lo, hi, q_mz, q_it):
    sims = np.zeros(hi - lo, np.float32)
    for a in range(lo, hi, COS_CHUNK):
        b = min(a + COS_CHUNK, hi)
        sims[a - lo:b - lo] = cosine_range(lib, a, b, q_mz, q_it)
    return sims

def tanimoto_words(a, b):
    inter = np.bitwise_and(a[:, :, None].astype(np.uint64),
                           b[None, :, :].astype(np.uint64)).sum(axis=1).astype(np.float64)
    ua = a.sum(axis=1).astype(np.float64)[:, None]
    ub = b.sum(axis=1).astype(np.float64)[None, :]
    union = ua + ub - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0).astype(np.float32)

# ============================================================================
# main pipeline
# ============================================================================
def run_core(tr, te, out_path, n_mols=None, max_lib=None, pool_extra=None, logf=print):
    T0 = time.time()
    def log(msg):
        logf(f"[{time.time()-T0:7.1f}s] {msg}")

    if out_path is None:
        out_path = os.path.join('/kaggle/working' if os.path.isdir('/kaggle/working')
                                else os.path.dirname(os.path.abspath(__file__)), 'submission.csv')
    log(f"cpus={os.cpu_count()}")
    log(f"train: {len(tr):,} spectra")

    nms = np.array([neutral_mass(p, a) for p, a in zip(tr.precursor_mz.to_numpy(), tr.adduct.to_numpy())], np.float64)
    valid = np.isfinite(nms)
    tr = tr[valid].reset_index(drop=True)
    nms = nms[valid]
    if max_lib and len(tr) > max_lib:
        tr = tr.iloc[:max_lib].reset_index(drop=True)
        nms = nms[:max_lib]
        log(f"limiting library to {len(tr):,}")
    log(f"library: {len(tr):,} spectra with finite neutral mass")

    # ---- curate peaks (parallel numpy)
    prec = tr.precursor_mz.to_numpy(np.float64)
    mzs = tr.ms2_mzs.to_numpy()
    its = tr.ms2_normalized_intensities.to_numpy()
    def _curate_one(i):
        return clean_q(mzs[i], its[i], prec[i], MAX_PEAKS)
    ncpu = min(os.cpu_count() or 4, 16)
    log("curating library peaks")
    t0 = time.time()
    if ncpu > 1:
        with MPool(ncpu) as mp:
            cur = mp.map(_curate_one, range(len(tr)), chunksize=256)
    else:
        cur = [_curate_one(i) for i in range(len(tr))]
    log(f"curation done ({time.time()-t0:.0f}s)")

    # ---- key ids
    key_str = tr.inchikey14.to_numpy()
    keys = {}
    key_ids = np.empty(len(tr), np.int64)
    for i, k in enumerate(key_str):
        k2 = str(k)
        if k2 not in keys:
            keys[k2] = len(keys)
        key_ids[i] = keys[k2]
    log(f"{len(keys):,} unique structures in library")

    K = MAX_PEAKS
    lib_mz = np.zeros((K, len(tr)), np.float32)
    lib_it = np.zeros((K, len(tr)), np.float32)
    for j, (m, v) in enumerate(cur):
        c = min(len(m), K)
        if c:
            lib_mz[:c, j] = m[:c]
            lib_it[:c, j] = v[:c]
    order = np.argsort(nms, kind='stable')
    lib = {
        'nm': nms[order],
        'mz': lib_mz[:, order],
        'it': lib_it[:, order],
    }
    lib['norm'] = np.sqrt((lib['it'] ** 2).sum(axis=0)).astype(np.float32)
    # map library spectra (sorted) -> pool positions later; keep key strings
    lib['keystr'] = key_str[order]
    log(f"library built ({lib['nm'].size:,}, {lib['mz'].nbytes/1e9:.2f} GB)")
    del lib_mz, lib_it, cur

    # ---- representative per structure for analog channel
    bpi = tr.base_peak_intensity.to_numpy(np.float64)
    bpi = np.where(np.isfinite(bpi), bpi, 0.0)
    best = {}
    for i in range(len(tr)):
        k = str(key_str[i])
        if k not in best or bpi[i] > bpi[best[k]]:
            best[k] = i
    rep_idx = np.array(sorted(best.values()))
    # indices refer to tr (unsorted); re-map into sorted lib via order
    pos_in_tr = {i: j for j, i in enumerate(order)}
    rep = np.array([pos_in_tr[i] for i in rep_idx])
    rep_keystr = lib['keystr'][rep]
    rep_nm = lib['nm'][rep]
    ok = np.isfinite(rep_nm)
    rep, rep_nm, rep_keystr = rep[ok], rep_nm[ok], rep_keystr[ok]
    o = np.argsort(rep_nm, kind='stable')
    rep, rep_nm, rep_keystr = rep[o], rep_nm[o], rep_keystr[o]
    RA = ANALOG_PEAKS
    rep_mz = lib['mz'][:RA, rep]
    rep_it = lib['it'][:RA, rep]
    rep_lib = {'nm': rep_nm, 'mz': rep_mz, 'it': rep_it}
    rep_lib['norm'] = np.sqrt((rep_it ** 2).sum(axis=0)).astype(np.float32)
    log(f"rep library {len(rep):,}")

    # ---- candidate pool
    log("building candidate pool")
    psrc = tr[['inchikey14', 'normalized_smiles', 'molecular_formula']].drop_duplicates('inchikey14').reset_index(drop=True)
    if pool_extra is not None:
        psrc = pd.concat([psrc, pool_extra], ignore_index=True).drop_duplicates('inchikey14')
    pkeys = psrc.inchikey14.to_numpy()
    psmi = psrc.normalized_smiles.to_numpy()
    pmass = np.array([formula_mass(f) for f in psrc.molecular_formula.to_numpy()], np.float64)
    okp = np.isfinite(pmass)
    pkeys, psmi, pmass = pkeys[okp], psmi[okp], pmass[okp]
    log(f"pool: {len(pkeys):,} structures with formula mass")

    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdFingerprintGenerator
    RDLogger.DisableLog('rdApp.*')
    _GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    FW = 32
    def _fp_one(smi):
        m = Chem.MolFromSmiles(str(smi))
        if m is None:
            return None
        try:
            bv = _GEN.GetFingerprint(m).GetExplicitBitVector()
            w = np.zeros(FW, np.uint32)
            for b in bv.GetOnBits():
                w[b // 32] |= np.uint32(1) << (b % 32)
            return w
        except Exception:
            return None
    t0 = time.time()
    if ncpu > 1:
        with MPool(ncpu) as mp:
            fpr = mp.map(_fp_one, psmi, chunksize=256)
    else:
        fpr = [_fp_one(s) for s in psmi]
    log(f"fingerprints done ({time.time()-t0:.0f}s)")
    pool_fp = np.zeros((len(psmi), FW), np.uint32)
    for i, x in enumerate(fpr):
        if x is not None:
            pool_fp[i] = x
    del fpr

    po = np.argsort(pmass, kind='stable')
    pool = {
        'mass': pmass[po],
        'smiles': psmi[po],
        'keystr': pkeys[po],
        'fp': pool_fp[po],
    }
    pool_k2i = {str(k): i for i, k in enumerate(pool['keystr'])}
    lib['poolpos'] = np.array([pool_k2i.get(str(k), -1) for k in lib['keystr']], np.int64)
    rep_lib['poolpos'] = np.array([pool_k2i.get(str(k), -1) for k in rep_keystr], np.int64)
    log("pool ready")

    # ---- inference
    def predict_mid(g):
        specs = []
        for r in g.itertuples(index=False):
            nm = neutral_mass(r.precursor_mz, r.adduct)
            if not np.isfinite(nm):
                continue
            m64, i64 = clean_q(r.ms2_mzs, r.ms2_normalized_intensities, r.precursor_mz, MAX_PEAKS)
            if len(m64) < 3:
                continue
            m32, i32 = clean_q(r.ms2_mzs, r.ms2_normalized_intensities, r.precursor_mz, ANALOG_PEAKS)
            specs.append((nm, m64, i64, m32, i32))
        if not specs:
            return ['CCO'] * TOPN
        target = float(np.median(s[0] for s in specs))

        cand = np.arange(*_pool_window(pool['mass'], target))
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

        score = np.zeros(len(cand), np.float32)
        inv = {int(c): i for i, c in enumerate(cand)}
        best_direct = 0.0
        for nm, m64, i64, m32, i32 in specs:
            lo = int(np.searchsorted(lib['nm'], target * (1 - LIB_PPM_WIN * 1e-6), 'left'))
            hi = int(np.searchsorted(lib['nm'], target * (1 + LIB_PPM_WIN * 1e-6), 'right'))
            sims = cosine_chunks(lib, lo, hi, m64, i64)
            if len(sims) == 0:
                continue
            best_direct = max(best_direct, float(sims.max()))
            pos = lib['poolpos'][lo:hi]
            for i, p in enumerate(pos):
                p = int(p)
                if p < 0:
                    continue
                pp = inv.get(p, -1)
                if pp >= 0 and sims[i] > score[pp]:
                    score[pp] = sims[i]

        if best_direct < DIRECT_GATE:
            an_best = {}
            for nm, m64, i64, m32, i32 in specs:
                lo = int(np.searchsorted(rep_lib['nm'], target - ANALOG_DA_WIN, 'left'))
                hi = int(np.searchsorted(rep_lib['nm'], target + ANALOG_DA_WIN, 'right'))
                sims = cosine_chunks(rep_lib, lo, hi, m32, i32)
                if len(sims) == 0:
                    continue
                mask = sims >= ANALOG_MIN_SIM
                if not mask.any():
                    continue
                if len(sims) > N_ANALOG:
                    to = np.argpartition(-sims, N_ANALOG - 1)[:N_ANALOG]
                else:
                    to = np.nonzero(mask)[0]
                for i in to:
                    if sims[i] >= ANALOG_MIN_SIM:
                        p = int(rep_lib['poolpos'][lo + i])
                        if p >= 0:
                            if p not in an_best or sims[i] > an_best[p]:
                                an_best[p] = float(sims[i])
            if an_best:
                items = sorted(an_best.items(), key=lambda t: -t[1])[:2 * N_ANALOG]
                ak = np.array([k for k, _ in items], np.int64)
                as_ = np.array([s for _, s in items], np.float32)
                inside = ak[np.isin(ak, cand)]
                if len(inside):
                    inv2 = {int(c): i for i, c in enumerate(cand)}
                    tan = tanimoto_words(pool['fp'][cand], pool['fp'][inside])
                    ws = np.array([as_[int(np.where(ak == p)[0][0])] for p in inside], np.float32)
                    score = score + W_ANALOG * (tan * (ws ** SIM_POWER)[None, :]).max(axis=1)

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
    # ensure completeness
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
    tr = tbl.to_pandas()
    te = pd.read_parquet(TEST)
    return run_core(tr, te, out_path, n_mols=n_mols, max_lib=max_lib,
                    pool_extra=pool_extra, logf=logf)

def _pool_window(pool_mass, target, ppm=PPM_WIN):
    for p in (ppm,) + tuple(PPM_FALLBACK):
        tol = target * p * 1e-6
        lo = int(np.searchsorted(pool_mass, target - tol, 'left'))
        hi = int(np.searchsorted(pool_mass, target + tol, 'right'))
        if hi - lo >= TOPN:
            return lo, hi
    tol = target * 1000.0 * 1e-6
    return (int(np.searchsorted(pool_mass, target - tol, 'left')),
            int(np.searchsorted(pool_mass, target + tol, 'right')))

# ============================================================================
# kaggle entry
# ============================================================================
def kaggle_main():
    hits = glob.glob('/kaggle/input/**/test.parquet', recursive=True)
    comp = os.path.dirname(sorted(hits, key=len)[0])
    te_ids = sorted(set(pd.read_parquet(os.path.join(comp, 'test.parquet'),
                                         columns=['spectrum_id']).spectrum_id.astype(str)))
    sig = hashlib.md5(','.join(te_ids).encode()).hexdigest()
    is_rerun = bool(os.environ.get('KAGGLE_IS_COMPETITION_RERUN')) or sig != '654e030e4173780e945252b4a5adf70b'
    logf = print
    print(f"TEST_SIG={sig} IS_RERUN={is_rerun}", flush=True)
    import pyarrow.parquet as pq
    tbl = pq.read_table(os.path.join(comp, 'train.parquet'), columns=['ms2_mzs', 'ms2_normalized_intensities',
        'adduct', 'precursor_mz', 'inchikey14', 'base_peak_intensity', 'normalized_smiles',
        'molecular_formula'])
    tr = tbl.to_pandas()
    te = pd.read_parquet(os.path.join(comp, 'test.parquet'))
    if not is_rerun:
        run_core(tr, te, out_path='/kaggle/working/submission.csv', n_mols=12, max_lib=300_000, logf=logf)
    else:
        run_core(tr, te, out_path='/kaggle/working/submission.csv', logf=logf)

if __name__ == '__main__':
    if len(sys.argv) > 1:
        run(sys.argv[1], n_mols=int(sys.argv[2]) if len(sys.argv) > 2 else None)
    else:
        kaggle_main()
