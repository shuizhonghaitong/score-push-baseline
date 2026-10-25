"""Core CASMI 2026 pipeline: spectral library search + analog propagation.

numpy-only hot paths; rdkit is used elsewhere only for candidate fingerprints.
Designed to run both locally and inside a Kaggle notebook.
"""
from __future__ import annotations

import re
import numpy as np

# ----------------------------------------------------------------------------
# constants
# ----------------------------------------------------------------------------
INT_FLOOR = 0.002      # drop peaks below 0.2% of base peak
MAX_PEAKS = 64         # keep N most intense peaks after floor
ANALOG_PEAKS = 32      # peaks used for the (wider) analog search
MZ_TOL = 0.01          # Da tolerance for peak matching in cosine
PPM_WIN = 10.0         # candidate-structure mass window (ppm)
PPM_FALLBACK = (30.0, 100.0, 1000.0)
LIB_PPM_WIN = 15.0     # library-spectrum neutral-mass prefilter (ppm)
ANALOG_DA_WIN = 200.0  # analog propagation mass window (Da)
N_ANALOG = 60          # top similar representative spectra kept for analog channel
SIM_POWER = 3.0        # sim^p weighting for analog propagation
ANALOG_MIN_SIM = 0.30  # minimum cosine for an analog hit to count
DIRECT_GATE = 0.55     # skip analog channel when best direct sim >= this
TOPN = 25

# exact monoisotopic masses (Da)
_H = 1.00782503207
_E = 0.00054857990
PROTON = _H - _E
_H2O = 18.01056466
_NH4 = 14.0030740048 + 4 * _H
_NA = 22.9897692809
_K = 38.96370668
_CL = 34.96885268
_CH2O2 = 12.000000 + 2 * _H + 2 * 15.9949146196

# adduct -> (carrier mass, sign)  neutral mass = prec - sign*carrier (z = 1)
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

_AMU = {
    "C": 12.000000, "H": _H, "N": 14.0030740048, "O": 15.9949146196,
    "F": 18.99840322, "P": 30.97376163, "S": 31.97207100,
    "Cl": 34.96885268, "Br": 78.9183371, "I": 126.904473,
    "B": 11.0093054, "Si": 27.9769265325, "Se": 79.9165213,
    "Na": _NA, "K": _K, "Mg": 23.9850417, "Ca": 39.9625908, "Li": 7.0160034,
}
_FMLA_RE = re.compile(r"([A-Z][a-z]?)(\d*)")

def formula_mass(formula) -> float | None:
    if not formula:
        return None
    try:
        m = 0.0
        for el, num in _FMLA_RE.findall(formula):
            n = int(num) if num else 1
            m += n * _AMU[el]
        return m
    except (KeyError, TypeError):
        return None

def neutral_mass(prec_mz, adduct: str):
    try:
        prec_mz = float(prec_mz)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(prec_mz) or prec_mz <= 0:
        return None
    car = ADDUCTS.get(adduct)
    if car is None:
        return None
    carrier, sign = car
    return prec_mz - sign * carrier

def clean_peaks(mz, it, floor: float = INT_FLOOR, topk: int = MAX_PEAKS,
                prec_mz=None):
    """Intensity floor, drop peaks above precursor+2Da, keep top-k by intensity,
    sort by m/z. Returns aligned (mz, it) float32 arrays."""
    mz = np.asarray(mz, dtype=np.float64)
    it = np.asarray(it, dtype=np.float64)
    n = len(mz)
    if n == 0:
        return (np.empty(0, np.float32), np.empty(0, np.float32))
    mx = it.max()
    if mx <= 0 or not np.isfinite(mx):
        return (np.empty(0, np.float32), np.empty(0, np.float32))
    keep = it >= floor * mx
    if prec_mz is not None:
        keep &= mz <= prec_mz + 2.0
    idx = np.nonzero(keep)[0]
    if idx.size > topk:
        sel = np.argpartition(-it[idx], topk)[:topk]
        idx = idx[sel]
    idx = np.sort(idx)
    return mz[idx].astype(np.float32), it[idx].astype(np.float32)

# ----------------------------------------------------------------------------
# candidate pool
# ----------------------------------------------------------------------------
class Pool:
    """Unique structures (by inchikey14 id), sorted by neutral mass."""

    def __init__(self, key_ids: np.ndarray, smiles, mass: np.ndarray,
                 fp_words: np.ndarray | None = None):
        self.key_ids = key_ids        # (N,) int64
        self.smiles = smiles          # (N,) object
        self.mass = mass              # (N,) float64 sorted asc
        self.fp = fp_words            # (N, W) uint32 or None
        self.keyid_to_pos = {int(k): i for i, k in enumerate(key_ids)}

    def window(self, nm: float, ppm: float) -> np.ndarray:
        tol = nm * ppm * 1e-6
        lo = int(np.searchsorted(self.mass, nm - tol, "left"))
        hi = int(np.searchsorted(self.mass, nm + tol, "right"))
        return np.arange(lo, hi)

def build_pool(df, key_map: dict) -> Pool:
    """df: DataFrame[inchikey14(str), normalized_smiles, molecular_formula].
    key_map: inchikey14 str -> int id."""
    g = df.groupby("inchikey14")
    rep = g["normalized_smiles"].agg(lambda s: s.mode().iat[0] if len(s.mode()) else s.iat[0])
    fmla = g["molecular_formula"].agg(lambda s: s.mode().iat[0] if len(s.mode()) else s.iat[0])
    keys = list(rep.index)
    smiles = np.array([rep[k] for k in keys], dtype=object)
    mass = np.array([formula_mass(fmla[k]) for k in keys], dtype=np.float64)
    order = np.argsort(mass, kind="stable")
    mass = mass[order]
    keys = np.array(keys, dtype=object)[order]
    smiles = smiles[order]
    key_ids = np.array([int(key_map[str(k)]) for k in keys], dtype=np.int64)
    return Pool(key_ids, smiles, mass)

# ----------------------------------------------------------------------------
# spectral libraries
# ----------------------------------------------------------------------------
class Lib:
    """Curated spectra sorted by neutral mass. Arrays are (K, N) column-major
    per spectrum for fast slicing."""

    def __init__(self, nm, key14, mz, it, norm):
        self.nm = nm                  # (N,) float64 sorted
        self.key14 = key14            # (N,) int64
        self.mz = mz                  # (K, N) float32, padded with 0
        self.it = it                  # (K, N) float32, padded with 0
        self.norm = norm              # (N,) float32 L2 norm
        self.K = mz.shape[0]
        self.N = len(nm)

    def window(self, nm: float, ppm: float):
        tol = nm * ppm * 1e-6
        lo = int(np.searchsorted(self.nm, nm - tol, "left"))
        hi = int(np.searchsorted(self.nm, nm + tol, "right"))
        return lo, hi

    def window_da(self, nm: float, da: float):
        lo = int(np.searchsorted(self.nm, nm - da, "left"))
        hi = int(np.searchsorted(self.nm, nm + da, "right"))
        return lo, hi

def build_lib(nm, key14, spec_mz, spec_it, K: int = MAX_PEAKS) -> Lib:
    n = len(nm)
    order = np.argsort(nm, kind="stable")
    nm = np.asarray(nm)[order]
    key14 = np.asarray(key14, dtype=np.int64)[order]
    mz = np.zeros((K, n), np.float32)
    it = np.zeros((K, n), np.float32)
    for j in range(n):
        m = spec_mz[order[j]]
        v = spec_it[order[j]]
        c = min(len(m), K)
        if c:
            mz[:c, j] = m[:c]
            it[:c, j] = v[:c]
    norm = np.sqrt((it ** 2).sum(axis=0)).astype(np.float32)
    return Lib(nm, key14, mz, it, norm)

def cosine_range(lib: Lib, lo: int, hi: int, q_mz: np.ndarray, q_it: np.ndarray):
    """Matched-peak cosine of one query vs lib[lo:hi]. (sims, abs_idx)"""
    n_c = hi - lo
    if n_c == 0 or len(q_mz) == 0:
        return np.empty(0, np.float32), np.empty(0, np.int64)
    qmz = q_mz[:, None, None].astype(np.float32)
    qit = q_it[None, None, :].astype(np.float32)
    cmz = lib.mz[:, lo:hi]
    cit = lib.it[:, lo:hi]
    matched = np.where(np.abs(cmz[:, :, None] - qmz) <= MZ_TOL,
                       cit[:, :, None] * qit, 0.0).sum(axis=2)
    nq = float(np.dot(q_it, q_it))
    sims = np.zeros(n_c, np.float32)
    ok = lib.norm[lo:hi] > 0
    sims[ok] = matched[ok] / (np.sqrt(nq) * lib.norm[lo:hi][ok])
    return sims, np.arange(lo, hi, dtype=np.int64)

def cosine_chunks(lib: Lib, lo: int, hi: int, q_mz, q_it, chunk=8192):
    """Same as cosine_range but memory-bounded for wide windows."""
    sims = np.zeros(hi - lo, np.float32)
    for a in range(lo, hi, chunk):
        b = min(a + chunk, hi)
        s, _ = cosine_range(lib, a, b, q_mz, q_it)
        sims[a - lo:b - lo] = s
    return sims, np.arange(lo, hi, dtype=np.int64)

def tanimoto_words(a, b):
    """a: (N,W) uint32, b: (M,W) uint32 -> (N,M) float32 Tanimoto."""
    inter = np.bitwise_and(a[:, :, None].astype(np.uint64),
                           b[None, :, :].astype(np.uint64)).sum(axis=1).astype(np.float64)
    ua = a.sum(axis=1).astype(np.float64)[:, None]
    ub = b.sum(axis=1).astype(np.float64)[None, :]
    union = ua + ub - inter
    return np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0).astype(np.float32)
