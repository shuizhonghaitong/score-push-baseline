"""Ranking: up to TOPN candidate structures per molecule."""
from __future__ import annotations

import numpy as np

from casmi_core import (
    Lib, Pool, clean_peaks, neutral_mass,
    LIB_PPM_WIN, ANALOG_DA_WIN, ANALOG_PEAKS, N_ANALOG, SIM_POWER,
    TOPN, PPM_WIN, PPM_FALLBACK, ANALOG_MIN_SIM, DIRECT_GATE,
)

def molecule_candidates(pool: Pool, nm: float):
    """Candidate structure positions in widening mass windows; nearest-mass fill."""
    idx = pool.window(nm, PPM_WIN)
    for ppm in PPM_FALLBACK:
        if len(idx) >= TOPN:
            break
        idx = pool.window(nm, max(PPM_WIN, ppm))
    if len(idx) < TOPN * 2:
        lo = int(np.searchsorted(pool.mass, nm, "left"))
        extra = []
        d = 1
        while len(idx) + len(extra) < TOPN * 2 and d < len(pool.mass):
            if lo - d >= 0:
                extra.append(lo - d)
            if lo + d < len(pool.mass):
                extra.append(lo + d)
            d += 1
        if extra:
            idx = np.unique(np.concatenate([idx, np.array(extra, dtype=np.int64)]))
    return idx

def predict_molecule(lib: Lib, librep: Lib, pool: Pool, spectra,
                     w_direct: float = 1.0, w_analog: float = 0.5):
    """spectra: iterable of dicts with mz, it, prec, adduct.
    Returns (ranked_smiles, ranked_key_ids, diag)."""
    nms, cleaned, ancleaned = [], [], []
    for sp in spectra:
        nm = neutral_mass(sp["prec"], sp["adduct"])
        if nm is None:
            continue
        cmz, cit = clean_peaks(sp["mz"], sp["it"], prec_mz=sp["prec"])
        if len(cmz) < 3:
            continue
        amz, ait = clean_peaks(sp["mz"], sp["it"], topk=ANALOG_PEAKS, prec_mz=sp["prec"])
        nms.append(nm)
        cleaned.append((cmz, cit))
        ancleaned.append((amz, ait))
    if not nms:
        return [], [], {}
    target = float(np.median(nms))
    cand = molecule_candidates(pool, target)
    if len(cand) == 0:
        return [], [], {}

    score = np.zeros(len(cand), np.float32)
    inv = {int(c): i for i, c in enumerate(cand)}
    best_direct = 0.0
    for cmz, cit in cleaned:
        lo, hi = lib.window(target, LIB_PPM_WIN)
        sims, idx = _cos(lib, lo, hi, cmz, cit)
        if not len(sims):
            continue
        best_direct = max(best_direct, float(sims.max()))
        keys = lib.key14[idx]
        pos = np.fromiter((inv.get(int(k), -1) for k in keys), dtype=np.int64, count=len(keys))
        ok = pos >= 0
        if ok.any():
            np.maximum.at(score, pos[ok], sims[ok])
    # direct term
    score = score  # already max-aggregated per candidate

    diag = {"best_direct": best_direct, "n_cand": int(len(cand))}
    # ---- analog channel (gated)
    if best_direct < DIRECT_GATE and pool.fp is not None:
        an_keys, an_sims = [], []
        for amz, ait in ancleaned:
            lo, hi = librep.window_da(target, ANALOG_DA_WIN)
            sims, idx = _cos_chunks(librep, lo, hi, amz, ait)
            topn = min(N_ANALOG, int((sims >= ANALOG_MIN_SIM).sum()))
            if topn == 0:
                continue
            to = np.argpartition(-sims, min(topn, len(sims)) - 1)[:min(topn, len(sims))] if len(sims) > topn else np.arange(len(sims))
            to = to[np.argsort(-sims[to])]
            for i in to[:N_ANALOG]:
                if sims[i] >= ANALOG_MIN_SIM:
                    an_keys.append(int(librep.key14[idx[i]]))
                    an_sims.append(float(sims[i]))
        if an_keys:
            # dedupe: best sim per key
            best = {}
            for k, s in zip(an_keys, an_sims):
                if k not in best or s > best[k]:
                    best[k] = s
            items = sorted(best.items(), key=lambda t: -t[1])[: 2 * N_ANALOG]
            an_keys = np.array([k for k, _ in items], dtype=np.int64)
            an_sims = np.array([s for _, s in items], dtype=np.float32)
            ap = pool.keyid_to_pos
            pos = np.array([ap.get(int(k), -1) for k in an_keys], dtype=np.int64)
            inside = pos >= 0
            if inside.any():
                tan = _tanimoto(pool.fp[cand], pool.fp[pos[inside]])
                score = score + w_analog * np.maximum(tan * (an_sims[inside] ** SIM_POWER)[None, :], 0).max(axis=1)
            diag["n_analog"] = int(len(an_keys))

    order = np.argsort(-score, kind="stable")
    ranked = [int(p) for p in cand[order]]
    out_smi, out_keys = [], []
    seen = set()
    for p in ranked:
        if p in seen:
            continue
        seen.add(p)
        smi = pool.smiles[p]
        if not smi:
            continue
        out_smi.append(str(smi))
        out_keys.append(int(pool.key_ids[p]))
        if len(out_smi) >= TOPN:
            break
    return out_smi, out_keys, diag

def _cos(lib, lo, hi, q_mz, q_it):
    from casmi_core import cosine_range
    return cosine_range(lib, lo, hi, q_mz, q_it)

def _cos_chunks(lib, lo, hi, q_mz, q_it):
    from casmi_core import cosine_chunks
    return cosine_chunks(lib, lo, hi, q_mz, q_it)

def _tanimoto(a, b):
    from casmi_core import tanimoto_words
    return tanimoto_words(a, b)

def mrr25(ranked_keys, truth_key) -> float:
    for i, k in enumerate(ranked_keys[:TOPN], start=1):
        if int(k) == int(truth_key):
            return 1.0 / i
    return 0.0
