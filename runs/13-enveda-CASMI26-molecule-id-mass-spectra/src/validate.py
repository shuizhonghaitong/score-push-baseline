"""Local validation:
  1) visible test.parquet (samples from train) — sanity, expected ~1.0
  2) held-out molecules from train — proxy for class-1 hidden test
"""
from __future__ import annotations

import os, sys, time, json, argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from casmi_core import (Pool, build_pool, build_lib, clean_peaks,
                        neutral_mass, MAX_PEAKS, ANALOG_PEAKS)
from infer import predict_molecule, mrr25

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data",
                    "enveda-casmi26-molecule-id-mass-spectra")
ART = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "artifacts")

def find_file(name):
    hits = [os.path.join(DATA, name)]
    if not os.path.exists(hits[0]):
        # kaggle CLI may nest under competition dir
        import glob
        hits = glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "**", name), recursive=True)
    if not hits:
        raise FileNotFoundError(name)
    return hits[0]

def load_train(keyed=False):
    cols = ["molecule_id", "spectrum_id", "ms2_mzs", "ms2_normalized_intensities",
            "base_peak_intensity", "adduct", "ionization_mode", "precursor_mz",
            "normalized_smiles", "inchikey", "inchikey14", "molecular_formula"]
    t = pd.read_parquet(find_file("train.parquet"), columns=cols)
    return t

def build_key_map(ik14_series):
    uniq = np.sort(ik14_series.dropna().unique().astype(str).values)
    return {str(k): i for i, k in enumerate(uniq)}, uniq

def make_lib(t, key_map, max_spectra=None, rep=True):
    nm = t.precursor_mz.to_numpy(np.float64)
    add = t.adduct.to_numpy()
    from casmi_core import neutral_mass as _nm
    nms = np.array([_nm(p, a) for p, a in zip(nm, add)], dtype=np.float64)
    keep = np.isfinite(nms)
    keys = np.array([key_map.get(str(k), -1) for k in t.inchikey14.to_numpy()], dtype=np.int64)
    keep &= keys >= 0
    t2 = t[keep]
    nms = nms[keep]
    keys = keys[keep]
    if max_spectra and len(t2) > max_spectra:
        sel = np.random.default_rng(0).choice(len(t2), max_spectra, replace=False)
        t2, nms, keys = t2.iloc[sel], nms[sel], keys[sel]
    spec_mz, spec_it = [], []
    prec = t2.precursor_mz.to_numpy(np.float64)
    for mz, it in zip(t2.ms2_mzs, t2.ms2_normalized_intensities):
        m, v = clean_peaks(np.asarray(mz), np.asarray(it), prec_mz=prec[len(spec_mz)])
        spec_mz.append(m)
        spec_it.append(v)
    lib = build_lib(nms, keys, spec_mz, spec_it, K=MAX_PEAKS)
    librep = None
    if rep:
        # representative: max base_peak_intensity per structure
        bpi = t2.base_peak_intensity.to_numpy(np.float64)
        bpi = np.where(np.isfinite(bpi), bpi, 0.0)
        order = np.argsort(-bpi)
        seen = set(); rep_idx = []
        for i in order:
            k = int(keys[i])
            if k not in seen:
                seen.add(k); rep_idx.append(i)
        rep_idx = np.array(rep_idx)
        rep_mz = [spec_mz[i][:ANALOG_PEAKS] for i in rep_idx]
        rep_it = [spec_it[i][:ANALOG_PEAKS] for i in rep_idx]
        librep = build_lib(nms[rep_idx], keys[rep_idx], rep_mz, rep_it, K=ANALOG_PEAKS)
    return lib, librep, (t2, nms, keys)

def add_fps(pool, smiles):
    from rdkit import Chem
    from rdkit.Chem import rdFingerprintGenerator
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.*")
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    W = 32
    fp = np.zeros((len(pool), W), dtype=np.uint32)
    ok = 0
    for i, s in enumerate(smiles):
        m = Chem.MolFromSmiles(str(s))
        if m is None:
            continue
        bits = gen.GetFingerprint(m).GetExplicitBitVector()
        for b in bits.GetOnBits():
            fp[i, b // 32] |= np.uint32(1) << (b % 32)
        ok += 1
    pool.fp = fp
    print(f"[fps] {ok}/{len(pool)} parsed")

def evaluate(name, pool, lib, librep, queries, verbose=True):
    """queries: list of (molecule_id, truth_key, spectra[list of dict])"""
    total = 0.0
    hit1 = hit5 = hit25 = 0
    t0 = time.time()
    for qi, (mid, truth, specs) in enumerate(queries):
        smi, keys, diag = predict_molecule(lib, librep, pool, specs)
        r = mrr25(keys, truth)
        total += r
        if r >= 1.0: hit1 += 1
        if r >= 0.2: hit5 += 1
        if r > 0: hit25 += 1
        if verbose and (qi % 100 == 0 or qi == len(queries) - 1):
            print(f"[{name}] {qi+1}/{len(queries)} mrr={total/(qi+1):.4f} "
                  f"@1={hit1/(qi+1):.3f} @5={hit5/(qi+1):.3f} @25={hit25/(qi+1):.3f} "
                  f"({time.time()-t0:.0f}s)", flush=True)
    n = len(queries)
    res = {"name": name, "n": n, "mrr25": total / n, "top1": hit1 / n,
           "top5": hit5 / n, "top25": hit25 / n}
    print(json.dumps(res, indent=1))
    return res

def to_query_specs(df):
    """group spectra by molecule_id -> list of (mid, truth, [specs]) using inchikey14."""
    out = []
    for mid, g in df.groupby("molecule_id"):
        truth = int(next(iter(g.inchikey14.map(lambda k: g_key(k)))))
        specs = []
        for r in g.itertuples():
            specs.append({"mz": np.asarray(r.ms2_mzs, np.float64),
                          "it": np.asarray(r.ms2_normalized_intensities, np.float64),
                          "prec": float(r.precursor_mz), "adduct": r.adduct})
        out.append((mid, truth, specs))
    return out

_gkey = {}
def g_key(k):
    return _gkey.get(str(k), -1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["visible", "holdout"], default="holdout")
    ap.add_argument("--n-mols", type=int, default=800)
    ap.add_argument("--max-spectra", type=int, default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--skip-analog", action="store_true")
    ap.add_argument("--w-analog", type=float, default=0.5)
    args = ap.parse_args()

    os.makedirs(ART, exist_ok=True)
    t = load_train()
    key_map, uniq = build_key_map(t.inchikey14)
    _gkey.update(key_map)
    print(f"[data] {len(t):,} spectra, {len(uniq):,} structures")

    pd_ = t[["inchikey14", "normalized_smiles", "molecular_formula"]]
    if args.mode == "visible":
        te = pd.read_parquet(find_file("test.parquet"))
        lab = t.set_index("spectrum_id").inchikey14
        te = te[te.spectrum_id.isin(lab.index)].copy()
        te["inchikey14"] = te.spectrum_id.map(lab)
        lib, librep, _ = make_lib(t, key_map, max_spectra=args.max_spectra)
        pool = build_pool(pd_, key_map)
        add_fps(pool, pool.smiles)
        queries = to_query_specs(te)
    else:
        rng = np.random.default_rng(args.seed)
        mols = np.sort(t.molecule_id.unique())
        hold = set(rng.choice(mols, args.n_mols, replace=False))
        t_hold = t[t.molecule_id.isin(hold)]
        t_lib = t[~t.molecule_id.isin(hold)]
        lib, librep, _ = make_lib(t_lib, key_map, max_spectra=args.max_spectra)
        # pool built from full train (structures known) but library excludes hold spectra
        pool = build_pool(pd_, key_map)
        add_fps(pool, pool.smiles)
        queries = to_query_specs(t_hold)

    from infer import predict_molecule
    global predict_molecule
    res = evaluate(args.mode, pool, lib, librep, queries)
    with open(os.path.join(ART, f"eval_{args.mode}_{args.seed}.json"), "a") as f:
        f.write(json.dumps(res) + "\n")

if __name__ == "__main__":
    main()
