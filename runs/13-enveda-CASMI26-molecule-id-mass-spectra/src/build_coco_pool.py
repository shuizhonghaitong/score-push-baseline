"""Build a COCONUT 2.0 candidate pool artifact (public data, attachable to notebooks).

Output: artifacts/coco_pool.parquet with columns:
  ik14 (str 14), smiles (str), mass (float64)
and artifacts/coco_fp.npy: (N, 32) uint32 Morgan2/2048 words, same row order.
"""
import os, sys, glob, time
import numpy as np
import pandas as pd

RUN = "/mnt/disk3/liuyutong_home/workspace/skill/score-push/runs/13-enveda-CASMI26-molecule-id-mass-spectra"
ZIP = os.path.join(RUN, "work", "coconut", "coconut_complete-10-2024.csv.zip")
OUT = os.path.join(RUN, "artifacts")

def main():
    import zipfile
    t0 = time.time()
    z = zipfile.ZipFile(ZIP)
    name = z.namelist()[0]
    df = pd.read_csv(z.open(name))
    print(f"coconut: {df.shape}, cols={list(df.columns)} ({time.time()-t0:.0f}s)", flush=True)
    sys.stdout.flush()

    # expected-ish columns
    ik_col = next((c for c in df.columns if "inchikey" in c.lower() and "14" not in c.lower()), None)
    smi_col = next((c for c in df.columns if c.lower() in ("smiles", "canonical smiles") or "smiles" in c.lower()), None)
    fmla_col = next((c for c in df.columns if "formula" in c.lower()), None)
    print("ik_col", ik_col, "smi_col", smi_col, "fmla_col", fmla_col, flush=True)
    assert ik_col and smi_col and fmla_col

    d = df[[ik_col, smi_col, fmla_col]].dropna()
    d["ik14"] = d[ik_col].astype(str).str.slice(0, 14)
    d = d.drop_duplicates("ik14")
    print(f"unique ik14: {len(d):,}", flush=True)

    sys.path.insert(0, os.path.join(RUN, "work", "nb_v1"))
    from casmi_v1 import formula_mass
    d["mass"] = d[fmla_col].map(formula_mass)
    d = d[np.isfinite(d.mass)]
    d = d.sort_values("mass")
    print(f"with finite mass: {len(d):,}", flush=True)

    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdFingerprintGenerator
    RDLogger.DisableLog('rdApp.*')
    from multiprocessing import Pool
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
    FW = 32
    def fp_one(s):
        m = Chem.MolFromSmiles(str(s))
        if m is None:
            return None
        bv = gen.GetFingerprint(m).GetExplicitBitVector()
        w = np.zeros(FW, np.uint32)
        for b in bv.GetOnBits():
            w[b // 32] |= np.uint32(1) << (b % 32)
        return w
    with Pool(min(os.cpu_count(), 8)) as mp:
        fpr = mp.map(fp_one, d.smiles.tolist(), chunksize=512)
    fp = np.zeros((len(d), FW), np.uint32)
    for i, w in enumerate(fpr):
        if w is not None:
            fp[i] = w
    print(f"fps done: {sum(1 for w in fpr if w is not None):,} ({time.time()-t0:.0f}s)", flush=True)

    os.makedirs(OUT, exist_ok=True)
    d[["ik14", "smiles", "mass"]].to_parquet(os.path.join(OUT, "coco_pool.parquet"), index=False)
    np.save(os.path.join(OUT, "coco_fp.npy"), fp)
    print("saved", os.path.join(OUT, "coco_pool.parquet"), os.path.join(OUT, "coco_fp.npy"))

if __name__ == "__main__":
    main()
