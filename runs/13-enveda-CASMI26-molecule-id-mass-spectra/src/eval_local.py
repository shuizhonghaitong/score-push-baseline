"""Local evaluation of work/nb_v1/casmi_v1.py against proxies:
   P1: hold out ONE spectrum per molecule, keep its other spectra in the library.
   P2: hold out ALL spectra of N molecules; pool still contains the structures.
   VIS: visible test.parquet (labels from train) — inflated sanity check.
"""
import os, sys, time, json, argparse
import numpy as np
import pandas as pd

RUN = "/mnt/disk3/liuyutong_home/workspace/skill/score-push/runs/13-enveda-CASMI26-molecule-id-mass-spectra"
DATA = os.path.join(RUN, "data", "enveda-casmi26-molecule-id-mass-spectra")
sys.path.insert(0, os.path.join(RUN, "work", "nb_v1"))
import casmi_v1  # noqa: E402

TRAIN_COLS = ['ms2_mzs', 'ms2_normalized_intensities', 'adduct', 'precursor_mz',
              'inchikey14', 'base_peak_intensity', 'normalized_smiles', 'molecular_formula',
              'molecule_id', 'spectrum_id']
TEST_COLS = ['molecule_id', 'spectrum_id', 'ms2_mzs', 'ms2_normalized_intensities',
             'adduct', 'precursor_mz', 'base_peak_intensity', 'ionization_mode',
             'instrument_type', 'collision_energy_ev']

def load_train():
    t0 = time.time()
    tr = pd.read_parquet(os.path.join(DATA, "train.parquet"), columns=TRAIN_COLS)
    print(f"train loaded: {len(tr):,} rows ({time.time()-t0:.0f}s)")
    return tr

def mrr25(ranked_keys, truth):
    for i, k in enumerate(ranked_keys[:25], start=1):
        if k == truth:
            return 1.0 / i
    return 0.0

def evaluate(sub, truth_map, keymap):
    """sub: DataFrame(molecule_id, smiles); truth_map: molecule_id -> inchikey14 str"""
    total, hit1, hit5 = 0.0, 0, 0
    miss = 0
    for r in sub.itertuples(index=False):
        smis = [s for s in str(r.smiles).split(";") if s]
        truth = keymap.get(truth_map[r.molecule_id])
        if truth is None:
            miss += 1
            continue
        r_ = mrr25([keymap.get(s, None) for s in smis], truth)
        total += r_
        if r_ >= 1.0: hit1 += 1
        if r_ >= 0.2: hit5 += 1
    n = len(sub)
    print(json.dumps({"n": n, "missing_truth": miss, "mrr25": round(total/n, 4),
                      "top1": round(hit1/n, 4), "top5": round(hit5/n, 4)}, indent=1))
    return total / n

def build_keymap(smiles_series):
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdMolDescriptors
    RDLogger.DisableLog('rdApp.*')
    km = {}
    for i, s in enumerate(smiles_series):
        if i % 20000 == 0:
            print(f"  keymap {i}/{len(smiles_series)}", flush=True)
        m = Chem.MolFromSmiles(str(s))
        if m is None:
            continue
        ik = rdMolDescriptors.CanonicalRankSMILES(m)  # placeholder, not used
    return km

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["p1", "p2", "vis"], default="p2")
    ap.add_argument("--n-mols", type=int, default=400)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    tr = load_train()
    rng = np.random.default_rng(args.seed)
    os.makedirs(os.path.join(RUN, "artifacts"), exist_ok=True)

    if args.mode == "vis":
        te = pd.read_parquet(os.path.join(DATA, "test.parquet"))
        out = os.path.join(RUN, "artifacts", "sub_vis.csv")
        casmi_v1.run_core(tr, te, out_path=out, logf=print)
        lab = tr.set_index("spectrum_id").inchikey14.to_dict()
        te = pd.read_parquet(os.path.join(DATA, "test.parquet"), columns=["spectrum_id", "molecule_id"])
        truth_map = {}
        for mid, g in te.groupby("molecule_id"):
            ik = [lab[s] for s in g.spectrum_id if s in lab]
            truth_map[mid] = ik[0] if ik else None
        smi_to_ik = {}
        for s, k in zip(tr.normalized_smiles, tr.inchikey14):
            s = str(s)
            if s not in smi_to_ik:
                smi_to_ik[s] = str(k)
        evaluate(pd.read_csv(out), truth_map, smi_to_ik)
        sys.exit(0)

    # P1 / P2
    mols = tr.molecule_id.unique()
    hold_mols = set(rng.choice(mols, args.n_mols, replace=False))
    hold = tr.molecule_id.isin(hold_mols)
    t_hold = tr[hold].reset_index(drop=True)
    t_lib = tr[~hold].reset_index(drop=True)
    print(f"hold {t_hold.molecule_id.nunique()} molecules / {len(t_hold):,} spectra; lib {len(t_lib):,}")

    truth_map = t_hold.groupby("molecule_id").inchikey14.first().to_dict()
    pool_extra = None
    if args.mode == "p2":
        lib_df = t_lib
        test_df = t_hold
        pool_extra = t_hold[["inchikey14", "normalized_smiles", "molecular_formula"]].drop_duplicates("inchikey14")
    else:
        # P1: keep all but one spectrum per held molecule in the library
        drop_idx = t_hold.groupby("molecule_id").head(1).index
        lib_df = pd.concat([t_lib, t_hold.loc[~t_hold.index.isin(drop_idx)]], ignore_index=True)
        test_df = t_hold.loc[t_hold.index.isin(drop_idx)]
        print(f"P1: test {len(test_df)} spectra; lib {len(lib_df):,}")

    out = os.path.join(RUN, "artifacts", f"sub_{args.mode}.csv")
    casmi_v1.run_core(lib_df, test_df, out_path=out, pool_extra=pool_extra, logf=print)

    # keymap: smiles -> inchikey14 (from train, using rdkit only to validate smiles)
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog('rdApp.*')
    smi_to_ik = {}
    for s, k in zip(tr.normalized_smiles, tr.inchikey14):
        s = str(s)
        if s not in smi_to_ik:
            smi_to_ik[s] = str(k)
    keymap = smi_to_ik
    evaluate(pd.read_csv(out), truth_map, keymap)
