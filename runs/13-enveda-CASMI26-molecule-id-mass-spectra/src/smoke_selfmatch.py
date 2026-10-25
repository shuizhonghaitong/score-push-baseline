"""Synthetic self-match smoke test for the CASMI V2 engine (no real data needed).
Generates M synthetic molecules (unique mass + spectrum), builds a fake library,
queries with the same molecules, and reports MRR@25 / top1. Used as a fast local
regression check when the engine is modified.
"""
import os, sys, time, random
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "work", "nb_v1"))
import casmi_v2 as E

def make_data(M, seed=0, n_train_per_mol=1, n_iso_mass_bump=True):
    rng = np.random.default_rng(seed)
    rows = []
    mol_info = []
    used = set()
    for i in range(M):
        # choose a formula mass by picking carbon count
        nc = int(rng.integers(3, 20))
        formula = f"C{nc}H{2*nc+2}O{int(rng.integers(0,3))}"
        nm = E.formula_mass(formula)
        # ensure mass not colliding
        key = round(nm, 3)
        k = 0
        while key in used and k < 50:
            nc += 1; formula = f"C{nc}H{2*nc+2}O{int(rng.integers(0,3))}"
            nm = E.formula_mass(formula); key = round(nm,3); k += 1
        used.add(key)
        ik = f"AAAAAAAAAAAA{i:03d}"  # 14-char-ish inchikey14
        smi = f"{'C'*(nc)}{'O'*(int(rng.integers(0,3)))}"
        # random spectrum
        npeaks = int(rng.integers(40, 120))
        mzs = np.sort(rng.uniform(15, nm-2, npeaks)).astype(np.float32)
        ints = rng.uniform(0.001, 1.0, npeaks)
        ints = (ints/ints.sum()).astype(np.float32)
        mol_info.append((i, nm, ik, smi, formula, mzs.copy(), ints.copy()))
    # train rows
    for (i, nm, ik, smi, formula, mzs, ints) in mol_info:
        for t in range(n_train_per_mol):
            add = "[M+H]+"
            prec = nm + E.PROTON
            rows.append(dict(ms2_mzs=list(mzs), ms2_normalized_intensities=list(ints),
                             adduct=add, precursor_mz=float(prec), inchikey14=ik,
                             base_peak_intensity=float(np.max(ints)*1e6),
                             normalized_smiles=smi, molecular_formula=formula,
                             molecule_id=f"m{i}", spectrum_id=f"s{i}_{t}"))
    tr = pd.DataFrame(rows)
    # test rows: one spectrum per mol (self match)
    trows = []
    for (i, nm, ik, smi, formula, mzs, ints) in mol_info:
        add = "[M+H]+"
        prec = nm + E.PROTON
        trows.append(dict(molecule_id=f"m{i}", spectrum_id=f"q{i}",
                          ms2_mzs=list(mzs), ms2_normalized_intensities=list(ints),
                          base_peak_intensity=float(np.max(ints)*1e6),
                          adduct=add, precursor_mz=float(prec)))
    te = pd.DataFrame(trows)
    truth = {f"m{i}": f"{'C'*(int(m[3].replace('O','')))}" for i,*m in []}  # placeholder
    truth_smis = {f"m{i}": smi for (i, nm, ik, smi, formula, mzs, ints) in mol_info}
    return tr, te, truth_smis

def main():
    M = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    ntpm = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    tr, te, truth_smis = make_data(M, n_train_per_mol=ntpm)
    print(f"synthetic: {M} mols, train {len(tr)} rows")
    t0=time.time()
    out = os.path.join(os.path.dirname(__file__), "..", "work", "smoke_out.csv")
    sub = E.run_core(tr, te, out_path=out, logf=lambda m: print("  [eng]", m))
    dt=time.time()-t0
    smi_to_ik = {}
    for s,k in zip(tr.normalized_smiles, tr.inchikey14):
        smi_to_ik.setdefault(str(s), str(k))
    # build ik set per mol
    ik_of_smi = {str(s): str(k) for s,k in zip(tr.normalized_smiles, tr.inchikey14)}
    truth_ik = {}
    for (mid, s) in truth_smis.items():
        truth_ik[mid] = ik_of_smi.get(s)
    tot=0.0; top1=0; top5=0
    for r in sub.itertuples(index=False):
        smis=[x for x in str(r.smiles).split(";") if x]
        ti = truth_ik.get(r.molecule_id)
        for rank,s in enumerate(smis[:25],1):
            if ik_of_smi.get(s)==ti:
                tot+=1.0/rank
                if rank==1: top1+=1
                if rank<=5: top5+=1
                break
    n=len(sub)
    print(f"\nRESULT M={M} ntpm={ntpm}  MRR@25={tot/n:.4f} top1={top1/n:.4f} top5={top5/n:.4f}  time={dt:.1f}s")

if __name__=="__main__":
    main()
