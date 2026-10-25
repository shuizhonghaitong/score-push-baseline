"""Build ice/gl runner input for ALL 400 test mols from scored-kernel candidate pool.
Usage: .venv/bin/python work/build_sim_input.py <pool_dir> <out.json>
pool_dir contains e004_candidate_inputs.json (from the scored V465 fork)."""
import json, os, sys
import pandas as pd, numpy as np

RUN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
pool_dir, out_path = sys.argv[1], sys.argv[2]
cand = json.load(open(os.path.join(pool_dir, 'e004_candidate_inputs.json')))
df = pd.read_parquet(f'{RUN}/data/enveda-casmi26-molecule-id-mass-spectra/test.parquet')

def clean(v):
    if v is None: return None
    if isinstance(v, (list, np.ndarray)): return None if len(v) == 0 else v
    if pd.isna(v): return None
    return v

jobs = []
for mid, row in df.drop_duplicates('molecule_id').iterrows():
    m = str(row['molecule_id'])
    seen, cands = set(), []
    for ch in ('prefusion', 'engine2', 'pubchem', 'base_tail', 'fusion_overflow'):
        v = cand.get(ch, {}).get(m)
        if ch == 'engine2': v = (v or {}).get('smiles', [])
        elif ch == 'pubchem': v = (v or {}).get('pc', [])
        for s in v or []:
            s = str(s or '').strip()
            if s and s != 'CCO' and s not in seen:
                seen.add(s); cands.append(s)
    if not cands:
        continue
    ce = clean(row['collision_energy_ev'])
    spec = {'mz': list(np.asarray(row['ms2_mzs'], dtype=float)),
            'it': list(np.asarray(row['ms2_normalized_intensities'], dtype=float)),
            'prec': float(row['precursor_mz']),
            'adduct': str(row['adduct']),
            'mode': str(row['ionization_mode']),
            'instrument': str(row['instrument_type']),
            'ce_ev': list(np.asarray(ce, dtype=float)) if ce is not None else None,
            'ce_orig': str(clean(row['collision_energy_orig'])) if clean(row['collision_energy_orig']) is not None else None,
            'ce_units': str(clean(row['collision_energy_orig_units'])) if clean(row['collision_energy_orig_units']) is not None else None}
    jobs.append({'mid': m, 'cands': cands, 'spectra': [spec]})
json.dump(jobs, open(out_path, 'w'))
print(f'{len(jobs)} mols, {sum(len(j["cands"]) for j in jobs)} cand-jobs, max {max(len(j["cands"]) for j in jobs)}')
