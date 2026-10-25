"""CASMI26 labels kernel: true SMILES / InChIKey14 for the visible test molecules.

The visible test overlaps train (per run analysis). This CPU script finds, for each
test molecule_id, the matching train row(s) via (A) a shared molecule_id column,
(B) a shared spectrum_id column, or (C) neutral-mass matching on precursor m/z.
Outputs /kaggle/working/labels.csv|json + labels_manifest.json. No GPU, no internet.
"""
import glob, json, os, time
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

T0 = time.time()
comp = os.path.dirname(sorted(glob.glob('/kaggle/input/**/test.parquet', recursive=True), key=len)[0])
TRAIN = os.path.join(comp, 'train.parquet')
TEST = os.path.join(comp, 'test.parquet')
print('COMP', comp, flush=True)

schema = pq.read_schema(TRAIN)
tcols = list(schema.names)
print('TRAIN_COLS', tcols, flush=True)

te = pd.read_parquet(TEST, columns=['molecule_id', 'spectrum_id', 'precursor_mz', 'adduct'])
ids = sorted(te['molecule_id'].astype(str).unique())
print('n_test_mols', len(ids), flush=True)

H, NA, K, NH4, H2O = 1.007276466, 22.989218935, 38.963156364, 18.033827225, 17.002739323
ADDUCTS = {'[M+H]+': (1.0, H), '[M+Na]+': (1.0, NA), '[M+K]+': (1.0, K), '[M+NH4]+': (1.0, NH4),
           '[M-H]-': (-1.0, H), '[M-H2O+H]+': (1.0, H2O - H)}

def neutral_mass(mz, adduct):
    z, d = ADDUCTS.get(adduct, (None, None))
    if z is None or not np.isfinite(mz):
        return np.nan
    return (mz - z * d) / abs(z)

rows, strategy_count = [], {}
strategy = None
train_tbl = None

if 'molecule_id' in tcols:
    strategy = 'molecule_id'
    sub = pq.read_table(TRAIN, columns=['molecule_id'] + [c for c in ['inchikey14', 'normalized_smiles', 'molecular_formula'] if c in tcols])
    train_tbl = sub.to_pandas()
elif 'spectrum_id' in tcols:
    strategy = 'spectrum_id'
    sub = pq.read_table(TRAIN, columns=['spectrum_id'] + [c for c in ['inchikey14', 'normalized_smiles', 'molecular_formula'] if c in tcols])
    train_tbl = sub.to_pandas()

if strategy in ('molecule_id', 'spectrum_id'):
    smi_col = 'normalized_smiles' if 'normalized_smiles' in train_tbl.columns else ('smiles' if 'smiles' in train_tbl.columns else None)
    ik_col = 'inchikey14' if 'inchikey14' in train_tbl.columns else None
    if strategy == 'molecule_id':
        m = train_tbl[train_tbl['molecule_id'].astype(str).isin(set(ids))]
        for mid in ids:
            r = m[m['molecule_id'].astype(str) == mid]
            if len(r) == 0:
                rows.append((mid, np.nan, np.nan, 'miss', 0)); continue
            smi = r[smi_col].dropna().mode() if smi_col else pd.Series([np.nan])
            ik = r[ik_col].dropna().mode() if ik_col else pd.Series([np.nan])
            rows.append((mid, smi.iloc[0] if len(smi) else np.nan, ik.iloc[0] if len(ik) else np.nan, 'molecule_id', len(r)))
    else:
        te_m = te[['molecule_id', 'spectrum_id']]
        j = te_m.merge(train_tbl, on='spectrum_id', how='left')
        for mid in ids:
            r = j[j['molecule_id'] == mid].dropna(subset=['inchikey14'] if ik_col else [smi_col]) if (ik_col or smi_col) else j[j['molecule_id'] == mid]
            r = j[j['molecule_id'] == mid]
            hit = r[r[[c for c in (smi_col, ik_col) if c]].notna().any(axis=1)] if (smi_col or ik_col) else r
            if len(hit) == 0:
                rows.append((mid, np.nan, np.nan, 'miss', 0)); continue
            vals = hit[smi_col].dropna().astype(str).str.strip() if smi_col else pd.Series(dtype=str)
            ikv = hit[ik_col].dropna().astype(str).str.strip() if ik_col else pd.Series(dtype=str)
            smi = vals.mode().iloc[0] if len(vals) else np.nan
            ik = ikv.mode().iloc[0] if len(ikv) else np.nan
            rows.append((mid, smi, ik, 'spectrum_id', len(hit)))
else:
    strategy = 'mass_match'
    want = [c for c in ['inchikey14', 'normalized_smiles', 'molecular_formula', 'precursor_mz', 'adduct'] if c in tcols]
    tr = pq.read_table(TRAIN, columns=want).to_pandas()
    tr['nm'] = [neutral_mass(a, b) for a, b in zip(tr['precursor_mz'], tr['adduct'])]
    tr = tr[np.isfinite(tr['nm'])]
    print('train rows with neutral mass:', len(tr), flush=True)
    te_n = te.groupby('molecule_id').agg(prec=('precursor_mz', 'mean'), add=('adduct', 'first')).reindex(ids)
    for mid in ids:
        p, a = float(te_n.loc[mid, 'prec']), str(te_n.loc[mid, 'add'])
        nm = neutral_mass(p, a)
        if not np.isfinite(nm):
            rows.append((mid, np.nan, np.nan, 'miss_no_mass', 0)); continue
        lo, hi = nm - nm * 10e-6, nm + nm * 10e-6
        m = tr[(tr['nm'] >= lo) & (tr['nm'] <= hi)]
        if len(m) == 0:
            rows.append((mid, np.nan, np.nan, 'miss', 0)); continue
        ikset = set(m['inchikey14'].dropna().astype(str).unique()) if 'inchikey14' in m else set()
        smset = set(m['normalized_smiles'].dropna().astype(str).unique()) if 'normalized_smiles' in m else set()
        tag = 'mass_match' if (len(ikset) <= 1 or len(smset) <= 1) else 'mass_ambiguous'
        smi = m['normalized_smiles'].dropna().mode().iloc[0] if ('normalized_smiles' in m and len(m['normalized_smiles'].dropna())) else np.nan
        ik = m['inchikey14'].dropna().mode().iloc[0] if ('inchikey14' in m and len(m['inchikey14'].dropna())) else np.nan
        rows.append((mid, smi, ik, tag, len(m)))

out = pd.DataFrame(rows, columns=['molecule_id', 'smiles', 'inchikey14', 'strategy', 'n_matches'])
out.to_csv('/kaggle/working/labels.csv', index=False)
ok = out['smiles'].notna()
lab = {str(r.molecule_id): str(r.smiles) for r in out[ok].itertuples()}
json.dump(lab, open('/kaggle/working/labels.json', 'w'))
manifest = dict(train_cols=tcols, strategy=strategy, n_test_mols=len(ids), n_labeled=int(ok.sum()),
                counts=out['strategy'].value_counts().to_dict(), elapsed_s=round(time.time() - T0, 1),
                test_sha=sorted(ids)[0] + '...' + sorted(ids)[-1])
json.dump(manifest, open('/kaggle/working/labels_manifest.json', 'w'), indent=1)
print('LABELS_DONE', json.dumps(manifest), flush=True)
print(out.head(8).to_string(), flush=True)
