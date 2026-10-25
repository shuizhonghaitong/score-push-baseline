"""Build {molecule_id: true_smiles} for the visible 400 test molecules from train.parquet
(the visible test overlaps train). Usage: python src/build_labels.py <train.parquet> <test.parquet> <out.json>"""
import json, sys
import pandas as pd

train_path, test_path, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
te = pd.read_parquet(test_path, columns=['molecule_id'])
ids = set(te.molecule_id.astype(str))
tr = pd.read_parquet(train_path, columns=[c for c in ['molecule_id', 'smiles', 'inchikey']
                                          if True])
# discover the label column
print('train columns sample:', list(tr.columns)[:20], flush=True)
lab_col = None
for c in tr.columns:
    if c.lower() in ('smiles', 'canonical_smiles', 'smiles_canonical'):
        lab_col = c
        break
if lab_col is None:
    raise SystemExit('no smiles column found in train: ' + str(list(tr.columns)))
tr = tr[tr.molecule_id.astype(str).isin(ids)][['molecule_id', lab_col]].drop_duplicates()
lab = {str(k): str(v) for k, v in zip(tr.molecule_id, tr[lab_col])}
print('labeled', len(lab), 'of', len(ids), flush=True)
json.dump(lab, open(out_path, 'w'))
