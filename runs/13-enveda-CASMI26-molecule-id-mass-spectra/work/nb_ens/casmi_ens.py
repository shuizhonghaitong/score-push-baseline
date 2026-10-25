"""CASMI26 ensemble: RRF fusion of two published top-25 lists (V465 fork output + backup engine output).
Inputs: datasets lilac1996/casmi26-v465-out (submission_v465.csv) and lilac1996/casmi26-entropy-out (submission_entropy.csv).
Writes /kaggle/working/submission.csv. Trivial CPU job; no models, no internet."""
import glob
import pandas as pd

def find(name):
    hits = glob.glob(f'/kaggle/input/**/{name}', recursive=True)
    assert hits, name
    return hits[0]

def load(name):
    df = pd.read_csv(find(name), dtype=str)
    df['molecule_id'] = df['molecule_id'].astype(str)
    return {r.molecule_id: [s.strip() for s in str(r.smiles).split(';') if s.strip() and s.strip() != 'CCO']
            for r in df.itertuples(index=False)}

a = load('submission_v465.csv')
b = load('submission_entropy.csv')
sample = pd.read_csv(glob.glob('/kaggle/input/**/sample_submission.csv', recursive=True)[0], dtype=str)
sample['molecule_id'] = sample['molecule_id'].astype(str)

K = 6.0
out = {}
for mid in sample.molecule_id:
    la, lb = a.get(mid, []), b.get(mid, [])
    scores = {}
    for r, smi in enumerate(la, 1):
        scores[smi] = scores.get(smi, 0.0) + 1.0 / (K + r)
    for r, smi in enumerate(lb, 1):
        scores[smi] = scores.get(smi, 0.0) + 0.85 / (K + r)
    order = sorted(scores, key=lambda s: -scores[s])
    order = (order + ['CCO'] * 25)[:25]
    out[mid] = ';'.join(order)

sub = pd.DataFrame({'molecule_id': sample.molecule_id, 'smiles': [out[m] for m in sample.molecule_id]})
assert len(sub) == len(sample) and sub.smiles.notna().all()
sub.to_csv('/kaggle/working/submission.csv', index=False)
print('ensemble rows', len(sub))
