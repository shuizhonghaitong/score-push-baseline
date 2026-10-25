"""Build a final submission CSV from V465 artifacts + chosen rerank config.

Usage:
  .venv/bin/python src/make_final_sub.py --artifacts DIR --labels labels.json \
      [--ice 1.0] [--gl 1.0] [--window 40] [--top 25] --out work/final_sub.csv

Row format matches the pipeline: (molecule_id, ';'.join(top25)); 'CCO' if no candidates.
"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rerank_local import load_pool, build_molecule_pool, rerank_molecule

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--artifacts', required=True)
    ap.add_argument('--labels', required=True)
    ap.add_argument('--ice', type=float, default=1.0)
    ap.add_argument('--gl', type=float, default=1.0)
    ap.add_argument('--window', type=int, default=40)
    ap.add_argument('--top', type=int, default=25)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    cand, sim = load_pool(a.artifacts)
    labels = json.load(open(a.labels))
    mids = set(labels.keys())
    for grp in ('prefusion', 'engine2', 'pubchem', 'base_tail', 'fusion_overflow'):
        mids |= set((cand.get(grp) or {}).keys())
    rows, n_pad = [], 0
    for mid in sorted(mids):
        pool = build_molecule_pool(mid, cand, sim)
        order = rerank_molecule(pool, a.ice, a.gl, None, a.window, a.top)
        order = [s for s in order if s and s != 'CCO']
        if order:
            rows.append((mid, ';'.join(order[:a.top])))
        else:
            rows.append((mid, 'CCO'))
            n_pad += 1
    import pandas as pd
    pd.DataFrame(rows, columns=['molecule_id', 'smiles']).to_csv(a.out, index=False)
    test_p = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          'data', 'enveda-casmi26-molecule-id-mass-spectra', 'test.parquet')
    extra = ''
    if os.path.exists(test_p):
        tm = set(pd.read_parquet(test_p, columns=['molecule_id'])['molecule_id'].astype(str))
        missing = tm - set(mids)
        if missing:
            extra = f' WARNING missing_test_mols={len(missing)} {sorted(missing)[:5]}'
        else:
            extra = ' test_mols_complete'
    print(f'wrote {a.out}: rows={len(rows)} padded_CCO={n_pad} '
          f'ice={a.ice} gl={a.gl} window={a.window} top={a.top}{extra}')

if __name__ == '__main__':
    main()
