"""Stage-3 decision: baseline (as-submitted) in-sample MRR vs rerank sweep.

Usage:
  .venv/bin/python src/stage3_decision.py --artifacts DIR --labels labels.json --submitted submission.csv

Builds each molecule's candidate pool once, then scores the submitted list and a
grid of (ice, gl, window) configs under the official InChIKey14 MRR@25.
Prints top configs and a GO/NO-GO line (GO if best - baseline >= 0.002).
"""
import argparse, itertools, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rerank_local import load_pool, build_molecule_pool, rerank_molecule
from metric_common import metric_identity, mrr_topk

def parse_submitted(csv_path):
    import pandas as pd
    s = pd.read_csv(csv_path)
    lists = {}
    for _, row in s.iterrows():
        mid = str(row['molecule_id'])
        parts = [p.strip() for p in str(row['smiles']).split(';') if p.strip() and p.strip() != 'CCO']
        seen, ded = set(), []
        for p in parts:
            k = metric_identity(p)
            if k is not None and k not in seen:
                seen.add(k)
                ded.append(p)
        lists[mid] = ded[:25]
    return lists

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--artifacts', required=True)
    ap.add_argument('--labels', required=True)
    ap.add_argument('--submitted', required=True)
    ap.add_argument('--threshold', type=float, default=0.002)
    a = ap.parse_args()
    t0 = time.time()
    cand, sim = load_pool(a.artifacts)
    labels = json.load(open(a.labels))
    submitted = parse_submitted(a.submitted)

    pools, mids = {}, []
    for mid in labels:
        if mid not in cand.get('prefusion', {}) and mid not in cand.get('engine2', {}):
            continue
        pool = build_molecule_pool(mid, cand, sim)
        if pool:
            pools[mid] = pool
            mids.append(mid)
    print(f'pools built: {len(mids)} mols in {time.time()-t0:.0f}s', flush=True)

    lab_subset = {m: labels[m] for m in mids}
    base_mrr, base_ranks = mrr_topk(submitted, lab_subset, k=25)
    print(f'BASELINE (as-submitted) MRR@25 = {base_mrr:.5f}  (matched mols={len(mids)})', flush=True)

    results = []
    for ice, gl, window in itertools.product((0.0, 0.5, 1.0, 1.5, 2.0), (0.0, 0.5, 1.0, 1.5, 2.0), (25, 40, 60, 80)):
        if ice == 0.0 and gl == 0.0:
            continue
        lists = {m: rerank_molecule(pools[m], ice, gl, None, window, 25) for m in mids}
        mrr, _ = mrr_topk(lists, lab_subset, k=25)
        results.append((mrr, ice, gl, window))
        print(f'  ice={ice} gl={gl} window={window} mrr={mrr:.5f}', flush=True)
    results.sort(key=lambda r: -r[0])
    print('\nTOP 10:')
    for mrr, ice, gl, window in results[:10]:
        print(f'  mrr={mrr:.5f} ice={ice} gl={gl} window={window} delta={mrr-base_mrr:+.5f}')
    best = results[0]
    go = best[0] - base_mrr >= a.threshold
    print(f'\nDECISION: best={best[0]:.5f} (ice={best[1]} gl={best[2]} window={best[3]}) '
          f'baseline={base_mrr:.5f} delta={best[0]-base_mrr:+.5f} threshold={a.threshold} -> {"GO" if go else "NO-GO"}')

if __name__ == '__main__':
    main()
