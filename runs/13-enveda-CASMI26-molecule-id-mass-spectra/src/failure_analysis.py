"""Per-molecule failure analysis of a scored run.
Usage: .venv/bin/python src/failure_analysis.py <artifacts_dir> <labels.json>
artifacts_dir: submission.csv + e004_candidate_inputs.json (+ lilac_sim_scores.json)
"""
import json, os, sys, collections
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metric_common import metric_identity, mrr_topk

art, lab_path = sys.argv[1], sys.argv[2]
labels = json.load(open(lab_path))
sub = pd.read_csv(os.path.join(art, 'submission.csv'), dtype=str)
cand = json.load(open(os.path.join(art, 'e004_candidate_inputs.json')))
sp = os.path.join(art, 'lilac_sim_scores.json')
sim = json.load(open(sp)) if os.path.exists(sp) else {'ice': {}, 'gl': {}}

final = {r.molecule_id: [s.strip() for s in str(r.smiles).split(';')] for r in sub.itertuples(index=False)}
mrr, per_rank = mrr_topk({m: final[m] for m in final}, labels, k=25)
print(f'FINAL MRR@25 (in-sample) = {mrr:.4f}', flush=True)

rows = []
for mid, lab in labels.items():
    lab_id = metric_identity(lab)
    fl = final.get(mid, [])
    rank = next((i + 1 for i, s in enumerate(fl) if metric_identity(s) == lab_id), None)
    chans = {}
    for ch in ('prefusion', 'engine2', 'pubchem', 'base_tail', 'fusion_overflow'):
        v = cand.get(ch, {}).get(mid)
        if ch == 'engine2': v = (v or {}).get('smiles', [])
        elif ch == 'pubchem': v = (v or {}).get('pc', [])
        v = v or []
        r = next((i + 1 for i, s in enumerate(v) if metric_identity(s) == lab_id), None)
        if r: chans[ch] = r
    ice = (sim.get('ice', {}).get(mid) or {})
    gl = (sim.get('gl', {}).get(mid) or {})
    lab_ice = ice.get(lab); lab_gl = gl.get(lab)
    top1 = fl[0] if fl else None
    rows.append(dict(mid=mid, final_rank=rank, in_final=rank is not None,
                     top1_in_final_same_id=(top1 is not None and metric_identity(top1) == lab_id),
                     channels=chans, n_pool=sum(1 for ch in ('prefusion','engine2','pubchem','base_tail','fusion_overflow') for s in [cand.get(ch,{}).get(mid)] if s),
                     ice_lab=lab_ice, gl_lab=lab_gl,
                     ice_top1=ice.get(top1), gl_top1=gl.get(top1)))

df = pd.DataFrame(rows)
df.to_csv(os.path.join(art, 'failure_analysis.csv'), index=False)
hit = df['in_final'].sum()
print(f'label found in final top-25: {hit}/{len(df)}', flush=True)
print('rank dist:', df['final_rank'].value_counts().sort_index().to_dict(), flush=True)
missing = df[~df['in_final']]
print(f'\nMISSING from final ({len(missing)}):', flush=True)
for r in missing.itertuples(index=False):
    print(f"  {r.mid} chans={r.channels} ice_lab={r.ice_lab} gl_lab={r.gl_lab}", flush=True)
# how many missing were in some channel (re-rank fixable) vs absent from pool
fixable = sum(1 for r in missing.itertuples(index=False) if r.channels)
print(f'missing-but-in-pool (rerank-fixable): {fixable}; absent from all channels: {len(missing)-fixable}', flush=True)
# same-formula top1 (simulator-fixable)
sf = df[df['top1_in_final_same_id'] & ~df['in_final']]
print(f'top1 present but not label (incl same-identity): n/a (same_id counted as hit)', flush=True)
