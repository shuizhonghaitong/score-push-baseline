"""Local re-ranking harness for CASMI26 V465 artifacts.

Faithfully reproduces the pipeline's same-formula simulator re-rank (fuse.rerank
semantics: z with ddof=1 over covered members, missing->0, in-place group re-sort
within the top-N window) and stacks the GLACIER term. Candidate pool is rebuilt
from the dumped channels via RRF. CPU-only.

Inputs (dir): e004_candidate_inputs.json, lilac_sim_scores.json, labels.json
Usage:
  python src/rerank_local.py --artifacts DIR --labels labels.json [--sweep] [--window 40]
"""
import argparse, json, os, sys, math
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metric_common import metric_identity
from rdkit import Chem
from rdkit.Chem.rdMolDescriptors import CalcMolFormula

def _formula(smi):
    try:
        m = Chem.MolFromSmiles(smi)
        return CalcMolFormula(m) if m is not None else None
    except Exception:
        return None

def load_pool(art):
    cand = json.load(open(os.path.join(art, 'e004_candidate_inputs.json')))
    p = os.path.join(art, 'lilac_sim_scores.json')
    sim = json.load(open(p)) if os.path.exists(p) else {'ice': {}, 'gl': {}}
    return cand, sim

def build_molecule_pool(mid, cand, sim):
    pool = {}
    def add(smi, channel, rank):
        smi = (smi or '').strip()
        if not smi or smi == 'CCO':
            return
        d = pool.setdefault(smi, {'ch': {}, 'ice': None, 'gl': None, 'form': None})
        d['ch'].setdefault(channel, rank)
    for i, s in enumerate(cand.get('prefusion', {}).get(mid, []) or [], 1): add(s, 'prefusion', i)
    e2 = cand.get('engine2', {}).get(mid, {}) or {}
    for i, s in enumerate(e2.get('smiles', []) or [], 1): add(s, 'engine2', i)
    pc = cand.get('pubchem', {}).get(mid, {}) or {}
    for i, s in enumerate(pc.get('pc', []) or [], 1): add(s, 'pubchem', i)
    for i, s in enumerate(cand.get('base_tail', {}).get(mid, []) or [], 1): add(s, 'base_tail', i)
    for i, s in enumerate(cand.get('fusion_overflow', {}).get(mid, []) or [], 1): add(s, 'fusion_overflow', i)
    ice = sim.get('ice', {}).get(mid, {}) or {}
    gl = sim.get('gl', {}).get(mid, {}) or {}
    for smi, d in pool.items():
        d['ice'] = ice.get(smi)
        d['gl'] = gl.get(smi)
        d['form'] = _formula(smi)
    return pool

def _z(vals):
    """pandas groupby z: ddof=1 over non-missing finite; n<2 or sd==0 -> 0; missing -> 0."""
    ok = [v is not None and isinstance(v, (int, float)) and math.isfinite(float(v)) for v in vals]
    xs = [float(v) for v, o in zip(vals, ok) if o]
    n = len(xs)
    if n < 2:
        return [0.0] * len(vals)
    mu = sum(xs) / n
    sd = math.sqrt(sum((x - mu) ** 2 for x in xs) / (n - 1))
    if not sd > 0:
        return [0.0] * len(vals)
    return [((float(v) - mu) / sd) if o else 0.0 for v, o in zip(vals, ok)]

def _dedupe_keep_order(smis):
    out, seen = [], set()
    for s in smis:
        k = metric_identity(s)
        if k is None or k in seen:
            continue
        seen.add(k); out.append(s)
    return out

def rrf_base_order(pool, ch_w=None, k=6.0):
    ch_w = ch_w or {'prefusion': 1.0, 'engine2': 0.6, 'pubchem': 0.35, 'base_tail': 0.2, 'fusion_overflow': 0.2}
    scored = {}
    for smi, d in pool.items():
        s = sum(ch_w.get(ch, 0.2) / (k + r) for ch, r in d['ch'].items())
        scored[smi] = s
    order = sorted(scored, key=lambda s: (-scored[s], s))
    return order, {s: scored[s] for s in order}

def _clean(v):
    try:
        return float(v) if v is not None and math.isfinite(float(v)) else None
    except (TypeError, ValueError):
        return None

def same_formula_rerank(order, base_scores, pool, ice_lam, gl_lam, window=40, min_covered=2, tie_eps=1e-9):
    """Exact replica of gl_fuse.rerank_multi with two re-scorers (ice, gl)."""
    n = len(order)
    if n == 0:
        return order
    use = []
    if ice_lam:
        use.append((lambda s: _clean(pool[s]['ice']), float(ice_lam)))
    if gl_lam:
        use.append((lambda s: _clean(pool[s]['gl']), float(gl_lam)))
    if not use:
        return order
    m = min(window, n)
    head = order[:m]
    groups = {}
    for pos, smi in enumerate(head):
        f = pool[smi]['form'] or smi
        groups.setdefault(f, []).append(pos)
    order = list(order)
    for f, idx in groups.items():
        if len(idx) < 2:
            continue
        cols = []
        for of, lam in use:
            v = [of(head[p]) for p in idx]
            cols.append((v, lam, sum(x is not None for x in v)))
        if not any(c >= max(2, min_covered) for _, _, c in cols):
            continue
        fused = _z([base_scores[head[p]] for p in idx])
        n_act = 0
        for v, lam, c in cols:
            if c >= 2:
                zi = _z(v)
                fused = [a + lam * b for a, b in zip(fused, zi)]
                n_act += 1
        if n_act >= 2 and tie_eps > 0:
            fused = [round(x / tie_eps) for x in fused]
        new = [head[idx[q]] for q in sorted(range(len(idx)), key=lambda q: (-fused[q], q))]
        for slot_pos, src in zip(idx, new):
            order[slot_pos] = src
    return order

def rerank_molecule(pool, ice_lam=1.0, gl_lam=1.0, ch_w=None, window=40, top=25, k=6.0):
    if not pool:
        return []
    order, base_scores = rrf_base_order(pool, ch_w, k)
    order = _dedupe_keep_order(order)
    base_scores = {s: base_scores[s] for s in order}
    order = same_formula_rerank(order, base_scores, pool, ice_lam, gl_lam, window)
    return order[:top]

def evaluate(art, labels, ice_lam=1.0, gl_lam=1.0, ch_w=None, window=40, top=25):
    cand, sim = load_pool(art)
    lists = {}
    for mid in labels:
        if mid not in cand.get('prefusion', {}) and mid not in cand.get('engine2', {}):
            continue
        pool = build_molecule_pool(mid, cand, sim)
        lists[mid] = rerank_molecule(pool, ice_lam, gl_lam, ch_w, window, top)
    from metric_common import mrr_topk
    return mrr_topk(lists, labels, k=top)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--artifacts', required=True)
    ap.add_argument('--labels', required=True)
    ap.add_argument('--sweep', action='store_true')
    ap.add_argument('--ice', type=float, default=1.0)
    ap.add_argument('--gl', type=float, default=1.0)
    ap.add_argument('--window', type=int, default=40)
    args = ap.parse_args()
    labels = json.load(open(args.labels))
    if args.sweep:
        best = (0, None)
        for il in (0.0, 0.5, 1.0, 1.5, 2.0):
            for gl in (0.0, 0.5, 1.0, 1.5, 2.0):
                m, _ = evaluate(args.artifacts, labels, il, gl, window=args.window)
                print(f'ice={il} gl={gl} w={args.window} mrr={m:.4f}', flush=True)
                if m > best[0]:
                    best = (m, (il, gl))
        print('BEST', best, flush=True)
    else:
        m, per = evaluate(args.artifacts, labels, args.ice, args.gl, window=args.window)
        print(f'MRR@25 = {m:.4f}', flush=True)
        ranks = [v for v in per.values() if v]
        if ranks:
            import collections
            c = collections.Counter(min(r, 6) for r in ranks)
            print('rank dist (1,2,3,4,5,6+):', dict(sorted(c.items())), flush=True)

if __name__ == '__main__':
    main()
