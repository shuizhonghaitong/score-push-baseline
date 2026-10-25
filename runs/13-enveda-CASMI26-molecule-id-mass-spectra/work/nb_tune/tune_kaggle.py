# Use the Dependency Manager
# pip install rdkit==2026.3.3
# pip install numba
"""Kaggle-side tuning for CASMI26 V2 engine.
Builds the library ONCE, derives ground truth for the visible test (its molecules
are drawn from train, so a >=GT_SIM train twin gives the identity), then runs a
param grid of infer() settings and reports MRR@25 per setting -> eval_grid.json.
"""
import os, glob, time, json
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from casmi_v2 import (build_lib, infer, extract_flat, pa_mask, neutral_mass_array,
                      neutral_mass, _clean, search_pre,
                      INT_FLOOR, MAX_PEAKS, INT_POWER, ENT_WEIGHT)

GT_SIM = 0.92      # ground-truth twin similarity threshold
GT_PPM = 30.0      # ground-truth search window (ppm)
GT_MZTOL = 0.02    # looser mz tol for the twin lookup

T0 = time.time()
def log(m): print(f"[{time.time()-T0:7.1f}s] {m}", flush=True)

hits = glob.glob('/kaggle/input/**/train.parquet', recursive=True)
COMP = os.path.dirname(sorted(hits, key=len)[0])
TEST = os.path.join(COMP, 'test.parquet')
log("reading train")
tbl = pq.read_table(os.path.join(COMP, 'train.parquet'),
    columns=['ms2_mzs', 'ms2_normalized_intensities', 'adduct', 'precursor_mz',
             'inchikey14', 'base_peak_intensity', 'normalized_smiles', 'molecular_formula'])
nms = neutral_mass_array(tbl.column('precursor_mz').to_numpy(),
                         np.asarray(tbl.column('adduct').to_pylist(), dtype=object))
mask = np.isfinite(nms)
flat = extract_flat(tbl, mask)
tr = tbl.filter(pa_mask(mask)).drop(['ms2_mzs', 'ms2_normalized_intensities']).to_pandas()
log(f"train {len(tr):,}")
te = pd.read_parquet(TEST)
log(f"test {len(te)} spectra, {te.molecule_id.nunique()} mols")

log("building lib (once)")
L = build_lib(tr, flat=flat, logf=log)
smi_to_ik = {str(s): str(k) for s, k in zip(L['pool']['smiles'], L['pool']['keystr'])}
log(f"pool {len(L['pool']['smiles']):,} structures; smi map {len(smi_to_ik):,}")

def ground_truth(L, te):
    lib = L['lib']
    truth, bestsims = {}, []
    for mid, g in te.groupby('molecule_id'):
        cand, bestsim = {}, -1.0
        for r in g.itertuples(index=False):
            nm = neutral_mass(r.precursor_mz, r.adduct)
            if not np.isfinite(nm):
                continue
            qm, qp = _clean(np.asarray(r.ms2_mzs, np.float32),
                            np.asarray(r.ms2_normalized_intensities, np.float32),
                            INT_FLOOR, MAX_PEAKS, INT_POWER, ENT_WEIGHT)
            if len(qm) < 3:
                continue
            tol = nm * GT_PPM * 1e-6
            lo = int(np.searchsorted(lib['snm'], nm - tol, 'left'))
            hi = int(np.searchsorted(lib['snm'], nm + tol, 'right'))
            if hi <= lo:
                continue
            sc = search_pre(qm, qp, lo, hi, lib['roff'], lib['rmz'], lib['rit'], GT_MZTOL)
            i = int(np.argmax(sc))
            if sc[i] > bestsim:
                bestsim = float(sc[i])
            if sc[i] >= GT_SIM:
                ik = str(lib['keystr'][lo + i]); cand[ik] = cand.get(ik, 0) + 1
        bestsims.append(best if (best := bestsim) else bestsim)
        truth[str(mid)] = max(cand, key=cand.get) if cand else None
    bs = np.array(best for best in bestsims)
    return truth, bs

log(f"computing ground truth (twin sim>={GT_SIM}, window {GT_PPM}ppm)")
truth, bsims = ground_truth(L, te)
matched = sum(1 for v in truth.values() if v)
log(f"ground truth matched {matched}/{len(truth)}  "
    f"bestsim med={np.median(bsims):.3f} p25={np.percentile(bsims,25):.3f} "
    f"frac>={GT_SIM}:{np.mean(np.asarray(bsims)>=GT_SIM):.3f}")

def mrr25(smis, tik):
    for i, s in enumerate(smis[:25], 1):
        if smi_to_ik.get(s) == tik:
            return 1.0 / i
    return 0.0

def evaluate(sub, truth):
    tot, top1, top5, n = 0.0, 0, 0, 0
    for r in sub.itertuples(index=False):
        t = truth.get(r.molecule_id)
        if t is None:
            continue
        n += 1
        smis = [x for x in str(r.smiles).split(';') if x]
        v = mrr25(smis, t); tot += v
        if v >= 1.0: top1 += 1
        if v >= 0.2: top5 += 1
    return {'n': n, 'mrr25': round(tot / n, 4) if n else 0.0,
            'top1': round(top1 / n, 4) if n else 0.0,
            'top5': round(top5 / n, 4) if n else 0.0}

settings = [
    {},
    dict(DIRECT_GATE=0.5),
    dict(DIRECT_GATE=0.4),
    dict(DIRECT_GATE=0.7),
    dict(LIB_PPM_WIN=15.0),
    dict(LIB_PPM_WIN=5.0),
    dict(W_ANALOG=0.3),
    dict(W_ANALOG=0.8),
    dict(ANALOG_MIN_SIM=0.3),
    dict(ANALOG_MIN_SIM=0.5),
    dict(DIRECT_GATE=0.5, LIB_PPM_WIN=15.0),
    dict(LIB_PPM_WIN=15.0, W_ANALOG=0.3),
]
results = []
for i, st in enumerate(settings):
    t0 = time.time()
    sub = infer(L, te, out_path=f'/kaggle/working/tune_{i:02d}.csv', **st)
    res = evaluate(sub, truth)
    results.append({'i': i, 'params': st, **res, 'sec': round(time.time() - t0, 1)})
    log(f"set{i:02d} {st} -> {res} ({time.time()-t0:.0f}s)")

log("saving baseline full submission")
sub0 = infer(L, te, out_path='/kaggle/working/submission_baseline.csv')
base = evaluate(sub0, truth)
log(f"BASELINE -> {base}")

out = {'n_test_mols': int(te.molecule_id.nunique()),
       'ground_truth_matched': matched, 'ground_truth_total': len(truth),
       'gt_sim': GT_SIM, 'baseline': base, 'grid': results}
with open('/kaggle/working/eval_grid.json', 'w') as f:
    json.dump(out, f, indent=1)
log("DONE -> eval_grid.json")
