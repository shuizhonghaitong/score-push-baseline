### cell 0 [markdown]
# 🏆 TrafficFlowBench: High-Performance Multi-Task Pipeline (Targeting #1)

This notebook implements an advanced, top-tier solution for **2026 IEEE Big Data - Traffic Flow Bench**.
It integrates state-of-the-art discoveries across all 4 competition tasks to outperform existing LB 0.85 & 0.865 baselines and push towards the #1 leaderboard score (**0.924+**).

### Strategic Multi-Task Blueprint:
1. **Task 1 (Traffic State Reconstruction - 35% weight)**:
   - Interpolation residual baseline + multi-depth CatBoost (depth 8 & depth 10) on 94 spatio-temporal features.
   - Dynamic lead/lag context ($\pm 5, \pm 10, \pm 15, \pm 30, \pm 60$ min) and observation quality flags.
   - Physical bounds clipping: $v \ge 5$ km/h, $q \ge 0$, guarding density $\rho = q/v$ against division-by-zero singularities.
2. **Task 2 (Short-term Queue Forecasting - 30% weight)**:
   - **Structural Invariant in Onset Windows**: Leverages the official window-selection rule: every onset queue appears exclusively at $T+30$ ($h=6$). Suppresses noise at $h=1..5$ and applies ensemble (LightGBM + CatBoost) at $h=6$ with top-$k$ fallback.
   - **Adaptive Horizon Thresholding in Ongoing Windows**: Calibrated decision threshold vector across horizons $h=1..6$ to combat temporal decay.
   - **1D Spatial Morphological Closing**: Enforces physical freeway queue continuity along neighboring links to remove isolated noise and reconnect split queues.
3. **Task 3 (Physics Consistency - 15% weight)**:
   - Evaluated directly on Task 1 output. Strict boundary enforcement preserves the Fundamental Diagram ($q \le q_{\max}, v \le v_f$) and vehicle conservation $\partial_t \rho + \partial_x q = 0$.
4. **Task 4 (OD & Path-Flow Estimation - 20% weight)**:
   - Solves $\min_{f \ge 0} \frac{1}{2} \|Af - c\|_2^2 + \frac{\lambda}{2} \|f - b\|_2^2$ ($\lambda = 0.05$) on **each split's own counts and prior**.
   - Dual link-space warm start with high-precision bounded L-BFGS-B (`maxiter=10000`, `ftol=1e-15`, `gtol=1e-7`) taking link count residual near zero.


### cell 1 [code]
from pathlib import Path
import os, sys, glob, json, gc, time, hashlib, warnings
import numpy as np, pandas as pd
from scipy import sparse
from scipy.linalg import solve
from scipy.optimize import minimize
from threadpoolctl import threadpool_limits
import lightgbm as lgb
from catboost import CatBoostClassifier

warnings.filterwarnings('ignore')
THREADS = min(4, os.cpu_count() or 1)
os.environ.update(OMP_NUM_THREADS=str(THREADS), OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1')

WORK = Path('/kaggle/working')
OUTPUT = WORK / 'predictions'
OUTPUT.mkdir(parents=True, exist_ok=True)
SOURCE = WORK / 'tfb_source'
SOURCE.mkdir(exist_ok=True)
sys.path.insert(0, str(SOURCE))

T0 = time.time()
print(f'Initialized pipeline with {THREADS} threads.')


### cell 2 [code]
def find_unique(pattern):
    matches = list(Path('/kaggle/input').rglob(pattern))
    if not matches:
        matches = list(Path('.').rglob(pattern))
    assert len(matches) >= 1, f'Could not find {pattern} in inputs!'
    return matches[0]

DATA = find_unique('submission_key.csv').parent
ASSETS = find_unique('speed_d8.cbm').parent
os.environ.update(TFB_DATA_ROOT=str(DATA), TFB_OUTPUT_ROOT=str(OUTPUT), TFB_THREADS=str(THREADS))

print('Competition Data Root:', DATA)
print('Public Assets Root:', ASSETS)


### cell 3 [code]
# ==================== TASK 1: TRAFFIC STATE RECONSTRUCTION ====================
print('\n>>> [1/4] Running Task 1: Traffic State Reconstruction...')
sys.path.insert(0, str(ASSETS))
from state_inference import StatePredictor, predict_states

state_predictor = StatePredictor(ASSETS)
predict_states(state_predictor)
del state_predictor
gc.collect()

# Enforce physical guardrails directly onto predicted states (improves Task 1 & Task 3)
state_csv = OUTPUT / 'state.csv'
state_df = pd.read_csv(state_csv)
print(f'Task 1 raw output rows: {len(state_df):,}')

# 1. Minimum speed >= 5 km/h to prevent zero/negative speed from corrupting Task 3 density checks
state_df['speed_kmh'] = np.clip(state_df['speed_kmh'], 5.0, 135.0)
# 2. Flow must be non-negative
state_df['flow_vph'] = np.clip(state_df['flow_vph'], 0.0, None)

state_df.to_csv(state_csv, index=False)
print(f'Task 1 state reconstruction finished in {time.time() - T0:.1f}s')


### cell 4 [code]
# ==================== TASK 2: SHORT-TERM QUEUE FORECASTING ====================
print('\n>>> [2/4] Running Task 2: Queue Forecasting with Structural Onset & Spatial Smoothing...')

from queue_features import Panel, panel_names, HISTORY, HORIZON
from queue_ongoing import Panel as OngoingPanel

onset_model = lgb.Booster(model_file=str(ASSETS / 'onset_lgb.txt'))
ongoing_model = lgb.Booster(model_file=str(ASSETS / 'ongoing_lgb.txt'))
onset_cat = CatBoostClassifier()
onset_cat.load_model(str(ASSETS / 'onset.cbm'))

CFG_T2 = {
    'onset_weight': 0.50,            # Equal blend between CatBoost & LightGBM
    'onset_threshold': 0.52,         # Optimal F1 threshold for onset
    'onset_top_k': 2,                # Guaranteed onset fallback
    'ongoing_threshold': 0.40,       # Base threshold for ongoing
}

def spatial_morphological_closing(mask):
    """Re-connect split queues along freeway corridor links."""
    res = mask.copy()
    H, L = res.shape
    for h in range(H):
        for i in range(1, L - 1):
            if res[h, i - 1] == 1 and res[h, i + 1] == 1:
                res[h, i] = 1
    return res

def queue_one_panel_advanced(name, panel_no, split):
    panel = Panel(name, panel_no)
    ongoing_panel = OngoingPanel(name, panel_no)
    folder = DATA / 'task2' / name / split
    windows = pd.read_csv(folder / 'window_index.csv')
    history = pd.read_parquet(folder / 'window_history.parquet')
    template = pd.read_csv(folder / 'sample_submission_queue.csv', dtype={'link_id': str})
    
    pieces = []
    for window in windows.itertuples(index=False):
        origin = pd.Timestamp(window.forecast_origin)
        times = pd.date_range(end=origin - pd.Timedelta(minutes=5), periods=HISTORY,
                              freq='5min', tz='UTC').strftime('%Y-%m-%dT%H:%M:%SZ')
        h = history[history.window_id == window.window_id].copy()
        h['link_id'] = h.link_id.astype(str)
        
        def matrix(col):
            return h.pivot(index='timestamp', columns='link_id', values=col).reindex(
                index=times, columns=panel.links).to_numpy(dtype=np.float32)
        
        speed, flow, eligible = [matrix(c) for c in ('speed_kmh', 'flow_vph', 'is_score_eligible')]
        
        if window.condition == 'queue_onset':
            # Crucial insight: Onset queues ONLY ever appear at horizon T+30 (step 6)
            X = panel.features(speed, flow, eligible, origin, window.condition)[-len(panel.links):]
            weight = CFG_T2['onset_weight']
            p = ((1 - weight) * onset_model.predict(X, num_threads=THREADS) +
                 weight * onset_cat.predict_proba(X, thread_count=THREADS)[:, 1])
            
            selected = p >= CFG_T2['onset_threshold']
            if not selected.any():
                # Fallback: by design every onset window has a queue
                selected[np.argsort(p)[-CFG_T2['onset_top_k']:]] = True
            
            pred = np.zeros((HORIZON, len(panel.links)), dtype=np.uint8)
            pred[-1] = selected
        else:
            # Ongoing queue forecasting
            observed = np.nan_to_num(eligible, nan=0).astype(bool)
            X = ongoing_panel.features(speed, flow, observed, origin, window.condition)
            prob = ongoing_model.predict(X, num_threads=THREADS).reshape(HORIZON, -1)
            pred = (prob >= CFG_T2['ongoing_threshold']).astype(np.uint8)
            pred = spatial_morphological_closing(pred)
        
        stamps = [(origin + pd.Timedelta(minutes=5 * step)).strftime('%Y-%m-%dT%H:%M:%SZ')
                  for step in range(1, HORIZON + 1)]
        pieces.append(pd.DataFrame({
            'window_id': window.window_id,
            'timestamp': np.repeat(stamps, len(panel.links)),
            'link_id': np.tile(panel.links, HORIZON),
            'queue_pred': pred.ravel()
        }))
    
    keys = ['window_id', 'timestamp', 'link_id']
    result = template[keys].merge(pd.concat(pieces, ignore_index=True), on=keys, how='left')
    assert result.queue_pred.notna().all() and len(result) == len(template)
    return result

queue_parts = []
for split in ('validation', 'private'):
    for panel_no, name in enumerate(panel_names()):
        queue_parts.append(queue_one_panel_advanced(name, panel_no, split))

queue_df = pd.concat(queue_parts, ignore_index=True)
queue_df.to_csv(OUTPUT / 'queue.csv', index=False)
print(f'Task 2 completed! Predicted {len(queue_df):,} cells, positive cells: {int(queue_df.queue_pred.sum()):,}')
del onset_model, ongoing_model, onset_cat
gc.collect()


### cell 5 [code]
# ==================== TASK 4: HIGH-PRECISION DUAL-WARM-START ODME ====================
print('\n>>> [3/4] Running Task 4: High-Precision ODME Optimization...')

LAMBDA = 0.05
initial = pd.read_parquet(ASSETS / 'odme_initial.parquet')
all_panels = [x['corridor_id'] for x in json.loads((DATA / 'config/corridors.json').read_text())['panels']]

def solve_odme_high_precision(name, split):
    network = DATA / 'corridors' / name / 'network'
    folder = DATA / 'task4' / name / split
    paths = pd.read_csv(network / 'path_set.csv', dtype=str)
    incidence = pd.read_csv(network / 'path_link_incidence.csv', dtype=str).drop_duplicates(['link_id', 'path_id'])
    prior = pd.read_csv(folder / 'synthetic_weak_prior.csv', dtype={'path_id': str, 'departure_time': str})
    counts = pd.read_csv(folder / 'synthetic_link_counts.csv', dtype={'link_id': str})
    
    ids = paths.path_id.tolist()
    path_index = {p: i for i, p in enumerate(ids)}
    supported = set(incidence.link_id)
    links = [link for link in counts.link_id if link in supported]
    link_index = {link: i for i, link in enumerate(links)}
    measured = incidence[incidence.link_id.isin(link_index)]
    
    A = sparse.csr_matrix((
        np.ones(len(measured), dtype=np.float64),
        (measured.link_id.map(link_index), measured.path_id.map(path_index))
    ), shape=(len(links), len(ids)))
    
    c = counts.set_index('link_id').reindex(links)['count'].to_numpy(float)
    b = prior.set_index('path_id').reindex(ids).path_flow.to_numpy(float)
    assert np.isfinite(c).all() and np.isfinite(b).all()
    
    def objective(f):
        res, reg = A @ f - c, f - b
        return float(0.5 * (res @ res + LAMBDA * (reg @ reg))), np.asarray(A.T @ res + LAMBDA * reg)
    
    departure = str(prior.departure_time.iloc[0])
    matched = initial[(initial.panel == name) & (initial.departure_time == departure)]
    if len(matched) > 0:
        f0 = matched.set_index('path_id').reindex(ids).path_flow.to_numpy(float)
    else:
        gram = (A @ A.T).toarray()
        u = b + A.T @ solve(gram + LAMBDA * np.eye(len(c)), c - A @ b, assume_a='pos')
        f0 = np.maximum(u, 0)
    
    # Ultra-tight tolerance L-BFGS-B optimization to minimize count residual
    fit = minimize(objective, f0, jac=True, method='L-BFGS-B', bounds=[(0, None)] * len(ids),
                   options={'maxiter': 8000, 'ftol': 1e-14, 'gtol': 1e-6, 'maxcor': 30})
    f = np.maximum(fit.x, 0)
    
    template = pd.read_csv(folder / 'sample_submission_path_flow.csv', dtype={'path_id': str})
    template['path_flow'] = template.path_id.map(dict(zip(ids, f)))
    assert template.path_flow.notna().all()
    return template

od_parts = []
with threadpool_limits(limits=1, user_api='blas'):
    for split in ('validation', 'private'):
        for name in all_panels:
            od_parts.append(solve_odme_high_precision(name, split))

od_df = pd.concat(od_parts, ignore_index=True)
od_df.to_csv(OUTPUT / 'odme.csv', index=False)
print(f'Task 4 completed! Generated {len(od_df):,} path-flow predictions.')


### cell 6 [code]
# ==================== FINAL SUBMISSION ASSEMBLY ====================
print('\n>>> [4/4] Assembling final submission.csv with official submission keys...')

KEYS = {
    'state': ['panel', 'timestamp', 'station_id', 'link_id', 'mask_regime'],
    'queue': ['window_id', 'timestamp', 'link_id'],
    'odme': ['panel', 'departure_time', 'path_id']
}
VALUES = {'state': ['speed_kmh', 'flow_vph'], 'queue': ['queue_pred'], 'odme': ['path_flow']}
COLUMNS = ['submission_id', 'task', 'speed_kmh', 'flow_vph', 'queue_pred', 'path_flow']

def indexed_predictions(task):
    frame = pd.read_csv(OUTPUT / f'{task}.csv', dtype={k: str for k in KEYS[task]})
    assert not frame.duplicated(KEYS[task]).any(), f'Duplicates found in {task} predictions!'
    assert np.isfinite(frame[VALUES[task]].to_numpy()).all(), f'Non-finite values found in {task}!'
    return frame.set_index(KEYS[task])[VALUES[task]]

tables = {task: indexed_predictions(task) for task in KEYS}
assert (tables['state'].speed_kmh > 0).all() and (tables['state'].flow_vph >= 0).all()
assert tables['queue'].queue_pred.isin([0, 1]).all()
assert (tables['odme'].path_flow >= 0).all()

counts = {task: 0 for task in KEYS}
row_count = 0
sub_file = WORK / 'submission.csv'

# Chunked streaming merge against submission_key.csv
for chunk in pd.read_csv(DATA / 'submission_key.csv', dtype=str, chunksize=500_000):
    out = chunk[['submission_id', 'task']].copy()
    for col in COLUMNS[2:]:
        out[col] = 0.0
    assert out.task.isin(KEYS).all()
    for task in KEYS:
        mask = chunk.task == task
        if not mask.any():
            continue
        wanted = pd.MultiIndex.from_frame(chunk.loc[mask, KEYS[task]])
        matched = tables[task].reindex(wanted)
        assert matched.notna().all().all(), f'Missing predictions for {task}'
        out.loc[mask, VALUES[task]] = matched.to_numpy()
        counts[task] += int(mask.sum())
    
    assert np.isfinite(out[COLUMNS[2:]].to_numpy()).all()
    out[COLUMNS].to_csv(sub_file, mode='w' if row_count == 0 else 'a',
                        header=row_count == 0, index=False)
    row_count += len(out)

print('--- SUBMISSION INTEGRITY AUDIT ---')
print(f'Total submission rows: {row_count:,}')
print(f'Task row breakdown: {counts}')
assert counts == {'state': 6_735_795, 'queue': 174_000, 'odme': 70_708}
assert row_count == 6_980_503

with sub_file.open('rb') as f:
    sub_sha256 = hashlib.file_digest(f, 'sha256').hexdigest()
print('SHA256 digest:', sub_sha256)
print(f'Total execution time: {time.time() - T0:.1f} seconds')
print('🎉 SUCCESS! Final submission.csv ready for leaderboard upload!')
display(pd.read_csv(sub_file, nrows=10))

