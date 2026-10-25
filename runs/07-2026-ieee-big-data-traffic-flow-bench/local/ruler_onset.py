"""9-month onset/ongoing ruler (official window-IoU metric).

Evaluates public onset (LGBM+CBM), v5-style onset boost (trained on first 8 months),
and public ongoing model over all candidate origins of all 9 train months.
Focus metric: mean IoU on the 3 latest months (2030-12, 2031-01, 2031-02) and overall.

Usage (env set): python local/ruler_onset.py [out_json]
"""
import os, sys, json, gc, time
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from catboost import CatBoostClassifier

ASSETS = Path(os.environ['TFB_ASSET_ROOT'])
sys.path.insert(0, str(ASSETS))
from config import RELEASE, THREADS
from queue_features import Panel, panel_names, HISTORY, HORIZON, candidate_origins
from queue_ongoing import Panel as OngoingPanel

ALL_MONTHS = ["2030-06", "2030-07", "2030-08", "2030-09", "2030-10", "2030-11",
              "2030-12", "2031-01", "2031-02"]
LATEST = {"2030-12", "2031-01", "2031-02"}
BOOST_SEEDS = (20270201, 20270202, 20270203)
BOOST_VARIANTS = ((15, 0.04), (31, 0.045), (63, 0.03))

def build_wins():
    """All candidate origins, all 9 months. Returns windows list + per-panel recurrence profiles."""
    windows = []
    rec_profile = {}
    for pi, name in enumerate(panel_names()):
        p = Panel(name, pi)
        L = len(p.links)
        files = sorted((RELEASE / 'corridors' / name / 'train' / 'mainline_states').glob('**/*.parquet'),
                       key=lambda f: f.parent.name)
        files = [f for f in files if f.parent.name.removeprefix('year_month=') in ALL_MONTHS]
        day_q, day_dow, file_off = [], [], []
        for file in files:
            times, v, q, e = p.train_day(file)
            n_days = len(times) // 288
            file_off.append(len(day_q))
            vm = v.reshape(n_days, 288, L)
            em = e.reshape(n_days, 288, L)
            dows = np.asarray(pd.to_datetime(times, utc=True).dayofweek, dtype=np.int64)
            for d in range(n_days):
                day_q.append(((vm[d] <= p.cut) & em[d] & np.isfinite(vm[d])).astype(np.uint8))
                day_dow.append(int(dows[288 * d]))
        day_q = np.stack(day_q); day_dow = np.asarray(day_dow)
        D = day_q.shape[0]
        dow_pos = np.empty(D, dtype=np.int32)
        dow_groups = {}
        seen = np.zeros(7, dtype=np.int32)
        for d in range(D):
            g = int(day_dow[d]); dow_pos[d] = seen[g]; dow_groups.setdefault(g, []).append(d); seen[g] += 1
        dows_cum = {g: np.vstack([np.zeros((1, 288, L), dtype=np.int32),
                                  np.cumsum(day_q[days], axis=0)]) for g, days in dow_groups.items()}
        g_cum = np.vstack([np.zeros((1, 288, L), dtype=np.int32), np.cumsum(day_q, axis=0)])
        prof = np.zeros((7, 288, L, 6), dtype=np.float32)
        last28 = day_q[max(0, D - 28):].mean(axis=0).astype(np.float32)
        for g in range(7):
            if g in dow_groups:
                m = day_q[dow_groups[g]].mean(axis=0)
                for o, col in enumerate((-1, 0, 1)):
                    prof[g, :, :, o] = np.roll(m, -col, axis=0)
                    prof[g, :, :, 3 + o] = np.roll(last28, -col, axis=0)
        rec_profile[name] = prof
        op = OngoingPanel(name, pi)
        for file, off in zip(files, file_off):
            month = file.parent.name.removeprefix('year_month=')
            times, v, q, e = p.train_day(file)
            d = off + np.arange(len(times)) // 288
            dow_arr = day_dow[d]; dowpos_arr = dow_pos[d]
            cand = candidate_origins(p, times, v, e)
            for t in cand['queue_onset']:
                y = (v[t + HORIZON] <= p.cut) & np.isfinite(v[t + HORIZON])
                elig = e[t + HORIZON] & np.isfinite(v[t + HORIZON])
                if not elig.any():
                    continue
                x = p.features(v[t-HISTORY:t], q[t-HISTORY:t], e[t-HISTORY:t],
                               pd.Timestamp(times[t]), 'queue_onset')[-len(p.links):]
                g = int(dow_arr[t]); k = int(dowpos_arr[t]); tod = t % 288
                cg = dows_cum[g]
                lo = max(0, d[t] - 28); n28 = d[t] - lo
                rec = np.zeros((L, 6), dtype=np.float32)
                for o, col in enumerate((-1, 0, 1)):
                    tt = (tod + o) % 288
                    if k > 0:
                        rec[elig, o] = cg[k, tt] / k
                    if n28 > 0:
                        rec[elig, 3 + o] = (g_cum[d[t], tt] - g_cum[lo, tt]) / n28
                xb = np.hstack([x, rec])
                windows.append(dict(panel=name, condition='queue_onset', month=month,
                                    X=xb, y=y, eligible=elig, Xpub=x))
            for t in cand['queue_ongoing']:
                y = (v[t+1:t+HORIZON+1] <= p.cut) & np.isfinite(v[t+1:t+HORIZON+1])
                elig = e[t+1:t+HORIZON+1] & np.isfinite(v[t+1:t+HORIZON+1])
                if not elig.any():
                    continue
                observed = np.nan_to_num(e[t-HISTORY:t], nan=0).astype(bool)
                x = op.features(v[t-HISTORY:t], q[t-HISTORY:t], observed,
                                pd.Timestamp(times[t]), 'queue_ongoing')[-len(op.links):]
                windows.append(dict(panel=name, condition='queue_ongoing', month=month,
                                    X=x, y=y, eligible=elig, Xpub=x))
            print(name, file.parent.name, 'windows so far', len(windows), flush=True)
        del day_q, g_cum, dows_cum
        gc.collect()
    return windows, rec_profile

def window_iou(true, pred, elig):
    true = true & elig
    pred = pred & elig
    union = np.logical_or(true, pred).sum()
    return float(np.logical_and(true, pred).sum() / max(union, 1))

def main():
    t0 = time.time()
    windows, rec_profile = build_wins()
    n_on = sum(w['condition'] == 'queue_onset' for w in windows)
    print(f"windows: {len(windows)} (onset {n_on}, ongoing {len(windows)-n_on}) in {time.time()-t0:.0f}s", flush=True)

    onset_lgb = lgb.Booster(model_file=str(ASSETS / 'onset_lgb.txt'))
    onset_cat = CatBoostClassifier(); onset_cat.load_model(str(ASSETS / 'onset.cbm'))
    ongoing_lgb = lgb.Booster(model_file=str(ASSETS / 'ongoing_lgb.txt'))

    # public onset probs
    for w in windows:
        if w['condition'] == 'queue_onset':
            X = w['Xpub']
            w['p_pub'] = 0.5 * onset_lgb.predict(X, num_threads=THREADS) + \
                         0.5 * onset_cat.predict_proba(X, thread_count=THREADS)[:, 1]
    # train boost on first 8 months
    Xtr = np.concatenate([w['X'][w['eligible']] for w in windows
                          if w['condition'] == 'queue_onset' and w['month'] != '2031-02'])
    ytr = np.concatenate([w['y'][w['eligible']] for w in windows
                          if w['condition'] == 'queue_onset' and w['month'] != '2031-02'])
    print('boost train rows', Xtr.shape, 'pos', float(ytr.mean()), flush=True)
    boost_models = []
    for seed, (leaves, lr) in zip(BOOST_SEEDS, BOOST_VARIANTS):
        params = {'objective': 'binary', 'metric': 'binary_logloss', 'learning_rate': lr,
                  'num_leaves': leaves, 'min_data_in_leaf': 100, 'lambda_l2': 10,
                  'feature_fraction': .9, 'bagging_fraction': .9, 'bagging_freq': 1,
                  'num_threads': THREADS, 'verbosity': -1, 'seed': seed}
        m = lgb.train(params, lgb.Dataset(Xtr, label=ytr), num_boost_round=1500)
        boost_models.append(m)
        del m; gc.collect()
    for w in windows:
        if w['condition'] == 'queue_onset':
            w['p_boost'] = np.mean([m.predict(w['X'], num_threads=THREADS) for m in boost_models], axis=0)
    # ongoing probs
    for w in windows:
        if w['condition'] == 'queue_ongoing':
            w['p_ong'] = ongoing_lgb.predict(w['X'], num_threads=THREADS).reshape(HORIZON, -1)

    def eval_onset(w_blend, thr):
        vals_all, vals_latest = [], []
        for w in windows:
            if w['condition'] != 'queue_onset':
                continue
            p = w['p_pub'] if w_blend == 0.0 else (1 - w_blend) * w['p_pub'] + w_blend * w['p_boost']
            pred = (p >= thr)
            if not pred.any():
                pred[np.argmax(p)] = True
            predh = np.zeros((HORIZON, w['y'].shape[0]), dtype=bool)
            predh[-1] = pred
            i = window_iou(w['y'], predh, w['eligible'])
            vals_all.append(i)
            if w['month'] in LATEST:
                vals_latest.append(i)
        return (float(np.mean(vals_all)) if vals_all else 0.0,
                float(np.mean(vals_latest)) if vals_latest else 0.0)

    def eval_ongoing(thr):
        vals_all, vals_latest = [], []
        for w in windows:
            if w['condition'] != 'queue_ongoing':
                continue
            pred = w['p_ong'] >= thr
            pred = np.asarray(pred).reshape(HORIZON, -1)
            i = window_iou(w['y'], pred, w['eligible'])
            vals_all.append(i)
            if w['month'] in LATEST:
                vals_latest.append(i)
        return (float(np.mean(vals_all)) if vals_all else 0.0,
                float(np.mean(vals_latest)) if vals_latest else 0.0)

    rows = []
    for wb in (0.0, 0.3, 0.5, 0.7):
        for thr in [round(0.30 + 0.05 * i, 2) for i in range(9)]:
            oa, ol = eval_onset(wb, thr)
            rows.append(dict(kind='onset', blend=wb, thr=thr, iou_all=oa, iou_latest=ol))
    for thr in [round(0.30 + 0.025 * i, 3) for i in range(13)]:
        oa, ol = eval_ongoing(thr)
        rows.append(dict(kind='ongoing', thr=thr, iou_all=oa, iou_latest=ol))
    out = Path(sys.argv[1] if len(sys.argv) > 1 else 'local/ruler_results.json')
    out.write_text(json.dumps(rows, indent=1))
    df = pd.DataFrame(rows)
    for kind in ('onset', 'ongoing'):
        d = df[df.kind == kind].sort_values('iou_latest', ascending=False)
        print(f"=== {kind} top by latest-3-month IoU ===")
        print(d.head(8).to_string(index=False))
    print('saved', out)

if __name__ == '__main__':
    main()
