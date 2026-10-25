# ================= TASK 1 EXTRA (v3): LODO weekday x tod profiles, LGBM x3, U-Net x2 =================
USE_T1_EXTRA = True
T1_TREES_SHARE = 0.70
T1_PUB_SHARE = 0.50
LGBM_VARIANTS = ((20280101, 127, 0.04), (20280102, 63, 0.05), (20280103, 255, 0.03))
UNET_SEEDS = (20280201, 20280202)
UNET_WIDTH = 48
UNET_EPOCHS = 2
PAD_MIN = 48
SPEED_SCALE = 25.0
FLOW_SCALE = 600.0
STATE_BASE_FILE = 'state_v1.csv'
T1_EXTRA_OK = False

def _fill(v, fallback):
    return np.where(np.isfinite(v), v, fallback)

try:
    import prepare_state as pstate
    import torch
    import torch.nn as nn
    import torch.nn.functional as F

    lanes_of = lambda p: np.maximum(p.lanes, 1.0)

    class PanelProfiles:
        """Weekday x tod profiles from unmasked train days, with leave-one-day-out variants."""
        def __init__(self, name):
            p = base.Panel(name, 0)
            self.panel = p
            self.lanes = lanes_of(p)
            L = len(p.links)
            sumv = np.zeros((7, 288, L), dtype=np.float64)
            sumq = np.zeros((7, 288, L), dtype=np.float64)
            cnt = np.zeros((7, 288, L), dtype=np.int64)
            self.dow = []
            self.day_v = []
            self.day_q = []
            for month in base.TRAIN_MONTHS:
                times, v, q, e = base.matrices(base.month_frame(name, month), p)
                assert len(times) % 288 == 0, (name, month, len(times))
                n_days = len(times) // 288
                dow = np.asarray(pd.to_datetime(times, utc=True).dayofweek, dtype=np.int64)
                vd = v.reshape(n_days, 288, L)
                qd = (q / self.lanes[None, None, :]).reshape(n_days, 288, L)
                ed = e.reshape(n_days, 288, L)
                for d in range(n_days):
                    dd = int(dow[288 * d])
                    assert dow[288 * d] == dow[288 * d + 287], (name, month, d)
                    self.dow.append(dd)
                    self.day_v.append(vd[d])
                    self.day_q.append(qd[d])
                    fin = ed[d] & np.isfinite(vd[d]) & np.isfinite(qd[d])
                    sumv[dd] += np.where(fin, vd[d], 0.0)
                    sumq[dd] += np.where(fin, qd[d], 0.0)
                    cnt[dd] += fin
            self.sumv = sumv
            self.sumq = sumq
            self.cnt = cnt
            infv = sumv / np.maximum(cnt, 1)
            self.prof_in = np.where(np.isfinite(infv), infv, p.vf[None, None, :])
            infq = sumq / np.maximum(cnt, 1)
            self.prof_q_in = np.where(np.isfinite(infq), infq, 0.6 * (p.cap / self.lanes)[None, None, :])
            self.n_days = len(self.dow)
            print('profiles', name, 'days:', self.n_days, flush=True)

        def lodo(self, d):
            dd = self.dow[d]
            vv = self.day_v[d]
            qq = self.day_q[d]
            fin = np.isfinite(vv) & np.isfinite(qq)
            c = self.cnt[dd]
            den = np.maximum(c - fin.astype(np.int64), 1)
            pv = (self.sumv[dd] - np.where(fin, vv, 0.0)) / den
            pq = (self.sumq[dd] - np.where(fin, qq, 0.0)) / den
            return (np.where(np.isfinite(pv), pv, self.panel.vf[None, None, :]),
                    np.where(np.isfinite(pq), pq, 0.6 * (self.panel.cap / self.lanes)[None, None, :]))

        def month_profile(self, times):
            """(n_rows, L) inference profile for a non-train month."""
            n_rows = len(times)
            day_of = np.arange(n_rows) // 288
            tod = np.arange(n_rows) % 288
            dow = np.asarray(pd.to_datetime(times, utc=True).dayofweek, dtype=np.int64)
            pv = self.prof_in[dow[day_of], tod, :]
            pq = self.prof_q_in[dow[day_of], tod, :]
            return pv, pq

    def ext_X(p, pi, times, v, q, e, rows, cols, regime, pv_row, pq_row):
        X, bv, bq = pstate.features(p, pi, times, v, q, e, rows, cols, regime)
        extra = np.column_stack([pv_row, pq_row,
                                 bv / np.maximum(pv_row, 1e-3),
                                 bq / np.maximum(pq_row, 1e-3)])
        return np.column_stack([X, extra]).astype(np.float32), bv, bq

    # ---- extended feature sets (mirrors prepare_state) ----
    prof_cache = {}
    xx, yv, yq, xv, yvv, yqv = [], [], [], [], [], []
    original_seed = base.SEED
    for pi, name in enumerate(base.all_panels()):
        if pi != 0:
            continue
        p = base.Panel(name, pi)
        prof = PanelProfiles(name)
        prof_cache[name] = prof
        n_days = prof.n_days
        day_dow = np.asarray(prof.dow, dtype=np.int64)
        for month in (base.TRAIN_MONTHS[0], base.VALID_MONTH):
            times, v, q, e = base.matrices(base.month_frame(name, month), p)
            isval = month == base.VALID_MONTH
            for rep in range(1 if isval else 2):
                base.SEED = original_seed + rep * 7919
                mask, regime = base.simulated_mask(name, times, v, q, e)
                rows, cols = base.sampled_targets(mask, times, 1500 if isval else 750)
                v_masked = np.where(mask, np.nan, v)
                q_masked = np.where(mask, np.nan, q)
                day_of = rows // 288
                tod = rows % 288
                if isval:
                    pv_full, pq_full = prof_cache[name].month_profile(times)
                    pv_row = pv_full[rows, cols]
                    pq_row = pq_full[rows, cols]
                    del pv_full, pq_full
                else:
                    pv_full = np.empty((n_days, 288, len(p.links)), dtype=np.float32)
                    pq_full = np.empty_like(pv_full)
                    for d in np.unique(day_of):
                        a, b = prof.lodo(int(d))
                        pv_full[d] = a
                        pq_full[d] = b
                    pv_row = pv_full[day_of, tod, cols]
                    pq_row = pq_full[day_of, tod, cols]
                    del pv_full, pq_full
                X, bv, bq = ext_X(p, pi, times, v_masked, q_masked, e, rows, cols, regime, pv_row, pq_row)
                rv = v[rows, cols] - bv
                rq = q[rows, cols] / p.lanes[cols] - bq
                if isval:
                    xv.append(X); yvv.append(rv); yqv.append(rq)
                else:
                    xx.append(X); yv.append(rv); yq.append(rq)
                del X, rv, rq, v_masked, q_masked
            gc.collect()
            print('ext features', name, month, flush=True)
    base.SEED = original_seed
    Xtr = np.concatenate(xx); del xx
    yvtr = np.concatenate(yv); yqtr = np.concatenate(yq); del yv, yq
    Xva = np.concatenate(xv); yvv = np.concatenate(yvv); yqv = np.concatenate(yqv); del xv, yvv, yqv
    print('ext features ready', Xtr.shape, Xva.shape, flush=True)

    # ---- 3 LGBM variants x 2 channels ----
    lgbm_models = {'speed': [], 'flow': []}
    valid_sets = {'speed': lgb.Dataset(Xva, label=yvv), 'flow': lgb.Dataset(Xva, label=yqv)}
    for seed, leaves, lr in LGBM_VARIANTS[:1]:
        for channel, ytr in (('speed', yvtr),):
            params = {'objective': 'regression_l2', 'metric': 'rmse', 'learning_rate': lr,
                      'num_leaves': leaves, 'min_data_in_leaf': 200, 'feature_fraction': 0.9,
                      'bagging_fraction': 0.9, 'bagging_freq': 1, 'lambda_l2': 10.0,
                      'num_threads': THREADS, 'verbosity': -1, 'seed': seed}
            m = lgb.train(params, lgb.Dataset(Xtr, label=ytr), num_boost_round=150,
                          valid_sets=[valid_sets[channel]],
                          callbacks=[lgb.early_stopping(150), lgb.log_evaluation(500)])
            lgbm_models[channel].append(m)
            print('lgbm', channel, seed, 'best_iter:', m.best_iteration, flush=True)
    del Xtr, yvtr, yqtr, Xva, yvv, yqv, valid_sets
    gc.collect()

    # ---- U-Net day grids for train + validation months ----
    class UNet(nn.Module):
        def __init__(self, c_in=9, w=48):
            super().__init__()
            block = lambda a, b: nn.Sequential(nn.Conv2d(a, b, 3, padding=1), nn.GroupNorm(8, b), nn.GELU())
            self.e1 = block(c_in, w); self.e2 = block(w, 2 * w); self.e3 = block(2 * w, 4 * w)
            self.mid = block(4 * w, 4 * w)
            self.u2 = block(8 * w, 2 * w); self.u1 = block(4 * w, w)
            self.head = nn.Conv2d(2 * w, 2, 1)
            self.pool = nn.MaxPool2d(2)
        def forward(self, x):
            a1 = self.e1(x)
            a2 = self.e2(self.pool(a1))
            a3 = self.e3(self.pool(a2))
            b = self.mid(a3)
            b = self.u2(torch.cat([F.interpolate(b, size=a3.shape[2:], mode='nearest'), a3], 1))
            b = self.u1(torch.cat([F.interpolate(b, size=a2.shape[2:], mode='nearest'), a2], 1))
            b = torch.cat([F.interpolate(b, size=a1.shape[2:], mode='nearest'), a1], 1)
            return self.head(b)

    max_links = max(len(prof_cache[n].panel.links) for n in base.all_panels())
    pad = int(((max(PAD_MIN, max_links) + 7) // 8) * 8)
    print('pad links:', pad, 'max_links:', max_links, flush=True)

    unet_data = []
    base.SEED = original_seed
    for name in [base.all_panels()[0]]:
        p = prof_cache[name].panel
        lanes = lanes_of(p)
        static = np.zeros((4, 288, pad), dtype=np.float16)
        static[0] = (p.vf / 100.0).astype(np.float16)
        static[1] = (p.cap / lanes / 100.0).astype(np.float16)
        static[2] = (p.cut / 100.0).astype(np.float16)
        static[3, :, :len(p.links)] = 1.0
        frames = {}
        for month in (*base.TRAIN_MONTHS, base.VALID_MONTH):
            frames[month] = base.matrices(base.month_frame(name, month), p)
        days = []
        for month in (*base.TRAIN_MONTHS, base.VALID_MONTH):
            times = frames[month][0]
            n_days = len(times) // 288
            for d in range(n_days):
                days.append((month, d))
        dyn = np.zeros((len(days), 5, 288, pad), dtype=np.float16)
        tidx, tdv, tdq = [], [], []
        tday_off = [0]
        for k, (month, d) in enumerate(days):
            times, v, q, e = frames[month]
            d0 = 288 * d
            vd = v[d0:d0 + 288]; qd = q[d0:d0 + 288]; ed = e[d0:d0 + 288]
            if month == base.VALID_MONTH:
                pv_d = prof_cache[name].prof_in[int(np.asarray(pd.to_datetime(times, utc=True).dayofweek)[d0])]
                pq_d = prof_cache[name].prof_q_in[int(np.asarray(pd.to_datetime(times, utc=True).dayofweek)[d0])]
            else:
                pv_d, pq_d = prof_cache[name].lodo(d)
            fin_v = np.isfinite(vd)
            fin_q = np.isfinite(qd)
            dyn[k, 0] = _fill(vd, pv_d).astype(np.float16)
            dyn[k, 1] = _fill(qd / lanes[None, :], pq_d).astype(np.float16)
            dyn[k, 2] = pv_d.astype(np.float16)
            dyn[k, 3] = pq_d.astype(np.float16)
            dyn[k, 4] = (fin_v & fin_q & ed).astype(np.float16)
            mask, _reg = base.simulated_mask(name, times, v, q, e)
            target = mask[d0:d0 + 288] & ed & fin_v & fin_q
            if target.any():
                ti = np.argwhere(target)
                tidx.append(ti.astype(np.int16))
                tdv.append(((vd - pv_d) / SPEED_SCALE)[target].astype(np.float16))
                tdq.append(((qd / lanes[None, :] - pq_d) / FLOW_SCALE)[target].astype(np.float16))
                tday_off.append(tday_off[-1] + len(ti))
        unet_data.append({'dyn': dyn, 'static': static,
                          'tidx': np.concatenate(tidx) if tidx else np.zeros((0, 2), dtype=np.int16),
                          'tdv': np.concatenate(tdv) if tdv else np.zeros(0, dtype=np.float16),
                          'tdq': np.concatenate(tdq) if tdq else np.zeros(0, dtype=np.float16),
                          'off': np.asarray(tday_off, dtype=np.int32),
                          'val_days': [k for k, (month, _d) in enumerate(days) if month == base.VALID_MONTH],
                          'n_days': len(days)})
        del frames
        gc.collect()
        print('unet data', name, len(days), flush=True)
    base.SEED = original_seed

    flat = []
    val_items = []
    for pi_d, data in enumerate(unet_data):
        for d in range(data['n_days']):
            if d in data['val_days']:
                val_items.append((pi_d, d))
            else:
                flat.append((pi_d, d))
    val_days_set = set(val_items)

    def batch_loss(model, items):
        x = torch.empty((len(items), 9, 288, pad), dtype=torch.float32)
        for i, (pi_d, d) in enumerate(items):
            data = unet_data[pi_d]
            x[i, :5] = torch.from_numpy(data['dyn'][d])
            x[i, 5:] = torch.from_numpy(data['static'])
        out = model(x)
        loss = torch.zeros(())
        for i, (pi_d, d) in enumerate(items):
            data = unet_data[pi_d]
            a, b = data['off'][d], data['off'][d + 1]
            if b > a:
                ti = data['tidx'][a:b]
                loss = loss + ((out[i, 0, ti[:, 0], ti[:, 1]] - torch.from_numpy(data['tdv'][a:b].astype(np.float32))).pow(2).mean()
                               + (out[i, 1, ti[:, 0], ti[:, 1]] - torch.from_numpy(data['tdq'][a:b].astype(np.float32))).pow(2).mean())
        return loss / max(1, len(items))

    unets = []
    for seed in UNET_SEEDS:
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        model = UNet(9, UNET_WIDTH).float()
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        best = None
        hist = []
        for epoch in range(UNET_EPOCHS):
            model.train()
            idx = rng.permutation(len(flat))
            order = [flat[j] for j in idx]
            tot, nbatch = 0.0, 0
            for k in range(0, len(order), 4):
                items = order[k:k + 4]
                loss = batch_loss(model, items)
                opt.zero_grad()
                loss.backward()
                opt.step()
                tot += float(loss)
                nbatch += 1
            model.eval()
            with torch.no_grad():
                val = float(sum(batch_loss(model, val_items[k:k + 4]).item() for k in range(0, len(val_items), 4)) / max(1, len(val_items) // 4 + 1))
            hist.append(val)
            print(f'unet seed {seed} epoch {epoch} train {tot / max(1, nbatch):.5f} val {val:.5f}', flush=True)
            if best is None or val < best[0] - 1e-5:
                best = (val, {kk: vv.detach().clone() for kk, vv in model.state_dict().items()})
            if len(hist) > 12 and val >= min(hist[-12:]):
                break
        model.load_state_dict(best[1])
        model.eval()
        unets.append(model)
        gc.collect()
        print('unet trained', seed, 'best val', best[0], flush=True)

    # ---- inference on task1 target rows: LGBM + U-Net ----
    pieces = []
    for split in ('validation', 'private'):
        for pi, name in enumerate(base.all_panels()):
            if pi != 0:
                continue
            p = prof_cache[name].panel
            lanes = lanes_of(p)
            times, v, q, e = base.matrices(base.month_frame(name, '', split=split), p)
            target = pd.read_csv(DATA / 'task1' / name / split / 'sample_submission_state.csv',
                                 dtype={'station_id': str, 'link_id': str})
            rows = times.get_indexer(target.timestamp)
            cols = target.link_id.map({l: i for i, l in enumerate(p.links)}).to_numpy(dtype=int)
            regime = target.mask_regime.map({'R1': 0, 'R2': 1, 'R3': 2}).to_numpy()
            assert (rows >= 0).all() and (cols >= 0).all()
            pv_full, pq_full = prof_cache[name].month_profile(times)
            X, bv, bq = ext_X(p, pi, times, v, q, e, rows, cols, regime,
                              pv_full[rows, cols], pq_full[rows, cols])
            sp = np.mean([m.predict(X, num_threads=THREADS) for m in lgbm_models['speed']], axis=0)
            fp = np.mean([m.predict(X, num_threads=THREADS) for m in lgbm_models['flow']], axis=0)
            n_days = len(times) // 288
            dow = np.asarray(pd.to_datetime(times, utc=True).dayofweek, dtype=np.int64)
            din = np.zeros((n_days, 5, 288, pad), dtype=np.float16)
            for d in range(n_days):
                d0 = 288 * d
                vd = v[d0:d0 + 288]; qd = q[d0:d0 + 288]; ed = e[d0:d0 + 288]
                pv_d = prof_cache[name].prof_in[dow[d0]]
                pq_d = prof_cache[name].prof_q_in[dow[d0]]
                din[d, 0] = _fill(vd, pv_d).astype(np.float16)
                din[d, 1] = _fill(qd / lanes[None, :], pq_d).astype(np.float16)
                din[d, 2] = pv_d.astype(np.float16)
                din[d, 3] = pq_d.astype(np.float16)
                din[d, 4] = (np.isfinite(vd) & np.isfinite(qd) & ed).astype(np.float16)
            static = np.zeros((4, 288, pad), dtype=np.float16)
            static[0] = (p.vf / 100.0).astype(np.float16)
            static[1] = (p.cap / lanes / 100.0).astype(np.float16)
            static[2] = (p.cut / 100.0).astype(np.float16)
            static[3, :, :len(p.links)] = 1.0
            outs = []
            for m in unets:
                chunk = []
                with torch.no_grad():
                    for d in range(0, n_days, 4):
                        days_b = list(range(d, min(d + 4, n_days)))
                        x = torch.empty((len(days_b), 9, 288, pad), dtype=torch.float32)
                        for j, dd in enumerate(days_b):
                            x[j, :5] = torch.from_numpy(din[dd])
                            x[j, 5:] = torch.from_numpy(static)
                        chunk.append(m(x)[1].numpy())
                outs.append(np.concatenate(chunk, axis=0))
            dv = np.mean(outs, axis=0)
            day_of = rows // 288
            tod = rows % 288
            unet_speed = pv_full[rows, cols] + SPEED_SCALE * dv[0, day_of, tod, cols]
            unet_flow = (pq_full[rows, cols] + FLOW_SCALE * dv[1, day_of, tod, cols]) * lanes[cols]
            pieces.append(pd.DataFrame({
                'panel': name, 'timestamp': target.timestamp, 'link_id': target.link_id,
                'speed_kmh': np.clip(bv + sp, 1, 180),
                'flow_vph': np.clip((bq + fp) * lanes[cols], 0, 150000),
                'unet_speed': np.clip(unet_speed, 1, 180),
                'unet_flow': np.clip(unet_flow, 0, 150000)}))
            del X, sp, fp, din, outs, dv
            gc.collect()
            print('t1extra infer', name, split, flush=True)
    extra_state = pd.concat(pieces, ignore_index=True)
    extra_state.to_csv(OUTPUT / 'state_t1extra.csv', index=False)
    print('t1extra state written:', len(extra_state), flush=True)

    # ---- blend: 50% v1 CatBoost + 50% (70% LGBM + 30% U-Net) ----
    v1s = pd.read_csv(OUTPUT / 'state_v1.csv', dtype={'station_id': str, 'link_id': str})
    assert len(v1s) == len(extra_state)
    assert (v1s.panel.values == extra_state.panel.values).all()
    assert (v1s.timestamp.values == extra_state.timestamp.values).all()
    assert (v1s.link_id.values == extra_state.link_id.values).all()
    blend = v1s.copy()
    for col, lcol, ucol in (('speed_kmh', 'speed_kmh', 'unet_speed'), ('flow_vph', 'flow_vph', 'unet_flow')):
        mine = T1_TREES_SHARE * extra_state[lcol].to_numpy() + (1.0 - T1_TREES_SHARE) * extra_state[ucol].to_numpy()
        blend[col] = T1_PUB_SHARE * v1s[col].to_numpy() + (1.0 - T1_PUB_SHARE) * mine
    assert np.isfinite(blend[['speed_kmh', 'flow_vph']].to_numpy()).all()
    blend.to_csv(OUTPUT / 'state_blend.csv', index=False)
    STATE_BASE_FILE = 'state_blend.csv'
    T1_EXTRA_OK = True
    print('state blend written', flush=True)
except Exception as err:
    import traceback
    traceback.print_exc()
    print('T1 EXTRA FAILED, falling back to v1 state:', repr(err), flush=True)
    T1_EXTRA_OK = False


FD_ALPHA = 0.15
CONS_ALPHA = 0.40
CONS_PASSES = 5

def fd_params(name):
    fd = pd.read_csv(DATA / 'corridors' / name / 'network' / 'fd_parameters.csv', dtype={'link_id': str})
    have = [c for c in ('free_speed_kmh', 'capacity_vph', 'lanes', 'k_jam') if c in fd.columns]
    m = fd.groupby('link_id', as_index=False)[have].mean()
    m['lanes'] = m.lanes.fillna(1.0).clip(lower=1.0)
    m['critical_density'] = m.capacity_vph / m.free_speed_kmh.clip(lower=1.0)
    m['k_jam'] = m.k_jam.fillna(100.0 * m.lanes) if 'k_jam' in m.columns else 100.0 * m.lanes
    m['k_jam'] = np.maximum(m.k_jam, m.critical_density * 1.05)
    m['w'] = m.capacity_vph / (m.k_jam - m.critical_density)
    return m.set_index('link_id')

def physics_pass(name, split):
    panel = base.Panel(name, list(base.all_panels()).index(name))
    frame = base.month_frame(name, '', split=split)
    times, v, q, e = base.matrices(frame, panel)
    state = pd.read_csv(OUTPUT / STATE_BASE_FILE, dtype={'station_id': str, 'link_id': str},
                        usecols=['panel', 'timestamp', 'link_id', 'speed_kmh', 'flow_vph'])
    state = state[state.panel == name]
    state = state[state.timestamp.isin(times)]
    li = {l: i for i, l in enumerate(panel.links)}
    rows = times.get_indexer(state.timestamp)
    cols = state.link_id.map(li).to_numpy(dtype=int)
    v[rows, cols] = state.speed_kmh.to_numpy(float)
    q[rows, cols] = state.flow_vph.to_numpy(float)
    target = pd.read_csv(DATA / 'task1' / name / split / 'sample_submission_state.csv',
                         dtype={'station_id': str, 'link_id': str})
    trows = times.get_indexer(target.timestamp)
    tcols = target.link_id.map(li).to_numpy(dtype=int)
    fd = fd_params(name)
    vf = panel.vf.astype(float); cap = panel.cap.astype(float)
    k_crit = (cap / np.maximum(vf, 1.0)).astype(float)
    kj = fd.reindex(panel.links).k_jam.to_numpy(float)
    w = fd.reindex(panel.links).w.to_numpy(float)
    def soft_fd(V, Q):
        out = Q.copy()
        valid = np.isfinite(V) & np.isfinite(Q) & (V > 1.0)
        k = np.where(valid, Q / np.maximum(V, 1.0), 0.0)
        kc = k_crit[None, :]
        q_fd = np.where(k <= kc, vf[None, :] * k, w[None, :] * np.maximum(kj[None, :] - k, 0.0))
        reasonable = valid & (q_fd > 0) & (q_fd < cap[None, :] * 1.3)
        out[reasonable] = (1.0 - FD_ALPHA) * Q[reasonable] + FD_ALPHA * q_fd[reasonable]
        return out
    def conservation(Q):
        out = Q.copy()
        order = np.argsort(panel.milepost)
        pos = {int(i): r for r, i in enumerate(order)}
        links_sorted = [panel.links[int(i)] for i in order]
        idx_of = {l: i for i, l in enumerate(panel.links)}
        rank = np.array([pos[i] for i in range(len(panel.links))])
        for _pass in range(CONS_PASSES):
            for t in range(len(times)):
                fl = out[t]
                for r in range(len(order) - 1):
                    up = int(order[r]); dn = int(order[r + 1])
                    if rank[dn] != rank[up] + 1:
                        continue
                    q_up = fl[up]; q_dn = fl[dn]
                    if not (np.isfinite(q_up) and np.isfinite(q_dn)):
                        continue
                    residual = q_dn - q_up
                    correction = CONS_ALPHA * residual * 0.5
                    fl[up] = q_up + correction
                    fl[dn] = q_dn - correction
        return out
    q2 = soft_fd(v, q)
    q2 = conservation(q2)
    target['speed_kmh'] = v[trows, tcols]
    target['flow_vph'] = np.clip(q2[trows, tcols], 0, 150000)
    assert np.isfinite(target[['speed_kmh', 'flow_vph']].to_numpy()).all()
    return target[['panel', 'timestamp', 'station_id', 'link_id', 'mask_regime', 'speed_kmh', 'flow_vph']]

first = True
for split in ('validation', 'private'):
    for name in base.all_panels():
        part = physics_pass(name, split)
        part.to_csv(OUTPUT / 'state.csv', index=False, mode='w' if first else 'a', header=first)
        first = False
        print('physics', name, split, len(part), flush=True)
print('state final written', flush=True)

