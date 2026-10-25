# ---- onset boost (v5): all-candidate origins + same-dow / 28d recurrence features ----
ONSET_BOOST = True
ONSET_BOOST_OK = False
BOOST_SEEDS = (20270201, 20270202, 20270203)
BOOST_VARIANTS = ((15, 0.04), (31, 0.045), (63, 0.03))
onset_boost_models = []
boost_rec_profile = {}
if ONSET_BOOST:
    try:
        boost_Xtr, boost_ytr, boost_Xva, boost_yva = [], [], [], []
        for pi, name in enumerate(panel_names()):
            p = Panel(name, pi)
            L = len(p.links)
            files = sorted((DATA / 'corridors' / name / 'train' / 'mainline_states').glob('**/*.parquet'),
                           key=lambda f: f.parent.name)
            day_q, day_dow, file_off = [], [], []
            for file in files:
                times, v, q, e = p.train_day(file)
                n_days = len(times) // 288
                file_off.append(len(day_q))
                vm = v.reshape(n_days, 288, L)
                em = e.reshape(n_days, 288, L)
                dows = np.asarray(pd.to_datetime(times, utc=True).dayofweek, dtype=np.int64)
                for d in range(n_days):
                    assert dows[288 * d] == dows[288 * d + 287], (name, file, d)
                    day_q.append(((vm[d] <= p.cut) & em[d] & np.isfinite(vm[d])).astype(np.uint8))
                    day_dow.append(int(dows[288 * d]))
            day_q = np.stack(day_q); day_dow = np.asarray(day_dow)
            D = day_q.shape[0]
            dow_pos = np.empty(D, dtype=np.int32)
            dow_groups = {}
            seen = np.zeros(7, dtype=np.int32)
            for d in range(D):
                g = int(day_dow[d])
                dow_pos[d] = seen[g]
                dow_groups.setdefault(g, []).append(d)
                seen[g] += 1
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
            boost_rec_profile[name] = prof
            for file, off in zip(files, file_off):
                times, v, q, e = p.train_day(file)
                isval = file.parent.name.endswith('2031-02')
                cand = candidate_origins(p, times, v, e)['queue_onset']
                ntr = nva = 0
                for t in cand:
                    d = off + t // 288; tod = t % 288
                    g = int(day_dow[d]); k = int(dow_pos[d])
                    y = (v[t + HORIZON] <= p.cut) & np.isfinite(v[t + HORIZON])
                    elig = e[t + HORIZON] & np.isfinite(v[t + HORIZON])
                    if not elig.any():
                        continue
                    x = p.features(v[t-HISTORY:t], q[t-HISTORY:t], e[t-HISTORY:t], pd.Timestamp(times[t]), 'queue_onset')[-len(p.links):]
                    rec = np.zeros((L, 6), dtype=np.float32)
                    cg = dows_cum[g]
                    for o, col in enumerate((-1, 0, 1)):
                        tt = (tod + o) % 288
                        if k > 0:
                            rec[elig, o] = cg[k, tt] / k
                        lo = max(0, d - 28); n28 = d - lo
                        if n28 > 0:
                            rec[elig, 3 + o] = (g_cum[d, tt] - g_cum[lo, tt]) / n28
                    xb = np.hstack([x, rec])
                    (boost_Xva if isval else boost_Xtr).append(xb[elig])
                    (boost_yva if isval else boost_ytr).append(y[elig])
                    if isval: nva += int(elig.sum())
                    else: ntr += int(elig.sum())
                print('onset boost', name, file.parent.name, 'train', ntr, 'valid', nva, flush=True)
            del day_q, g_cum, dows_cum
            gc.collect()
        X = np.concatenate(boost_Xtr); y = np.concatenate(boost_ytr)
        Xv = np.concatenate(boost_Xva); yv = np.concatenate(boost_yva)
        print('onset boost ready to train', X.shape, Xv.shape, float(y.mean()), float(yv.mean()), flush=True)
        for seed, (leaves, lr) in zip(BOOST_SEEDS, BOOST_VARIANTS):
            params = {'objective': 'binary', 'metric': 'binary_logloss', 'learning_rate': lr, 'num_leaves': leaves,
                      'min_data_in_leaf': 100, 'lambda_l2': 10, 'feature_fraction': .9, 'bagging_fraction': .9,
                      'bagging_freq': 1, 'num_threads': THREADS, 'verbosity': -1, 'seed': seed}
            m = lgb.train(params, lgb.Dataset(X, label=y), num_boost_round=900,
                          valid_sets=[lgb.Dataset(Xv, label=yv)], callbacks=[lgb.early_stopping(70), lgb.log_evaluation(500)])
            m = lgb.train(params, lgb.Dataset(X, label=y), num_boost_round=1500, init_model=m,
                          valid_sets=[lgb.Dataset(Xv, label=yv)], callbacks=[lgb.early_stopping(70), lgb.log_evaluation(500)])
            onset_boost_models.append(m)
            del m
            gc.collect()
            print('onset boost member trained', seed, leaves, lr, flush=True)
        del X, y, Xv, yv
        gc.collect()
        ONSET_BOOST_OK = True
        print('ONSET BOOST OK:', len(onset_boost_models), 'members', flush=True)
    except Exception:
        import traceback
        traceback.print_exc()
        print('ONSET BOOST FAILED -> public onset only', flush=True)
