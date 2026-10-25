"""v2 builder: rank-based question prior + a * model -> top-3 (MAP@3).
Protocols:
  A (optimistic): LOO self-excluded prior + saved model oof
  B (clean qsplit): 5-fold over questions; knn rebuilt on in-fold rows, prior from in-fold rows
Model npz: te (3498,74), oof (1500,74) optional, codes (74,)
"""
import argparse, collections, os, sys
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metric import map3_from_topk, top3_from_scores

def rank_vec(counter, ci, topk=3):
    v = np.zeros(len(ci))
    for r, (c, n) in enumerate(counter.most_common(topk), 1):
        if c in ci:
            v[ci[c]] = 1.0 / r
    return v

def _norm(cnt, C):
    vv = np.zeros(C)
    t = sum(cnt.values())
    if t == 0:
        return vv
    for c2, n in cnt.items():
        if c2 in vv_idx:
            vv[vv_idx[c2]] = n / t
    return vv

def priors_from(tr_in, tr_out, ci):
    """prob-based prior: question dist if question has rows, else 0.8*topic + 0.2*global."""
    C = len(ci)
    global vv_idx
    vv_idx = ci
    qcnt = collections.defaultdict(collections.Counter)
    tcnt = collections.defaultdict(collections.Counter)
    gcnt = collections.Counter()
    for _, r in tr_in.iterrows():
        qcnt[r.question_id][r.root_code] += 1
        tcnt[r.topic_name][r.root_code] += 1
        gcnt[r.root_code] += 1
    gsm = _norm(gcnt, C)
    v = []
    for _, r in tr_out.iterrows():
        c = qcnt.get(r.question_id)
        if c and sum(c.values()) > 0:
            v.append(_norm(c, C))
        else:
            tvec = tcnt.get(r.topic_name)
            if tvec and sum(tvec.values()) > 0:
                v.append(_norm(tvec, C) * 0.8 + gsm * 0.2)
            else:
                v.append(gsm.copy())
    return np.array(v)

def knn_probs(E_pool, y_pool, q_pool, E_out, q_out, ci, k=8, bonus=0.3):
    C = len(ci)
    out = np.zeros((len(E_out), C), dtype=np.float32)
    sim = E_out @ E_pool.T
    nn = np.argsort(-sim, axis=1)[:, :k]
    for r in range(len(E_out)):
        nbs = nn[r]
        same = q_pool[nbs] == q_out[r]
        w = 1.0 + bonus * same
        cnt = np.zeros(C)
        for n, wt in zip(nbs, w):
            cnt[y_pool[n]] += wt
        out[r] = cnt
    return out

def qsplit_folds(tr, nfold=5, seed=7):
    qs = np.array(sorted(tr.question_id.unique()))
    rng = np.random.RandomState(seed)
    rng.shuffle(qs)
    fold_of_q = {q: i % nfold for i, q in enumerate(qs)}
    return np.array([fold_of_q[q] for q in tr.question_id])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--knn")
    ap.add_argument("--ft")
    ap.add_argument("--out", default="outputs/submission_v2.csv")
    ap.add_argument("--alphas", type=float, nargs="*", default=[0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0])
    args = ap.parse_args()

    tr = pd.read_csv(f"{ROOT}/data/train.csv")
    te = pd.read_csv(f"{ROOT}/data/test.csv")
    emb = np.load(f"{ROOT}/outputs/clip_embeddings.npz", allow_pickle=True)
    E_tr, E_te = emb["tr"], emb["te"]
    tr_q, te_q = np.array(emb["tr_qids"], dtype=object), np.array(emb["te_qids"], dtype=object)

    models = []
    codes = None
    for path in [args.knn, args.ft]:
        if not path:
            continue
        d = np.load(path if os.path.isabs(path) else f"{ROOT}/{path}", allow_pickle=True)
        c = list(d["codes"])
        assert codes is None or c == codes
        codes = c
        oof_m = d["oof"] if "oof" in d else None
        te_m = d["te"]
        if oof_m is not None and oof_m.mean(1).mean() > 2:
            oof_m = oof_m / oof_m.sum(1, keepdims=True)
        if te_m.mean(1).mean() > 2:
            te_m = te_m / te_m.sum(1, keepdims=True)
        models.append({"name": os.path.basename(path), "oof": oof_m, "te": te_m})
    ci = {c: i for i, c in enumerate(codes)}
    y = tr.root_code.map(ci).to_numpy()
    ylab = tr.root_code.to_numpy()

    # ---------- Protocol B: clean question-split for knn ----------
    folds = qsplit_folds(tr)
    knn_qsplit_oof = np.zeros((len(tr), len(codes)), dtype=np.float32)
    prior_qsplit = np.zeros((len(tr), len(codes)))
    for i in range(5):
        sel = folds == i
        tr_in, tr_out = tr[~sel], tr[sel]
        E_pool, y_pool, q_pool = E_tr[~sel], y[~sel], np.array(tr_q[~sel])
        k = knn_probs(E_pool, y_pool, q_pool, E_tr[sel], np.array(tr_q[sel]), ci)
        knn_qsplit_oof[sel] = k / k.sum(1, keepdims=True)
        prior_qsplit[sel] = priors_from(tr_in, tr_out, ci)
    for a in args.alphas:
        sc = map3_from_topk(top3_from_scores(prior_qsplit + a * knn_qsplit_oof, codes), ylab)
        print(f"B qsplit: prior + {a}*knn(qsplit) -> {sc:.4f}")
    print(f"B qsplit: prior only -> {map3_from_topk(top3_from_scores(prior_qsplit, codes), ylab):.4f}")
    print(f"B qsplit: knn only    -> {map3_from_topk(top3_from_scores(knn_qsplit_oof, codes), ylab):.4f}")

    # ---------- Protocol A: optimistic LOO with saved oofs ----------
    qcnt_all = collections.defaultdict(collections.Counter)
    for _, r in tr.iterrows():
        qcnt_all[r.question_id][r.root_code] += 1
    prior_loo = []
    for _, r in tr.iterrows():
        c = collections.Counter(qcnt_all[r.question_id])
        c[r.root_code] -= 1
        if c[r.root_code] <= 0:
            del c[r.root_code]
        if sum(c.values()) > 0:
            prior_loo.append(rank_vec(c, ci))
        else:
            prior_loo.append(rank_vec(qcnt_all[r.question_id], ci))
    prior_loo = np.array(prior_loo)
    for m in models:
        if m["oof"] is None:
            continue
        for a in args.alphas:
            sc = map3_from_topk(top3_from_scores(prior_loo + a * m["oof"], codes), ylab)
            print(f"A optimistic: prior_loo + {a}*{m['name']} -> {sc:.4f}")

    # ---------- pick alpha ----------
    # use B for knn (clean); for ft-only use A; for combo average the signals
    bestB = max(args.alphas, key=lambda a: map3_from_topk(top3_from_scores(prior_qsplit + a * knn_qsplit_oof, codes), ylab))
    if args.ft and not args.knn:
        bestA = max(args.alphas, key=lambda a: map3_from_topk(top3_from_scores(prior_loo + a * models[-1]["oof"], codes), ylab))
        a_ft, a_knn = bestA, None
    elif args.ft and args.knn:
        bestA = max(args.alphas, key=lambda a: map3_from_topk(top3_from_scores(prior_loo + a * models[-1]["oof"] + 0.5 * models[0]["oof"], codes), ylab))
        a_ft, a_knn = bestA, bestB
    else:
        a_ft, a_knn = None, bestB
    print("chosen: a_knn(B)=", a_knn, " a_ft(A)=", a_ft)

    # ---------- final test ----------
    prior_te = priors_from(tr, te, ci)
    s_te = prior_te
    if a_knn is not None:
        s_te = s_te + a_knn * models[0]["te"]
    if a_ft is not None:
        s_te = s_te + a_ft * models[-1]["te"]
    pred = top3_from_scores(s_te, codes)
    out = pd.DataFrame({"trace_id": te.trace_id, "root_code": [" ".join(p) for p in pred]})
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    out.to_csv(args.out if os.path.isabs(args.out) else f"{ROOT}/{args.out}", index=False)
    print("wrote", args.out, out.shape)

if __name__ == "__main__":
    main()
