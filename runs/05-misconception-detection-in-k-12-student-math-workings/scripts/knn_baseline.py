"""kNN over CLIP embeddings (question-aware): for each row, nearest labeled rows; prefer same question."""
import os, argparse
import numpy as np
import pandas as pd
from metric import map3_from_topk, top3_from_scores

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--same-q-bonus", type=float, default=0.3)
    args = ap.parse_args()

    tr = pd.read_csv(os.path.join(ROOT, "data/train.csv"))
    te = pd.read_csv(os.path.join(ROOT, "data/test.csv"))
    d = np.load(os.path.join(ROOT, "outputs/clip_embeddings.npz"), allow_pickle=True)
    E_tr, E_te = d["tr"], d["te"]
    tr_q, te_q = d["tr_qids"], d["te_qids"]
    codes = list(d["codes"])
    ci = {c: i for i, c in enumerate(codes)}
    y = tr.root_code.map(ci).values
    C = len(codes)

    tr_qi = {q: i for i, q in enumerate(tr_q)}
    te_qi = [tr_qi.get(q, -1) for q in te_q]

    # train-train similarity for OOF
    oof = np.zeros((len(tr), C), dtype=np.float32)
    te_p = np.zeros((len(te), C), dtype=np.float32)
    k = args.k
    chunk = 200
    for s in range(0, len(tr), chunk):
        e = min(s + chunk, len(tr))
        sim = E_tr[s:e] @ E_tr.T  # (c, N)
        np.fill_diagonal(sim[s:e], -1)
        nn = np.argsort(-sim, axis=1)[:, :k]
        for r in range(e - s):
            nbs = nn[r]
            same = tr_q[nbs] == tr_q[s + r]
            w = 1.0 + args.same_q_bonus * same
            cnt = np.zeros(C)
            for n, wt in zip(nbs, w):
                cnt[y[n]] += wt
            oof[s + r] = cnt
    for s in range(0, len(te), chunk):
        e = min(s + chunk, len(te))
        sim = E_te[s:e] @ E_tr.T
        nn = np.argsort(-sim, axis=1)[:, :k]
        for r in range(e - s):
            nbs = nn[r]
            same = tr_q[nbs] == te_q[s + r]
            w = 1.0 + args.same_q_bonus * same
            cnt = np.zeros(C)
            for n, wt in zip(nbs, w):
                cnt[y[n]] += wt
            te_p[s + r] = cnt
    ylab = [codes[i] for i in y]
    print("kNN OOF MAP@3:", round(map3_from_topk(top3_from_scores(oof, codes), ylab), 4))
    np.savez_compressed(os.path.join(ROOT, "outputs/knn_probs.npz"), oof=oof, te=te_p, codes=np.array(codes),
                        tr_ids=tr.trace_id.values, te_ids=te.trace_id.values, y=y)
    print("saved knn_probs.npz")

if __name__ == "__main__":
    main()
