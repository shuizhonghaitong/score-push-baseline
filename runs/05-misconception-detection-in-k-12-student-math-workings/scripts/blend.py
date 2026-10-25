"""Blend model score sets (row x code) into a top-3 submission.

Usage: python3 blend.py --oof outputs/v1_oof.npz [--ft outputs/finetune_probs.npz]
        [--vlm-tr outputs/vlm_answer_train.csv --vlm-te outputs/vlm_answer_test.csv]
        [--out outputs/submission_blend.csv]
Scores for each source are L2-normalized per row before weighting.
"""
import os, argparse
import numpy as np
import pandas as pd
from metric import map3_from_topk, top3_from_scores

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
tr = pd.read_csv(os.path.join(ROOT, "data/train.csv"))
te = pd.read_csv(os.path.join(ROOT, "data/test.csv"))

def norm(s):
    s = np.nan_to_num(s, nan=0.0)
    n = np.linalg.norm(s, axis=1, keepdims=True)
    n[n == 0] = 1
    return s / n

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oof", default=None)
    ap.add_argument("--ft", default=None)
    ap.add_argument("--vlm-tr", default=None)
    ap.add_argument("--vlm-te", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    # base: clip LR oof + prior (already blended optimally in v1)
    d = np.load(args.oof, allow_pickle=True)
    codes = list(d["codes"])
    ci = {c: i for i, c in enumerate(codes)}
    y = tr.root_code.map(ci).values
    S_tr = [norm(d["oof"])]
    S_te = []
    s_tr = d["oof"] + d["prior"] * 0.5
    s_te = None
    names = ["clip_lr"]
    if args.ft:
        f = np.load(args.ft, allow_pickle=True)
        S_tr.append(norm(f["oof"]))
        S_te.append(norm(f["te"]))
        names.append("finetune")
    if args.vlm_tr:
        vtr = pd.read_csv(args.vlm_tr).set_index("trace_id")
        vte = pd.read_csv(args.vlm_te).set_index("trace_id")
        NONE = ci["NONE"]
        a_tr = np.zeros((len(tr), len(codes)))
        a_te = np.zeros((len(te), len(codes)))
        for i, r in enumerate(tr.itertuples()):
            m = vtr.loc[r.trace_id, "vlm_match"] if r.trace_id in vtr.index else "unknown"
            if m == "yes":
                a_tr[i, NONE] = 1.0
            elif m == "no":
                a_tr[i, NONE] = -1.0
        for i, r in enumerate(te.itertuples()):
            m = vte.loc[r.trace_id, "vlm_match"] if r.trace_id in vte.index else "unknown"
            if m == "yes":
                a_te[i, NONE] = 1.0
            elif m == "no":
                a_te[i, NONE] = -1.0
        S_tr.append(norm(a_tr)); S_te.append(norm(a_te))
        names.append("vlm_answer")

    # global prior for test rows without question prior
    prior_te = np.zeros((len(te), len(codes)))
    for qid, g in tr.groupby("question_id"):
        cnt = np.bincount([ci[v] for v in g.root_code], minlength=len(codes))
        rows = te.index[te.question_id == qid]
        prior_te[rows] = cnt + 0.5
    glob = np.bincount(y, minlength=len(codes)).astype(float) + 0.5
    prior_te[prior_te.sum(1) <= 1.0] = glob
    prior_tr = np.zeros((len(tr), len(codes)))
    for qid, g in tr.groupby("question_id"):
        cnt = np.bincount([ci[v] for v in g.root_code], minlength=len(codes])
        rows = tr.index[tr.question_id == qid]
        prior_tr[rows] = cnt + 0.5

    import itertools
    best = (-1, None)
    grid = [0.0, 0.25, 0.5, 1.0, 1.5, 2.0]
    w_prior = [0.0, 0.25, 0.5, 1.0]
    combos = list(itertools.product([1.0] + grid, *[grid for _ in range(len(S_tr) - 1)], w_prior))
    for w in combos:
        s = w[0] * S_tr[0]
        for j in range(1, len(S_tr)):
            s = s + w[j] * S_tr[j]
        s = s + w[-1] * norm(prior_tr)
        sc = map3_from_topk(top3_from_scores(s, codes), [codes[i] for i in y])
        if sc > best[0]:
            best = (sc, w)
    sc, w = best
    print("best weights", dict(zip(names + ["prior"], w)), "MAP@3(train oof)=", round(sc, 4))

    s_te = w[0] * S_te[0] if S_te else 0.0
    for j in range(1, len(S_te)):
        s_te = s_te + w[j] * S_te[j]
    s_te = s_te + w[-1] * norm(prior_te)
    topk = top3_from_scores(s_te, codes)
    out = args.out or os.path.join(ROOT, "outputs/submission_blend.csv")
    with open(out, "w") as fh:
        fh.write("trace_id,root_code\n")
        for tid, pred in zip(te.trace_id, topk):
            fh.write(f"{tid},{ ' '.join(pred) }\n")
    print("wrote", out)

if __name__ == "__main__":
    main()
