"""v4 builder: prior + w_knn*knn16 + w_diag*vlm_diag (+ optional ft).
vlm_diag csvs: trace_id, p1, p2, p3  -> soft vector (1.0, 0.5, 0.333) normalized.
Usage:
 python scripts/build_v4.py --knn outputs/knn16_probs.npz \
   --diag-tr outputs/ds_diag/vlm_diag_train.csv --diag-te outputs/ds_diag/vlm_diag_test.csv \
   [--ft outputs/ds_ft/finetune_probs.npz] --w-knn 0.2 --w-diag 0.5 --out outputs/submission_v4.csv
"""
import argparse, os, sys
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metric import map3_from_topk, top3_from_scores
from build_v2 import priors_from, rank_vec

def diag_probs(csv_path, trace_ids, ci):
    df = pd.read_csv(csv_path).set_index("trace_id").reindex(trace_ids)
    C = len(ci)
    out = np.zeros((len(trace_ids), C), dtype=np.float32)
    w = np.array([1.0, 0.5, 1.0/3])
    for i, (tid, row) in enumerate(df.iterrows()):
        s = 0.0
        for j in range(3):
            c = row[f"p{j+1}"]
            if isinstance(c, str) and c in ci:
                out[i, ci[c]] += w[j]
                s += w[j]
        if s > 0:
            out[i] /= s
        else:
            out[i, ci["NONE"]] = 1.0
    return out

def map3_diag(df, gold, ci):
    # MAP@3 straight from p1/p2/p3
    tot = 0.0
    for (_, r), yi in zip(df.iterrows(), gold):
        seen = set(); sc = 0.0
        for k, p in enumerate([r.p1, r.p2, r.p3], 1):
            if p in seen: continue
            seen.add(p)
            if p == yi:
                sc = 1.0/k; break
        tot += sc
    return tot/len(df)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--knn"); ap.add_argument("--ft")
    ap.add_argument("--diag-tr"); ap.add_argument("--diag-te")
    ap.add_argument("--w-knn", type=float, default=0.2)
    ap.add_argument("--w-diag", type=float, default=0.5)
    ap.add_argument("--w-ft", type=float, default=0.0)
    ap.add_argument("--ans-tr"); ap.add_argument("--ans-te")
    ap.add_argument("--delta-yes", type=float, default=1.5)
    ap.add_argument("--delta-no", type=float, default=0.2)
    ap.add_argument("--grid", type=float, nargs="*", default=None)
    ap.add_argument("--out", default="outputs/submission_v4.csv")
    args = ap.parse_args()

    tr = pd.read_csv(f"{ROOT}/data/train.csv")
    te = pd.read_csv(f"{ROOT}/data/test.csv")
    codes = None
    for path in [args.knn, args.ft]:
        if path:
            d = np.load(path if os.path.isabs(path) else f"{ROOT}/{path}", allow_pickle=True)
            codes = list(d["codes"]); break
    if args.diag_tr:
        dd = pd.read_csv(args.diag_tr if os.path.isabs(args.diag_tr) else f"{ROOT}/{args.diag_tr}")
    assert codes is not None
    ci = {c: i for i, c in enumerate(codes)}
    y = tr.root_code.to_numpy()
    iNONE = ci["NONE"]

    prior_te = priors_from(tr, te, ci)
    s_te = prior_te.copy()

    # --- knn ---
    if args.knn:
        d = np.load(args.knn if os.path.isabs(args.knn) else f"{ROOT}/{args.knn}", allow_pickle=True)
        te_m = d["te"]
        if te_m.mean(1).mean() > 2: te_m = te_m / te_m.sum(1, keepdims=True)
        s_te = s_te + args.w_knn * te_m

    # --- diag ---
    diag_tr = diag_te = None
    if args.diag_tr:
        diag_tr = diag_probs(args.diag_tr if os.path.isabs(args.diag_tr) else f"{ROOT}/{args.diag_tr}", tr.trace_id, ci)
        # train diag MAP@3 (zero-shot, no leakage)
        ddt = pd.read_csv(args.diag_tr if os.path.isabs(args.diag_tr) else f"{ROOT}/{args.diag_tr}")
        print("diag train MAP@3 (direct p1/p2/p3):", round(map3_diag(ddt, y, ci), 4))
    if args.diag_te:
        diag_te = diag_probs(args.diag_te if os.path.isabs(args.diag_te) else f"{ROOT}/{args.diag_te}", te.trace_id, ci)
        s_te = s_te + args.w_diag * diag_te

    # --- vlm answer bit ---
    if args.ans_te:
        ve = pd.read_csv(args.ans_te if os.path.isabs(args.ans_te) else f"{ROOT}/{args.ans_te}").set_index("trace_id").reindex(te.trace_id).vlm_match
        yes = (ve == "yes").to_numpy(); no = (ve == "no").to_numpy()
        s_te[yes, iNONE] += args.delta_yes
        s_te[no, iNONE] -= args.delta_no

    # --- ft ---
    if args.ft:
        d = np.load(args.ft if os.path.isabs(args.ft) else f"{ROOT}/{args.ft}", allow_pickle=True)
        te_m = d["te"]
        if te_m.mean(1).mean() > 2: te_m = te_m / te_m.sum(1, keepdims=True)
        s_te = s_te + args.w_ft * te_m

    # --- grid on train (row-wise, zero-shot models are leakage-free) ---
    if diag_tr is not None and args.grid:
        # base train score components
        prior_tr = None
        # approximate train prior (full question dist; optimistic but fine for weight tuning of zero-shot diag)
        from collections import Counter
        gcnt = Counter(tr.root_code)
        gsm = np.zeros(len(codes))
        for c2, n in gcnt.items(): gsm[ci[c2]] = n / sum(gcnt.values())
        qcnt = {}
        for _, r in tr.iterrows():
            qcnt.setdefault(r.question_id, []).append(r.root_code)
        pt = []
        for _, r in tr.iterrows():
            c = Counter(qcnt[r.question_id])
            pt.append(np.array([c.get(x, 0)/sum(c.values()) if x in ci else 0.0 for x in codes]) if sum(c.values())>0 else gsm.copy())
        pt = np.array(pt)
        s_tr_base = pt.copy()
        if args.ans_tr:
            vt = pd.read_csv(args.ans_tr if os.path.isabs(args.ans_tr) else f"{ROOT}/{args.ans_tr}").set_index("trace_id").reindex(tr.trace_id).vlm_match
            s_tr_base[(vt == "yes").to_numpy(), iNONE] += args.delta_yes
            s_tr_base[(vt == "no").to_numpy(), iNONE] -= args.delta_no
        if args.knn:
            d = np.load(args.knn if os.path.isabs(args.knn) else f"{ROOT}/{args.knn}", allow_pickle=True)
            oof = d["oof"]
            if oof.mean(1).mean() > 2: oof = oof / oof.sum(1, keepdims=True)
            s_tr_base = s_tr_base + args.w_knn * oof
        print(f"train base (no diag) MAP@3: {map3_from_topk(top3_from_scores(s_tr_base, codes), y):.4f}")
        best = (0, None)
        for w in args.grid:
            s2 = s_tr_base + w * diag_tr
            sc = map3_from_topk(top3_from_scores(s2, codes), y)
            print(f"  w_diag={w}: {sc:.4f}")
            if sc > best[0]: best = (sc, w)
        print("best w_diag:", best)

    pred = top3_from_scores(s_te, codes)
    out = pd.DataFrame({"trace_id": te.trace_id, "root_code": [" ".join(p) for p in pred]})
    outp = args.out if os.path.isabs(args.out) else f"{ROOT}/{args.out}"
    os.makedirs(os.path.dirname(outp), exist_ok=True)
    out.to_csv(outp, index=False)
    print("wrote", outp, out.shape)

if __name__ == "__main__":
    main()
