"""v5 builder: prior + w_knn*knn + w_diag*diag + answer bit (dy/dn).
- answer bit: 'yes'->+dy on NONE, 'no'->-dn on NONE (uses provided vlm_match CSV, reflects test noise)
- diag: p1/p2/p3 soft vector (weights 1,0.5,0.33) normalized; leakage-free for zero-shot
- tunes w_diag (and optional dy/dn grid) on TRAIN MAP@3 using the same recipe.
Usage:
 python scripts/build_v5.py --knn outputs/knn16_probs.npz \
   --ans-tr <answer train csv> --ans-te <answer test csv> \
   [--diag-tr <diag train csv> --diag-te <diag test csv>] \
   --dy 1.5 --dn 0.2 --w-knn 0.2 --grid 0.0 0.3 0.6 1.0 1.5 2.0 [--grid-dy ...] [--grid-dn ...] \
   --out outputs/submission_v5.csv
"""
import argparse, os, sys
import numpy as np, pandas as pd
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metric import map3_from_topk, top3_from_scores
from build_v2 import priors_from

def load(path):
    return path if os.path.isabs(path) else f"{ROOT}/{path}"

def diag_probs(csv_path, trace_ids, ci):
    df = pd.read_csv(load(csv_path)).set_index("trace_id").reindex(trace_ids)
    C = len(ci)
    out = np.zeros((len(trace_ids), C), dtype=np.float32)
    w = np.array([1.0, 0.5, 1.0/3])
    for i, (tid, row) in enumerate(df.iterrows()):
        s = 0.0
        for j in range(3):
            c = row[f"p{j+1}"]
            if isinstance(c, str) and c in ci:
                out[i, ci[c]] += w[j]; s += w[j]
        if s > 0: out[i] /= s
        else: out[i, ci["NONE"]] = 1.0
    return out

def map3_diag(df, gold, ci):
    tot = 0.0
    for (_, r), yi in zip(df.iterrows(), gold):
        seen = set(); sc = 0.0
        for k, p in enumerate([r.p1, r.p2, r.p3], 1):
            if p in seen: continue
            seen.add(p)
            if p == yi: sc = 1.0/k; break
        tot += sc
    return tot/len(df)

def load_codes(*paths):
    for p in paths:
        if p and os.path.exists(load(p)) and p.endswith(".npz"):
            d = np.load(load(p), allow_pickle=True)
            if "codes" in d: return list(d["codes"])
    tr = pd.read_csv(f"{ROOT}/data/train.csv")
    return sorted(tr.root_code.unique()) + (["NONE"] if "NONE" not in set(tr.root_code) else [])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--knn"); ap.add_argument("--ans-tr"); ap.add_argument("--ans-te")
    ap.add_argument("--diag-tr"); ap.add_argument("--diag-te")
    ap.add_argument("--dy", type=float, default=1.5); ap.add_argument("--dn", type=float, default=0.2)
    ap.add_argument("--w-knn", type=float, default=0.2)
    ap.add_argument("--grid", type=float, nargs="*", default=[0.0])
    ap.add_argument("--grid-dy", type=float, nargs="*", default=None)
    ap.add_argument("--grid-dn", type=float, nargs="*", default=None)
    ap.add_argument("--out", default="outputs/submission_v5.csv")
    args = ap.parse_args()

    tr = pd.read_csv(f"{ROOT}/data/train.csv"); te = pd.read_csv(f"{ROOT}/data/test.csv")
    codes = load_codes(args.knn, args.diag_tr)
    ci = {c: i for i, c in enumerate(codes)}; iNONE = ci["NONE"]; C = len(codes)
    y = tr.root_code.to_numpy()
    print(f"n_codes={C}")

    knn = None
    if args.knn:
        d = np.load(load(args.knn), allow_pickle=True)
        oof = d["oof"]; te_m = d["te"]
        if oof.mean(1).mean() > 2: oof = oof/oof.sum(1, keepdims=True)
        if te_m.mean(1).mean() > 2: te_m = te_m/te_m.sum(1, keepdims=True)
        knn = (oof, te_m)

    ans_tr_m = ans_te_m = None
    if args.ans_tr:
        ans_tr_m = pd.read_csv(load(args.ans_tr)).set_index("trace_id").reindex(tr.trace_id).vlm_match.to_numpy()
    if args.ans_te:
        ans_te_m = pd.read_csv(load(args.ans_te)).set_index("trace_id").reindex(te.trace_id).vlm_match.to_numpy()

    diag_tr = diag_te = None
    if args.diag_tr:
        diag_tr = diag_probs(args.diag_tr, tr.trace_id, ci)
        ddt = pd.read_csv(load(args.diag_tr)).set_index("trace_id").reindex(tr.trace_id)
        print(f"diag train MAP@3 (direct): {map3_diag(ddt, y, ci):.4f}")
    if args.diag_te:
        diag_te = diag_probs(args.diag_te, te.trace_id, ci)

    # train base (no diag)
    prior_tr_full = None
    # full question prior for train (optimistic but consistent for relative tuning)
    qcnt = {}
    for _, r in tr.iterrows(): qcnt.setdefault(r.question_id, []).append(r.root_code)
    gcnt = Counter(y); gsm = np.array([gcnt.get(c,0)/gcnt.__len__() for c in codes]) if False else np.zeros(C)
    gsm = np.array([gcnt.get(c,0)/sum(gcnt.values()) for c in codes])
    s_tr = np.zeros((len(tr), C)); s_te = priors_from(tr, te, ci)
    for i, (_, r) in enumerate(tr.iterrows()):
        c = Counter(qcnt[r.question_id]); s = sum(c.values())
        v = np.zeros(C)
        for c2, n in c.items():
            if c2 in ci and n > 0: v[ci[c2]] = n
        s_tr[i] = v/s if s > 0 else gsm
    if knn: s_tr = s_tr + args.w_knn*knn[0]; s_te = s_te + args.w_knn*knn[1]

    def score_train(dy, dn, s):
        s2 = s.copy()
        if ans_tr_m is not None:
            s2[ans_tr_m == "yes", iNONE] += dy
            s2[ans_tr_m == "no", iNONE] -= dn
        return map3_from_topk(top3_from_scores(s2, codes), y)

    print(f"train base (no diag) MAP@3: {score_train(args.dy, args.dn, s_tr):.4f}")

    # grid search w_diag x dy x dn on train
    dys = args.grid_dy or [args.dy]; dns = args.grid_dn or [args.dn]
    best = (-1, None)
    for wd in args.grid:
        s_d = s_tr + (wd*diag_tr if diag_tr is not None else 0)
        for dy in dys:
            for dn in dns:
                sc = score_train(dy, dn, s_d)
                tag = f"w_diag={wd} dy={dy} dn={dn}: {sc:.4f}"
                print("  " + tag)
                if sc > best[0]: best = (sc, (wd, dy, dn))
    print("BEST train:", best)

    # build test with best (fall back to requested dy/dn if no grid found)
    if best[1] is not None:
        wd, dy, dn = best[1]
    else:
        wd, dy, dn = 0.0, args.dy, args.dn
    st = s_te.copy()
    if diag_te is not None: st = st + wd*diag_te
    if ans_te_m is not None:
        st[ans_te_m == "yes", iNONE] += dy
        st[ans_te_m == "no", iNONE] -= dn
    pred = top3_from_scores(st, codes)
    out = pd.DataFrame({"trace_id": te.trace_id, "root_code": [" ".join(p) for p in pred]})
    outp = load(args.out); os.makedirs(os.path.dirname(outp), exist_ok=True)
    out.to_csv(outp, index=False)
    print(f"final: w_diag={wd} dy={dy} dn={dn}  -> wrote {outp} {out.shape}")

if __name__ == "__main__":
    main()
