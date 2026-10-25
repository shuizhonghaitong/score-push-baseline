"""v6 builders: pure-LoRA top3, or blend (prior + wk*knn96b5 + bit + w_diag*diag_lora).
Usage:
  build_v6_lora.py --mode pure  --diag-te outputs/ds_lora/vlm_diag_test.csv --out outputs/submission_v6a_purelora.csv
  build_v6_lora.py --mode blend --diag-tr .../vlm_diag_train.csv --diag-te .../vlm_diag_test.csv \
      --w-diag 0.4 --w-knn 0.3 --dy 1.5 --dn 0.2 --out outputs/submission_v6b_blend.csv
"""
import argparse, os
import numpy as np, pandas as pd
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
import sys; sys.path.insert(0, os.path.join(ROOT, "scripts"))
from metric import top3_from_scores
from build_v5 import diag_probs, load_codes, load

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["pure", "blend"], default="blend")
    ap.add_argument("--diag-tr"); ap.add_argument("--diag-te")
    ap.add_argument("--w-diag", type=float, default=0.4)
    ap.add_argument("--w-knn", type=float, default=0.3)
    ap.add_argument("--dy", type=float, default=1.5); ap.add_argument("--dn", type=float, default=0.2)
    ap.add_argument("--out", default="outputs/submission_v6.csv")
    args = ap.parse_args()

    tr = pd.read_csv(f"{ROOT}/data/train.csv"); te = pd.read_csv(f"{ROOT}/data/test.csv")
    codes = load_codes(None, args.diag_tr)
    ci = {c: i for i, c in enumerate(codes)}

    dte = pd.read_csv(load(args.diag_te))
    if args.mode == "pure":
        rows = []
        for _, r in dte.iterrows():
            got = [c for c in [r.p1, r.p2, r.p3] if isinstance(c, str) and c in ci]
            seen = set(); top3 = []
            for c in got:
                if c not in seen: top3.append(c); seen.add(c)
            while len(top3) < 3: top3.append("NONE")
            rows.append(" ".join(top3[:3]))
        out = pd.DataFrame({"trace_id": dte.trace_id, "root_code": rows})
    else:
        from build_v2 import priors_from
        s = priors_from(tr, te, ci)
        k = np.load(f"{ROOT}/outputs/knn96b5_probs.npz", allow_pickle=True)["te"].astype(float)
        k = k / k.sum(1, keepdims=True)
        s = s + args.w_knn * k
        diag_te = diag_probs(args.diag_te, te.trace_id, ci)
        s = s + args.w_diag * diag_te
        ans = pd.read_csv(f"{ROOT}/outputs/ds_vlm/vlm_answer_test.csv").set_index("trace_id").reindex(te.trace_id).vlm_match.to_numpy()
        s[ans == "yes", ci["NONE"]] += args.dy
        s[ans == "no", ci["NONE"]] -= args.dn
        pred = top3_from_scores(s, codes)
        out = pd.DataFrame({"trace_id": te.trace_id, "root_code": [" ".join(p) for p in pred]})
    outp = load(args.out); os.makedirs(os.path.dirname(outp), exist_ok=True)
    out.to_csv(outp, index=False)
    print(f"{args.mode} -> {outp} {out.shape}")

if __name__ == "__main__":
    main()
