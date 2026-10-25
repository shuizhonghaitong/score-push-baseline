"""v3 builder: v2 (prior + knn + ft) + VLM answer-match bit applied to NONE score.
Usage: python scripts/build_v3.py --knn outputs/knn_probs.npz --ft outputs/ds_ft/finetune_probs.npz \
        --vlm-tr outputs/ds_vlm/vlm_answer_train.csv --vlm-te outputs/ds_vlm/vlm_answer_test.csv \
        --a-knn 0.2 --a-ft 1.0 --out outputs/submission_v3.csv
"""
import argparse, os, sys
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metric import map3_from_topk, top3_from_scores
from build_v2 import priors_from, qsplit_folds, knn_probs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--knn"); ap.add_argument("--ft")
    ap.add_argument("--vlm-tr"); ap.add_argument("--vlm-te")
    ap.add_argument("--a-knn", type=float, default=0.2)
    ap.add_argument("--a-ft", type=float, default=1.0)
    ap.add_argument("--deltas", type=float, nargs="*", default=[0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0])
    ap.add_argument("--delta-yes", type=float, default=0.4)
    ap.add_argument("--delta-no", type=float, default=0.2)
    ap.add_argument("--out", default="outputs/submission_v3.csv")
    args = ap.parse_args()

    tr = pd.read_csv(f"{ROOT}/data/train.csv")
    te = pd.read_csv(f"{ROOT}/data/test.csv")
    emb = np.load(f"{ROOT}/outputs/clip_embeddings.npz", allow_pickle=True)
    E_tr, E_te = emb["tr"], emb["te"]
    tr_q = np.array(emb["tr_qids"], dtype=object)

    models = []
    codes = None
    for path in [args.knn, args.ft]:
        if not path: continue
        d = np.load(path if os.path.isabs(path) else f"{ROOT}/{path}", allow_pickle=True)
        c = list(d["codes"])
        assert codes is None or c == codes
        codes = c
        models.append({"name": os.path.basename(path), "oof": d["oof"] if "oof" in d else None, "te": d["te"]})
    ci = {c: i for i, c in enumerate(codes)}
    iNONE = ci["NONE"]
    y = tr.root_code.to_numpy()
    tr_idx = tr.trace_id.map({t: i for i, t in enumerate(tr.trace_id)}).to_numpy()

    prior_te = priors_from(tr, te, ci)

    # ---------- VLM bit ----------
    vlm_tr = pd.read_csv(args.vlm_tr if os.path.isabs(args.vlm_tr) else f"{ROOT}/{args.vlm_tr}") if args.vlm_tr else None
    vlm_te = pd.read_csv(args.vlm_te if os.path.isabs(args.vlm_te) else f"{ROOT}/{args.vlm_te}") if args.vlm_te else None
    if vlm_tr is not None:
        m = vlm_tr.set_index("trace_id").reindex(tr.trace_id).vlm_match.to_numpy()
        gold = tr.final_answer_correct.to_numpy()
        yes = m == "yes"; no = m == "no"
        agree = np.where((yes == gold), 1, 0)
        unk = ~(yes | no)
        print(f"VLM train: yes={(yes).mean():.3f} no={(no).mean():.3f} unknown={(unk).mean():.3f}")
        print(f"VLM train agreement (yes==gold): {agree[~unk].mean():.4f} (n={~unk.sum()})")
        print(f"gold correct rate when vlm says yes: {gold[yes].mean():.3f}, when no: {gold[no].mean():.3f}")
    if vlm_te is not None:
        m_te = vlm_te.set_index("trace_id").reindex(te.trace_id).vlm_match.to_numpy()
        yes_te = m_te == "yes"; no_te = m_te == "no"
        print(f"VLM test: yes={(yes_te).mean():.3f} no={(no_te).mean():.3f}")
    else:
        yes_te = no_te = None

    # ---------- base test score ----------
    s_te = prior_te
    if args.a_knn and len(models) >= 1:
        s_te = s_te + args.a_knn * models[0]["te"]
    if args.a_ft and len(models) >= 2:
        s_te = s_te + args.a_ft * models[1]["te"]

    # ---------- tune delta on train (simulate VLM noise) ----------
    if vlm_tr is not None:
        # train base score with same recipe (oof)
        s_tr = None
        for w, m in zip([args.a_knn, args.a_ft], models):
            if m["oof"] is None: continue
            s_tr = (m["oof"] * w) if s_tr is None else s_tr + w * m["oof"]
        # add prior (LOO self-excluded) approximated by full question prior (optimistic, fine for delta tuning)
        qcnt = {}
        for _, r in tr.iterrows():
            qcnt.setdefault(r.question_id, []).append(r.root_code)
        from collections import Counter
        gcnt = Counter(tr.root_code)
        prio = []
        for _, r in tr.iterrows():
            c = Counter(qcnt[r.question_id]); c[r.root_code] -= 1
            if c[r.root_code] <= 0: del c[r.root_code]
            prio.append(build_v2_rank(c, gcnt, ci) if False else rankv(c, ci, gcnt))
        prio = np.array(prio)
        base = prio + s_tr
        print(f"train base MAP@3 (no VLM bit): {map3_from_topk(top3_from_scores(base, codes), y):.4f}")
        for d in args.deltas:
            s2 = base.copy()
            s2[yes, iNONE] += d
            s2[no, iNONE] -= d
            sc = map3_from_topk(top3_from_scores(s2, codes), y)
            print(f"  delta={d}: MAP@3={sc:.4f}")

    # ---------- final ----------
    if yes_te is not None:
        s_te[yes_te, iNONE] += args.delta_yes
        s_te[no_te, iNONE] -= args.delta_no
    pred = top3_from_scores(s_te, codes)
    out = pd.DataFrame({"trace_id": te.trace_id, "root_code": [" ".join(p) for p in pred]})
    outp = args.out if os.path.isabs(args.out) else f"{ROOT}/{args.out}"
    os.makedirs(os.path.dirname(outp), exist_ok=True)
    out.to_csv(outp, index=False)
    print("wrote", outp, out.shape)

def rankv(c, ci, gcnt):
    from build_v2 import rank_vec
    if c and sum(c.values()) > 0:
        return rank_vec(c, ci)
    return rank_vec(gcnt, ci)
if __name__ == "__main__":
    main()
