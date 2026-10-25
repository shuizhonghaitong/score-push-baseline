"""v1 baseline: CLIP image emb + question prior (+qtext emb) -> per-class scores, top-3 MAP@3."""
import os, json, argparse
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize
from sklearn.model_selection import StratifiedKFold
from metric import map3_from_topk, top3_from_scores

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load(feat="clip"):
    tr = pd.read_csv(os.path.join(ROOT, "data/train.csv"))
    te = pd.read_csv(os.path.join(ROOT, "data/test.csv"))
    base = feat  # "clip", "dino", "clip_dino"
    feats = []
    for f in base.split("_"):
        d = np.load(os.path.join(ROOT, f"outputs/{f}_train.npz"), allow_pickle=True)
        feats.append(d["emb"])
        d = np.load(os.path.join(ROOT, f"outputs/{f}_test.npz"), allow_pickle=True)
        feats.append(d["emb"])
    emb_tr = np.hstack(feats[0::2])
    emb_te = np.hstack(feats[1::2])
    dq = np.load(os.path.join(ROOT, "outputs/clip_qtext_train.npz"), allow_pickle=True)
    qmap = dict(zip(dq["qids"], dq["emb"]))
    qtr = np.array([qmap[q] for q in tr.question_id])
    qte = np.array([qmap.get(q, np.zeros(512)) for q in te.question_id])
    codes = sorted(tr.root_code.unique())
    ci = {c: i for i, c in enumerate(codes)}
    y = tr.root_code.map(ci).values
    return tr, te, emb_tr, emb_te, qtr, qte, codes, ci, y

def question_prior(df, codes, ci, df_all_labels=None):
    """Smoothed per-question label distribution over training rows."""
    prior = np.zeros((len(df), len(codes)))
    for qid, g in (df_all_labels if df_all_labels is not None else df).groupby("question_id"):
        cnt = np.bincount([ci[v] for v in g.root_code], minlength=len(codes))
        p = (cnt + 0.5) / (cnt.sum() + 0.5 * len(codes))
        rows = df.index[df.question_id == qid]
        prior[rows] = p
    return prior

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--feat", default="clip")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--w-prior", type=float, default=0.5)
    ap.add_argument("--w-qtext", type=float, default=0.0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    tr, te, emb_tr, emb_te, qtr, qte, codes, ci, y = load(args.feat)
    n_codes = len(codes)
    p_tr = question_prior(tr, codes, ci)
    p_te = question_prior(te, codes, ci, df_all_labels=tr)
    glob = np.bincount(y, minlength=n_codes).astype(float)
    glob = (glob + 0.5) / (glob.sum() + 0.5 * n_codes)

    X_tr = emb_tr
    X_te = emb_te
    if args.w_qtext:
        X_tr = np.hstack([emb_tr, args.w_qtext * qtr])
        X_te = np.hstack([emb_te, args.w_qtext * qte])
    X_tr = normalize(X_tr)
    X_te = normalize(X_te)

    skf = StratifiedKFold(n_splits=args.folds, shuffle=True, random_state=42)
    oof = np.zeros((len(tr), n_codes))
    for fold, (trn, val) in enumerate(skf.split(np.zeros(len(tr)), y)):
        m = LogisticRegression(C=args.C, max_iter=2000, solver="lbfgs")
        m.fit(X_tr[trn], y[trn])
        oof[np.ix_(val, m.classes_)] = m.predict_proba(X_tr[val])
        print(f"fold {fold}: MAP3(top3)={map3_from_topk(top3_from_scores(oof[val], codes), [codes[i] for i in y[val]]):.4f}", flush=True)

    # blending: score = a*model + b*prior + c*global
    base = oof
    best = (0, None)
    for a in [0.0, 0.25, 0.5, 0.75, 1.0, 1.5]:
        for b in [0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0]:
            s = a * base + b * p_tr
            s = s - s.min(axis=1, keepdims=True)
            score = map3_from_topk(top3_from_scores(s, codes), [codes[i] for i in y])
            if score > best[0]:
                best = (score, (a, b))
    print(f"best blend a,b={best[1]} MAP3={best[0]:.4f}")
    a, b = best[1]

    # full model
    m = LogisticRegression(C=args.C, max_iter=2000, solver="lbfgs")
    m.fit(X_tr, y)
    def full_proba(m, X):
        p = np.zeros((len(X), n_codes), dtype=np.float64)
        p[:, list(m.classes_)] = m.predict_proba(X)
        return p
    s_tr = a * full_proba(m, X_tr) + b * p_tr
    print("full-model MAP3(train, in-sample):", round(map3_from_topk(top3_from_scores(s_tr, codes), tr.root_code), 4))

    s_te = a * full_proba(m, X_te) + b * p_te
    s_te = np.where(np.isnan(s_te), 0.0, s_te)
    topk = top3_from_scores(s_te, codes)
    out = args.out or os.path.join(ROOT, f"outputs/submission_v1_{args.feat}.csv")
    with open(out, "w") as fh:
        fh.write("trace_id,root_code\n")
        for tid, pred in zip(te.trace_id, topk):
            fh.write(f"{tid},{ ' '.join(pred) }\n")
    np.savez_compressed(os.path.join(ROOT, f"outputs/v1_oof_{args.feat}.npz"), oof=base, prior=p_tr, codes=np.array(codes), te_model=full_proba(m, X_te))
    print("wrote", out)

if __name__ == "__main__":
    main()
