"""Analyze GRAI3 train-set error and learn a shape-correction residual.
base = Weibull(mean b, mean c) per sample; residual = label - base;
LOOS-ridge on sample-level handcrafted features predicts residual for test."""
import os, re, sys, json
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.cross_decomposition import PLSRegression

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
from train import build_dataset, A_cols

D = np.array([float(c) for c in C.SUPPORT_COLS])

def weib(b, c, d=D):
    return 100.0 * (1.0 - np.exp(-np.power(d / b, c)))

def photo_sample(fname):
    base = re.sub(r"_\d+\.(jpg|JPG)$", "", os.path.basename(fname))
    m = re.search(r"(F\d{3}|G\d{3}|H\d{3})$", base)
    return m.group(1) if m else None

def main():
    lab = pd.read_csv(f"{C.DATA_DIR}/Training_labels_updated.csv").set_index("sample_id")
    preds = json.load(open("state/grai3_preds_tr.json"))
    per = {}
    for p in preds:
        if "b" not in p:
            print("NO b for", p["file"]); continue
        s = photo_sample(p["file"])
        if s is None or s not in lab.index:
            print("unmapped", p["file"]); continue
        per.setdefault(s, []).append(p)
    sids = [s for s in lab.index if s in per and len(per[s]) >= 2]
    print(f"samples with >=2 grai3 preds: {len(sids)} of {len(lab)}")
    # per-sample base curve
    base = {}
    for s in sids:
        ps = per[s]
        b = float(np.mean([p["b"] for p in ps])); c = float(np.mean([p["c"] for p in ps]))
        base[s] = weib(b, c)
    Y = np.array([lab.loc[s, C.SUPPORT_COLS].values.astype(float) for s in sids])
    W = np.array([base[s] for s in sids])
    print("EMD(label vs GRAI3-base Weibull), train set:")
    print("  mean:", round(float(C.emd_trap_mean(Y, W)), 3),
          " median:", round(float(np.median([C.emd_trap_mean(Y[i:i+1], W[i:i+1]) for i in range(len(sids))])), 3))
    res = Y - W
    # sample-level features (same space as hand3: cal tl1600, mean+std)
    tr, te, sub = build_dataset()
    tr = tr.set_index("sid")
    cols = A_cols(tr.reset_index(), "hc")
    Xtr = tr.reindex(sids)[cols].values.astype(float)
    Xte = te[cols].values.astype(float)
    # LOOS residual prediction
    def loos_ridge(X, R, alphas):
        out = {}
        for a in alphas:
            errs = []
            for i in range(len(X)):
                m = np.ones(len(X), bool); m[i] = False
                mmd = Ridge(alpha=a); mmd.fit(X[m], R[m])
                Rp = mmd.predict(X[i:i+1])[0]
                errs.append(C.emd_trap_mean(Y[i:i+1], W[i:i+1] + Rp))
            out[a] = float(np.mean(errs))
        return out
    alphas = [1, 5, 15, 40, 120, 400]
    print("LOOS EMD of W + ridge(alpha) residual:")
    for a, e in loos_ridge(Xtr, res, alphas).items():
        print(f"  alpha={a:5d} LOOS={e:.4f}")
    # best alpha -> test residual
    best = min(loos_ridge(Xtr, res, alphas), key=lambda k: loos_ridge(Xtr, res, alphas)[k])
    Rr = Ridge(alpha=best); Rr.fit(Xtr, res)
    dte = Rr.predict(Xte)
    # test base curves from test preds
    tpreds = json.load(open("state/grai3_preds.json"))
    from build_grai3 import photo_to_sample, SAMPLE_MAP
    tper = {}
    for p in tpreds:
        if "b" not in p: continue
        s = photo_to_sample(p["file"])
        if s: tper.setdefault(s, []).append(p)
    rows = []
    for i, s in enumerate(sub["sample_id"]):
        ps = tper[s]
        b = float(np.mean([p["b"] for p in ps])); c = float(np.mean([p["c"] for p in ps]))
        Wt = weib(b, c)
        psd = C.enforce_rows([np.clip(Wt + dte[i], 0, 100)])[0]
        rows.append([s] + [f"{v:.4f}" for v in psd])
    fn = "submissions/grai3_corr.csv"
    with open(fn, "w") as f:
        f.write("sample_id," + ",".join(C.SUPPORT_COLS) + "\n")
        for r in rows:
            f.write(",".join(r) + "\n")
    print(f"wrote {fn} (alpha={best})")
    # also save per-sample train error table for inspection
    tbl = pd.DataFrame({"sample": sids, "emd": [C.emd_trap_mean(Y[i:i+1], W[i:i+1]) for i in range(len(sids))]}).sort_values("emd", ascending=False)
    print(tbl.to_string(index=False))

if __name__ == "__main__":
    main()
