"""Augmented photo-level model: hc70(tl1600 cal) + GRAI (lnb, c) -> per-photo PSD -> mean per sample."""
import os, sys, argparse
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import train2 as T

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=20.0)
    ap.add_argument("--tf", default="cum")
    ap.add_argument("--agg", default="mean", choices=["mean", "median"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    d, tr_ids, te_ids, Y, keys = T.load_photos()
    g = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "state", "grai_preds.npz"))
    assert (np.array(tr_ids) == g["tr_ids"]).all()
    Xtr = np.concatenate([d["hc1600"][0], g["tr_bc"]], axis=1)
    Xte = np.concatenate([d["hc1600"][1], g["te_bc"]], axis=1)

    tfn = T.TF[a.tf]
    Ytr_all = np.array([Y[s] for s in tr_ids])
    sc = StandardScaler()
    Ys = sc.fit_transform(tfn.enc(Ytr_all))
    m = Ridge(alpha=a.alpha)
    m.fit(Xtr, Ys)
    P_photo = tfn.dec(sc.inverse_transform(m.predict(Xte)))

    sub = pd.read_csv(f"{C.DATA_DIR}/sample_submission.csv")
    rows = []
    for s in sub["sample_id"]:
        mm = np.array([sid == s for sid in te_ids])
        X = P_photo[mm]
        psd = X.mean(axis=0) if a.agg == "mean" else np.median(X, axis=0)
        psd = C.enforce_rows([psd])[0]
        rows.append([s] + [f"{v:.4f}" for v in psd])
    with open(a.out, "w") as f:
        f.write("sample_id," + ",".join(C.SUPPORT_COLS) + "\n")
        for r in rows:
            f.write(",".join(r) + "\n")
    print(f"wrote {a.out} (alpha={a.alpha}, tf={a.tf}, agg={a.agg})")

if __name__ == "__main__":
    main()
