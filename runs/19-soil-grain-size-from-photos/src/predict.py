"""Produce submission CSVs from cached features. Configurable feature set / model / transform."""
import os, sys, json, argparse
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import BayesianRidge, ElasticNet

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
from train import build_dataset, CumulativeTransform, DeltaTransform, make_models, YScaled, loocv_score, A_cols

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fs", default="hc")
    ap.add_argument("--model", default="ridge_15")
    ap.add_argument("--tf", default="delta")
    ap.add_argument("--out", required=True)
    ap.add_argument("--loo", action="store_true", help="report LOOCV score of this config")
    a = ap.parse_args()

    tr, te, sub = build_dataset()
    tr = tr.sort_values("sample_id").reset_index(drop=True)
    te = te.merge(sub[["sample_id"]], left_on="sid", right_on="sample_id", how="inner")
    Y = tr[C.SUPPORT_COLS].values.astype(float)
    hc_cols = A_cols(tr, "hc")
    dino_cols = A_cols(tr, "dino")
    fs = {"hc": hc_cols, "dino": dino_cols, "hc+dino": hc_cols + dino_cols}[a.fs]
    tf = CumulativeTransform() if a.tf == "cum" else DeltaTransform()
    models = make_models()
    Xtr, Xte = tr[fs].values.astype(float), te[fs].values.astype(float)

    if a.loo:
        print(f"LOOCV({a.fs},{a.model},{a.tf}) = {loocv_score(Xtr, Y, models[a.model], tf):.4f}")

    m = YScaled(models[a.model], tf)
    m.fit(Xtr, Y)
    P = m.predict(Xte)
    out = sub.copy()
    # align rows to submission order (P rows follow te order)
    idx = {s: i for i, s in enumerate(te["sample_id"])}
    rows = []
    for s in out["sample_id"]:
        r = P[idx[s]]
        rows.append([s] + [f"{v:.4f}" for v in r])
    with open(a.out, "w") as f:
        f.write("sample_id," + ",".join(C.SUPPORT_COLS) + "\n")
        for r in rows:
            f.write(",".join(r) + "\n")
    print("wrote", a.out)
    df = pd.read_csv(a.out)
    print(df.to_string(index=False))

if __name__ == "__main__":
    main()
