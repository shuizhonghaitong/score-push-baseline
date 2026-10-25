"""Ensemble submission: handcrafted (calibrated, tl1600) + public GRAI CNN."""
import os, sys, json
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
from train import build_dataset, A_cols, CumulativeTransform, DeltaTransform, YScaled, make_models

def handcrafted_prediction():
    tr, te, sub = build_dataset()
    tr = tr.sort_values("sample_id").reset_index(drop=True)
    Y = tr[C.SUPPORT_COLS].values.astype(float)
    hc_cols = A_cols(tr, "hc")
    m = YScaled(make_models()["ridge_40"], CumulativeTransform())
    m.fit(tr[hc_cols].values.astype(float), Y)
    te = te.merge(sub[["sample_id"]], left_on="sid", right_on="sample_id", how="inner")
    P = m.predict(te[hc_cols].values.astype(float))
    return {s: P[i] for i, s in enumerate(te["sample_id"])}, sub

def grai_prediction(mode="Fmed"):
    g = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "state", "grai_preds.npz"))
    te_ids, F = g["te_ids"], g["te_F"]
    out = {}
    for s in sorted(set(te_ids)):
        m = np.array([i == s for i in te_ids])
        X = F[m]
        out[s] = X.mean(axis=0) if mode == "Fmean" else np.median(X, axis=0)
    return out

def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--w-hand", type=float, default=0.5)
    ap.add_argument("--grai-mode", default="Fmed")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    P_hand, sub = handcrafted_prediction()
    P_grai = grai_prediction(a.grai_mode)
    rows = []
    for s in sub["sample_id"]:
        h = P_hand[s]; g = C.enforce_rows([P_grai[s]])[0]
        mix = a.w_hand * h + (1 - a.w_hand) * g
        mix = C.enforce_rows([mix])[0]
        rows.append([s] + [f"{v:.4f}" for v in mix])
    with open(a.out, "w") as f:
        f.write("sample_id," + ",".join(C.SUPPORT_COLS) + "\n")
        for r in rows:
            f.write(",".join(r) + "\n")
    print(f"wrote {a.out} (w_hand={a.w_hand}, grai={a.grai_mode})")

if __name__ == "__main__":
    main()
