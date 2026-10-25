"""3-config OOF ensemble (ridge40cum, ridge15delta, pls3delta) on cal tl1600 hc,
photo-level preds averaged per sample; optional blend with GRAI Weibull F (median)."""
import os, sys, argparse
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
from train import build_dataset, A_cols, CumulativeTransform, DeltaTransform, YScaled, make_models

CONFIGS = [
    ("ridge_40", "cum"),
    ("ridge_15", "delta"),
    ("pls_3", "delta"),
]

def configs():
    models = make_models()
    tf = {"cum": CumulativeTransform(), "delta": DeltaTransform()}
    return [(name, models[name], tf[kind]) for name, kind in CONFIGS]

def oof_psd(tr, cols):
    Y = tr[C.SUPPORT_COLS].values.astype(float)
    X = tr[cols].values.astype(float)
    out = {}
    for name, fn, tfn in configs():
        oof = np.zeros_like(Y)
        for i in range(len(Y)):
            m = np.ones(len(Y), bool); m[i] = False
            mm = YScaled(fn, tfn); mm.fit(X[m], Y[m])
            oof[i] = mm.predict(X[i:i+1])[0]
        out[name] = oof
    return Y, out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--w-hand", type=float, default=1.0)
    ap.add_argument("--grai-mode", default="Fmed", choices=["Fmed", "Fmean"])
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.environ.setdefault("FEAT_HC", "feat_handcrafted_cal_tl1600.npz")
    tr, te, sub = build_dataset()
    tr = tr.sort_values("sample_id").reset_index(drop=True)
    cols = A_cols(tr, "hc")
    Y, oof = oof_psd(tr, cols)
    Ys = Y
    ens = np.mean([oof[n] for n, _, _ in configs()], axis=0)
    print(f"LOOS single: " + "  ".join(f"{n}={C.emd_trap_mean(Ys, oof[n]):.3f}" for n, _, _ in configs()))
    print(f"LOOS ensemble(3): {C.emd_trap_mean(Ys, ens):.4f}")
    P_ens = np.mean([
        YScaled(fn, tfn).fit(tr[cols].values.astype(float), Y).predict(te[cols].values.astype(float))
        for _, fn, tfn in configs()
    ], axis=0)
    g = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "state", "grai_preds.npz"))
    P_grai = {}
    for s in sorted(set(g["te_ids"])):
        X = g["te_F"][np.array([i == s for i in g["te_ids"]])]
        P_grai[s] = X.mean(axis=0) if a.grai_mode == "Fmean" else np.median(X, axis=0)
    rows = []
    for i, s in enumerate(sub["sample_id"]):
        h = P_ens[i]; gv = C.enforce_rows([P_grai[s]])[0]
        mix = a.w_hand * h + (1 - a.w_hand) * gv
        mix = C.enforce_rows([mix])[0]
        rows.append([s] + [f"{v:.4f}" for v in mix])
    with open(a.out, "w") as f:
        f.write("sample_id," + ",".join(C.SUPPORT_COLS) + "\n")
        for r in rows:
            f.write(",".join(r) + "\n")
    print(f"wrote {a.out} (w_hand={a.w_hand}, grai={a.grai_mode}, 3-config ens)")

if __name__ == "__main__":
    main()
