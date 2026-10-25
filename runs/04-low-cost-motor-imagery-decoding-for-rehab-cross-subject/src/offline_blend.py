"""Offline ML+DL blend: weights from OOF (per-file balanced acc), apply to test probs."""
import sys
import numpy as np
import pandas as pd

def load(name, path):
    return pd.read_csv(path)

def bal_acc_cv(P, tfile, ty):
    accs = []
    for f in np.unique(tfile):
        m = tfile == f
        k2 = int(m.sum()) // 2
        Pm = P[m]
        pred = np.zeros(int(m.sum()), int)
        pred[np.argpartition(-Pm, k2)[:k2]] = 1
        accs.append(float((pred == ty[m]).mean()))
    return float(np.mean(accs))

def main():
    oof_ml_path, oof_dl_path, ml_p_path, dl_p_path, out_path = sys.argv[1:6]
    w_step = float(sys.argv[6]) if len(sys.argv) > 6 else 0.1
    oofml = load("ml", oof_ml_path)["P"].to_numpy()
    oofdl = load("dl", oof_dl_path)
    ty = oofdl["y"].to_numpy()
    tfile = oofdl["file"].to_numpy()
    if oofdl.shape[1] > 5:  # per-arch columns
        dlcols = {
            "dl_mean": oofdl[["p_ms", "p_sc", "p_attn"]].mean(axis=1).to_numpy(),
            "dl_ms": oofdl["p_ms"].to_numpy(),
            "dl_sc": oofdl["p_sc"].to_numpy(),
            "dl_attn": oofdl["p_attn"].to_numpy(),
        }
    else:
        dlcols = {"dl_mean": oofdl["P"].to_numpy()}
    ml_p = load("mlp", ml_p_path)["P"].to_numpy()
    dl_p = load("dlp", dl_p_path)
    dl_col = sys.argv[7] if len(sys.argv) > 7 else "P"
    if dl_col not in dl_p.columns:
        print("dl column", dl_col, "not in", list(dl_p.columns), "-> using P")
        dl_col = "P"
    dlP = dl_p[dl_col].to_numpy()
    n = min(len(oofml), len(ty))
    assert len(ml_p) == 360
    cands = {"ml": oofml[:n], **{k: v[:n] for k, v in dlcols.items()}}
    results = []
    for name, v in cands.items():
        for w in np.arange(0.0, 1.0 + 1e-9, w_step):
            results.append((round(bal_acc_cv(w * oofml[:n] + (1 - w) * v, tfile[:n], ty[:n]), 4), name, round(float(w), 2)))
    results.sort(reverse=True)
    print("top OOF blend candidates (acc, dl_variant, w_ml):")
    for r in results[:8]:
        print("  ", r)
    best_acc, best_name, best_w = results[0]
    # test: if the chosen dl variant is a single arch and the dl probs file only has ensemble P, note it
    note = ""
    if best_name.startswith("dl_") and best_name != "dl_mean" and "P" in dl_p.columns:
        note = " (NOTE: using ensemble P as proxy for single-arch variant)"
    Pfin = best_w * ml_p + (1 - best_w) * dlP
    tsub = ml_p_file = None
    ml = load("mlfull", ml_p_path)
    tsub = ml["file"].to_numpy() if "file" in ml.columns else None
    # balanced per file
    predb = np.zeros(len(Pfin), int)
    files = sorted(set(tsub)) if tsub is not None else [0]
    if tsub is None:
        m = np.arange(len(Pfin))
        k2 = len(m) // 2
        predb[m[np.argpartition(-Pfin, k2)[:k2]]] = 1
    else:
        for f in files:
            m = np.flatnonzero(tsub == f)
            k2 = len(m) // 2
            predb[m[np.argpartition(-Pfin[m], k2)[:k2]]] = 1
    rows = [{"ID": str(i), "TARGET": "move" if predb[i] == 1 else "rest"} for i in range(len(Pfin))]
    out = pd.DataFrame(rows)
    out.to_csv(out_path, index=False)
    pd.DataFrame({"ID": range(len(Pfin)), "P_ml": ml_p, "P_dl": dlP, "P_fin": Pfin, "pred": predb}).to_csv(out_path + ".probs.csv", index=False)
    print("BEST: acc=%.4f %s w_ml=%.2f%s" % (best_acc, best_name, best_w, note))
    print("wrote", out_path, out.TARGET.value_counts().to_dict())

if __name__ == "__main__":
    main()
