"""2-seed DL + ML blend: OOF grid search over DL variants, writes test prob CSV.
usage: twoseed_blend.py <oof_ml> <oof_A> <oof_B> <test_ml> <test_A> <test_B> <out_probs.csv>
  test A/B expected to have cols P or p_ms/p_sc/p_attn; uses matching column set.
  prints top candidates; writes out_probs.csv with cols ID,file,P_fin
"""
import sys
import numpy as np
import pandas as pd

def bal_acc(P, tfile, ty):
    accs = []
    for f in np.unique(tfile):
        m = tfile == f
        k2 = int(m.sum()) // 2
        pred = np.zeros(int(m.sum()), int)
        pred[np.argpartition(-P[m], k2)[:k2]] = 1
        accs.append(float((pred == ty[m]).mean()))
    return float(np.mean(accs))

oof_ml_p, oof_A_p, oof_B_p, tml_p, tA_p, tB_p, out_p = sys.argv[1:8]
oofml = pd.read_csv(oof_ml_p)
A = pd.read_csv(oof_A_p)
B = pd.read_csv(oof_B_p)
ty = A["y"].to_numpy()
tfile = A["file"].to_numpy()
assert (A[["trial", "file", "y"]].values == B[["trial", "file", "y"]].values).all()

def cols(df, prefix):
    out = {}
    if "P" in df.columns:
        out["mean"] = df["P"].to_numpy()
    if "p_ms" in df.columns:
        for c in ("ms", "sc", "attn"):
            out[c] = df["p_" + c].to_numpy()
        out["mean"] = df[["p_ms", "p_sc", "p_attn"]].mean(axis=1).to_numpy()
    if "P_fb" in df.columns and df["P_fb"].notna().any():
        out["fb"] = df["P_fb"].to_numpy()
    return out

cA, cB = cols(A, "A"), cols(B, "b")
common = [c for c in cA if c in cB]
variants = {}
for c in common:
    variants[f"{c}_A"] = cA[c]
    variants[f"{c}_B"] = cB[c]
    variants[f"{c}_avg"] = 0.5 * (cA[c] + cB[c])

tml = pd.read_csv(tml_p)
tA = pd.read_csv(tA_p)
tB = pd.read_csv(tB_p)
assert len(tml) == len(tA) == len(tB) == 360
mlP = tml["P"].to_numpy()

def tcol(df, c):
    if c == "mean":
        if "P" in df.columns: return df["P"].to_numpy()
        return df[["p_ms", "p_sc", "p_attn"]].mean(axis=1).to_numpy()
    if c == "fb": return df["P_fb"].to_numpy()
    for cand in (f"P_{c}", f"p_{c}", f"P_{c.upper()}"):
        if cand in df.columns:
            return df[cand].to_numpy()
    raise KeyError(f"no column for {c} in {list(df.columns)}")

n = min(len(oofml), len(ty))
res = []
for name, v in variants.items():
    base = name.split("_A")[0].split("_B")[0].split("_avg")[0].split("4")[0]
    for w in np.arange(0.0, 1.0 + 1e-9, 0.05):
        res.append((round(bal_acc(w * oofml["P"].to_numpy()[:n] + (1 - w) * v[:n], tfile[:n], ty[:n]), 4), name, round(float(w), 2)))
res.sort(reverse=True)
print("top OOF 2-seed blend candidates (acc, variant, w_ml):")
for r in res[:10]:
    print("  ", r)
res2 = sorted(res, key=lambda r: (-r[0], r[1]))
best_acc, best_name, best_w = res2[0]
if best_name.endswith("_A"): dlP = tcol(tA, best_name[:-2])
elif best_name.endswith("_B"): dlP = tcol(tB, best_name[:-2])
else:
    dlP = 0.5 * (tcol(tA, best_name[:-4]) + tcol(tB, best_name[:-4]))
Pfin = best_w * mlP + (1 - best_w) * dlP
out = pd.DataFrame({"ID": range(360), "file": tml["file"].to_numpy() if "file" in tml.columns else tA["file"].to_numpy(), "P_fin": Pfin})
out.to_csv(out_p, index=False)
print("BEST acc=%.4f variant=%s w_ml=%.2f -> %s" % (best_acc, best_name, best_w, out_p))
