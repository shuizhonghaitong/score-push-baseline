"""Blend classical, DL and anchor predictions into a final submission.

Usage: python ensemble.py --classical cls.csv --dl dl.csv [--anchor anchor.csv] --out sub.csv
Evaluates against the two released GT rows (from sample_submission) when possible.
"""
import argparse
import math
import pandas as pd

TAU = {"pa_deg": 6.0, "fl_mm": 12.0, "mt_mm": 3.0}

def norm_mae(gt, pred):
    s = 0.0
    for c, t in TAU.items():
        s += (gt[c] - pred[c]).abs().mean() / t
    return s / 3.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classical", required=True)
    ap.add_argument("--dl")
    ap.add_argument("--anchor", default="dltrack_anchor.csv")
    ap.add_argument("--gt2")
    ap.add_argument("--out", required=True)
    ap.add_argument("--w-cls", type=float, default=1.0)
    ap.add_argument("--w-dl", type=float, default=1.0)
    ap.add_argument("--w-anchor", type=float, default=1.0)
    a = ap.parse_args()

    cls = pd.read_csv(a.classical)
    anc = pd.read_csv(a.anchor)
    df = cls.merge(anc, on="image_id", suffixes=("_c", "_a"))
    if a.dl:
        dl = pd.read_csv(a.dl)
        df = df.merge(dl, on="image_id", suffixes=("", "_d"))

    out = pd.DataFrame({"image_id": df.image_id})
    for col, c1, c2, c3 in [("pa_deg", "pa_deg_c", "pa_deg_a", "pa_deg_d"),
                            ("mt_mm", "mt_mm_c", "mt_mm_a", "mt_mm_d"),
                            ("fl_mm", "fl_mm_c", "fl_mm_a", "fl_mm_d")]:
        vals = []
        for r in df.itertuples():
            cand = [(a.w_cls, getattr(r, c1)), (a.w_anchor, getattr(r, c2))]
            if a.dl:
                cand.append((a.w_dl, getattr(r, c3)))
            # keep candidates finite and in physiological range
            lo, hi = {"pa_deg": (5, 45), "mt_mm": (10, 50), "fl_mm": (30, 200)}[col]
            okc = [(w, v) for w, v in cand if v == v and lo <= v <= hi]
            if not okc:
                okc = cand
            tw = sum(w for w, _ in okc)
            vals.append(sum(w * v for w, v in okc) / tw)
        out[col] = vals

    # geometric FL re-coupling: pull fl toward mt/sin(pa)
    geo = out.mt_mm / out.pa_deg.map(lambda p: max(math.sin(math.radians(p)), 1e-6))
    r = (geo / out.fl_mm).clip(0.7, 1.4)
    out["fl_mm"] = (out.fl_mm * r).clip(30, 200)
    out["pa_deg"] = out.pa_deg.clip(5, 45)
    out["mt_mm"] = out.mt_mm.clip(10, 50)
    out["fl_mm"] = out.fl_mm.clip(30, 200)
    out = out[["image_id", "pa_deg", "fl_mm", "mt_mm"]]
    out.to_csv(a.out, index=False)
    print(out.head(5).to_string())
    print("describe:\n", out[["pa_deg", "fl_mm", "mt_mm"]].describe().to_string())

    if a.gt2:
        gt = pd.read_csv(a.gt2, sep=";").head(2)
        m = gt.merge(out, on="image_id", suffixes=("_t", "_p"))
        s = 0.0
        for c, t in TAU.items():
            s += (m[f"{c}_p"] - m[f"{c}_t"]).abs().mean() / t
        print("GT2 score (this blend): %.4f" % (s / 3))
        for _, r2 in m.iterrows():
            print(f"  {r2.image_id}: true {r2.pa_deg_t}/{r2.fl_mm_t}/{r2.mt_mm_t} "
                  f"pred {r2.pa_deg_p:.2f}/{r2.fl_mm_p:.2f}/{r2.mt_mm_p:.2f}")

if __name__ == "__main__":
    main()
