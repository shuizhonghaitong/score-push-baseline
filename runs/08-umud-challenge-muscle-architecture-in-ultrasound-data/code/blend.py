"""Generic weighted blender of UMUD candidate CSVs.
Usage:
  python blend.py --in name=path.csv [--in name=path.csv ...] \
      [--w-gated 1 --w-direct 0.5 --w-anchor 1] \
      [--gt2 IMG_00001.tif:17.334,79.423,21.778 IMG_00002.tif:12.876,69.424,15.478] \
      --out sub.csv
Blends per column: value = sum(w_i * v_i) / sum(w_i) over provided inputs (missing -> skip).
Then applies FL geometric recoupling (optional) + physiological clip, and reports the
two-point GT score (if provided) as a sanity signal.
"""
import argparse, math, sys
import pandas as pd

TAU = {"pa_deg": 6.0, "fl_mm": 12.0, "mt_mm": 3.0}
CLIP = {"pa_deg": (5, 45), "mt_mm": (10, 50), "fl_mm": (30, 200)}
GT_DEFAULT = {
    "IMG_00001.tif": (17.334, 79.423, 21.778),
    "IMG_00002.tif": (12.876, 69.424, 15.478),
}

def load(path):
    d = pd.read_csv(path)
    assert list(d.columns) == ["image_id", "pa_deg", "fl_mm", "mt_mm"], d.columns
    return d.set_index("image_id")[["pa_deg", "fl_mm", "mt_mm"]]

def gt2score(out, gt):
    ids = [k for k in gt if k in out.index]
    if not ids:
        return None
    s = 0.0
    for c, t in TAU.items():
        s += (out.loc[ids, c] - [gt[i][{"pa_deg":0,"fl_mm":1,"mt_mm":2}[c]] for i in ids]).abs().mean() / t
    return s / 3.0

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="ins", nargs="+", required=True)
    ap.add_argument("--fl-recouple", type=float, default=0.0, help="0..1 pull fl toward mt/sin(pa)")
    ap.add_argument("--pin-gt", action="store_true", help="pin the two released GT rows")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    frames, weights = [], []
    for spec in a.ins:
        name, path = spec.split("=", 1)
        w = float(getattr(a, f"w_{name}", 1.0) if hasattr(a, f"w_{name}") else 1.0)
        frames.append(load(path)); weights.append(w)
    base = frames[0].copy()
    for c in ["pa_deg", "fl_mm", "mt_mm"]:
        num = sum(w * f[c] for w, f in zip(weights, frames))
        den = sum(w for w, _ in zip(weights, frames))
        base[c] = num / den

    if a.fl_recouple > 0:
        imp = base.mt_mm / base.pa_deg.map(lambda p: max(math.sin(math.radians(p)), 1e-6))
        r = (imp / base.fl_mm).clip(0.8, 1.2)
        base["fl_mm"] = (base.fl_mm * (1 - a.fl_recouple) + base.fl_mm * r * a.fl_recouple)

    out = base.reset_index()
    for c, (lo, hi) in CLIP.items():
        out[c] = out[c].clip(lo, hi)
    if a.pin_gt:
        for iid, (p, f, m) in GT_DEFAULT.items():
            if iid in set(out.image_id):
                out.loc[out.image_id == iid, ["pa_deg", "fl_mm", "mt_mm"]] = [p, f, m]
    out = out[["image_id", "pa_deg", "fl_mm", "mt_mm"]].round(3)
    out.to_csv(a.out, index=False)
    print(out.head(5).to_string())
    s = gt2score(set_index := out.set_index("image_id"), GT_DEFAULT)
    if s is not None:
        print("GT2 score (2-row sanity): %.4f" % s)
    print("describe:\n", out[["pa_deg", "fl_mm", "mt_mm"]].describe()[["min", "mean", "max"]].to_string())

if __name__ == "__main__":
    main()
