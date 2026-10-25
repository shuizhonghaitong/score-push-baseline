"""Build PSD submissions from GRAI3 API (b, c) predictions per test photo."""
import json, os, re, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

D = [float(c) for c in C.SUPPORT_COLS]  # 11 sieve sizes, mm

def weibull_F(b, c, d=D):
    d = np.asarray(d, float)
    return 100.0 * (1.0 - np.exp(-np.power(d / b, c)))

SAMPLE_MAP = {
    "HPC_Airbus BS6-3": ["HPC_Airbus BS6-3"],
    "HPC_Audorfring": ["HPC_Audorfring"],
    "HPC_Muenster_BS6_9_0-10m": ["HPC_Münster_BS6_9,0-10m", "HPC_Muenster_BS6_9,0-10m"],
    "HPC_Airbus BS10-4bis7": ["HPC_Airbus BS10-4bis7"],
    "HPC_Airbus BS12-5bis8": ["HPC_Airbus BS12-5bis8"],
    "HPC_Kleinkummerfeld 2-2": ["HPC_Kleinkummerfeld 2-2"],
    "HPC_Kleinkummerfeld 2-3": ["HPC_Kleinkummerfeld 2-3"],
    "HPC_Kleinkummerfeld 9-4": ["HPC_Kleinkummerfeld 9-4"],
    "HPC_Kleinkummerfeld 18-3": ["HPC_Kleinkummerfeld 18-3"],
    "HPC_Testfeld Lidl WHV": ["HPC_Testfeld Lidl WHV"],
}

def photo_to_sample(fname):
    base = re.sub(r"\s*\(\d+\)\.JPG$", "", os.path.basename(fname))
    base = re.sub(r"\.JPG$", "", base, flags=re.I)
    base = re.sub(r"^iPhone[\d_]*", "", base)
    for sub, keys in SAMPLE_MAP.items():
        for k in keys:
            if base == k:
                return sub
    return None

def main():
    preds = json.load(open("state/grai3_preds.json"))
    preds = [p for p in preds if "b" in p]
    per = {}
    for p in preds:
        s = photo_to_sample(p["file"])
        if s is None:
            print("UNMAPPED:", p["file"]); continue
        per.setdefault(s, []).append(p)
    sub = pd.read_csv(f"{C.DATA_DIR}/sample_submission.csv")
    out = {}
    for s in sub["sample_id"]:
        ps = per.get(s, [])
        if not ps:
            print("MISSING photos for", s); continue
        bs = np.array([p["b"] for p in ps]); cs = np.array([p["c"] for p in ps])
        Fphoto = np.stack([weibull_F(b, c) for b, c in zip(bs, cs)])
        out[s] = {
            "Fmean": Fphoto.mean(0),
            "Fmed": np.median(Fphoto, 0),
            "bc_mean": weibull_F(bs.mean(), cs.mean()),
            "bc_med": weibull_F(np.median(bs), np.median(cs)),
        }
        print(f"{s:28s} n={len(ps)} b~{bs.mean():.3f}/{np.median(bs):.3f} c~{cs.mean():.2f}/{np.median(cs):.2f}")
    for variant in ("Fmean", "Fmed", "bc_mean", "bc_med"):
        rows = []
        for s in sub["sample_id"]:
            psd = C.enforce_rows([out[s][variant]])[0]
            rows.append([s] + [f"{v:.4f}" for v in psd])
        fn = f"submissions/grai3_{variant}.csv"
        with open(fn, "w") as f:
            f.write("sample_id," + ",".join(C.SUPPORT_COLS) + "\n")
            for r in rows:
                f.write(",".join(r) + "\n")
        print("wrote", fn)

if __name__ == "__main__":
    main()
