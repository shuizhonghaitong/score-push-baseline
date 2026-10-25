import os, glob, json, shutil
import pandas as pd
import numpy as np

SRC = "/kaggle/input/tabular-time-series-trading-signal-classification"
OUT = "/kaggle/outputs"
os.makedirs(OUT, exist_ok=True)

files = sorted(glob.glob(SRC + "/**/*", recursive=True))
files = [f for f in files if os.path.isfile(f)]
print("files:", files)
summary = {}
for f in files:
    name = os.path.relpath(f, SRC)
    if not name.endswith(".csv"):
        print("skip", name)
        continue
    df = pd.read_csv(f)
    dest = os.path.join(OUT, name.replace("/", "_"))
    df.to_csv(dest, index=False)
    summary[name] = {
        "shape": list(df.shape),
        "cols": list(df.columns),
        "dtypes": {c: str(t) for c, t in df.dtypes.items()},
    }
    print("=" * 80)
    print(name, df.shape)
    print(df.head(2).to_string())
    if "Signal" in df.columns:
        print("Signal pos:", int(df["Signal"].sum()), "mean:", float(df["Signal"].mean()))
    for dcol in ["datetime", "datetime_utc"]:
        if dcol in df.columns:
            print(dcol, df[dcol].iloc[0], "->", df[dcol].iloc[-1])
            break

json.dump(summary, open(os.path.join(OUT, "summary.json"), "w"), indent=1)
print("DONE datafetch")
