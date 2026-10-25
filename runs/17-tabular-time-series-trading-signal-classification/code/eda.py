import pandas as pd, numpy as np, glob, sys, json
D = "/mnt/disk3/liuyutong_home/workspace/skill/score-push/runs/17-tabular-time-series-trading-signal-classification/data"
out = {}
for f in sorted(glob.glob(D + "/**/*.csv", recursive=True)):
    df = pd.read_csv(f)
    for c in ('datetime','datetime_utc','date'):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c])
    print("="*100)
    print(f, df.shape)
    print("cols:", list(df.columns))
    print(df.head(3).to_string())
    print(df.dtypes.to_string())
    if 'Signal' in df.columns:
        print("Signal mean:", df['Signal'].mean(), "pos:", int(df['Signal'].sum()))
    if 'datetime' in df.columns or 'datetime_utc' in df.columns:
        dcol = 'datetime' if 'datetime' in df.columns else 'datetime_utc'
        print(dcol, df[dcol].iloc[0], "->", df[dcol].iloc[-1], "monotonic:", (df[dcol].diff().dropna() > pd.Timedelta(0)).all())
    num = df.select_dtypes('number').columns
    print(df[num].describe().T.to_string())
    print("nulls:", int(df.isna().sum().sum()), "dup rows:", int(df.duplicated().sum()))
