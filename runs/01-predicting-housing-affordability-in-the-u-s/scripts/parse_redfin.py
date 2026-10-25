import warnings; warnings.filterwarnings('ignore')
from pathlib import Path
import numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parent.parent
TSV = ROOT/'external/redfin_cities/city_market_tracker.tsv000'
OUT = ROOT/'external/redfin_city_latest.csv'
cols = ['PERIOD_BEGIN','REGION_TYPE','REGION','CITY','STATE','STATE_CODE','PROPERTY_TYPE',
        'MEDIAN_SALE_PRICE','MEDIAN_SALE_PRICE_MOM','MEDIAN_SALE_PRICE_YOY',
        'MEDIAN_LIST_PRICE','MEDIAN_PPSF','MEDIAN_PPSF_YOY','INVENTORY','MONTHS_OF_SUPPLY','HOMES_SOLD']
print('streaming', TSV, TSV.stat().st_size/1e9, 'GB')
with open(TSV) as f:
    header = f.readline().rstrip('\n').split('\t')
    header = [h.strip('"') for h in header]
    idx = [header.index(c) for c in cols]
    pk_max = {}
    pk_rows = {}
    n = 0
    for line in f:
        parts = line.rstrip('\n').split('\t')
        if len(parts) < len(header): continue
        if parts[idx[1]].strip('"') != 'place': continue
        pk = parts[idx[2]] + '|' + parts[idx[4]] + '|' + parts[idx[5]]
        pb = parts[idx[0]]
        n += 1
        if pb >= pk_max.get(pk, ''):
            pk_max[pk] = pb
            pk_rows[pk] = [parts[i] for i in idx]
        if n % 5_000_000 == 0:
            print(n, 'rows, ', len(pk_max), 'keys', flush=True)
df = pd.DataFrame.from_dict(pk_rows, orient='index', columns=cols)
df = df.reset_index().rename(columns={'index':'pk'})
print('city keys:', len(df))
for c in cols:
    if c not in ('PERIOD_BEGIN','REGION_TYPE','REGION','CITY','STATE','STATE_CODE','PROPERTY_TYPE'):
        df[c] = pd.to_numeric(df[c], errors='coerce')
print(df[['PERIOD_BEGIN','PROPERTY_TYPE','MEDIAN_SALE_PRICE']].describe().round(1).to_string())
print('property types:', df.PROPERTY_TYPE.value_counts().to_dict())
print('periods:', sorted(df.PERIOD_BEGIN.unique())[-6:])
df.to_csv(OUT, index=False)
print('saved', OUT)
