import pandas as pd, numpy as np
pd.set_option('display.width', 250)
tr = pd.read_csv('data/train.csv', dtype={'Zip':str})
te = pd.read_csv('data/test.csv', dtype={'Zip':str})
tr['mon'] = pd.to_datetime(tr['Month']).dt.month
feats = ['SampleSize','RentIndex','AnnualMedianHouseholdIncome','OwnerPercent','RenterPercent','MedianHomeLengthofResidence','AvgNumberofChildren','AvgNumberofAdults','AvgHouseholdSize','AvgGenerationsinHousehold','YoungAdultinHouseholdPercent','SeniorAdultinHouseholdPercent']
a = tr[tr.mon==5].set_index('Zip'); b = tr[tr.mon==6].set_index('Zip')
j = a[['AffordabilityPercentageTrue']].join(b[['AffordabilityPercentageTrue']].rename(columns={'AffordabilityPercentageTrue':'y6'}), how='inner')
j = j.rename(columns={'AffordabilityPercentageTrue':'y5'})
d = j.y6 - j.y5
print("month-over-month diff (june-may): mean", round(d.mean(),3), "std", round(d.std(),3), "abs mean", round(d.abs().mean(),3), "median abs", round(d.abs().median(),3), "p90 abs", round(d.abs().quantile(.9),3))
print("corr(june,may) =", round(j.y5.corr(j.y6),4), " n=", len(j))
f5, f6 = a[feats], b[feats]
rel = (f6-f5).abs()/f5.abs().replace(0,np.nan)
print("mean relative change may->june:\n", rel.mean().sort_values().round(4).to_string())
print("\ntrain SampleSize:\n", tr['SampleSize'].describe().round(1).to_string())
print("test SampleSize:\n", te['SampleSize'].describe().round(1).to_string())
print("owner+renter range:", round((tr.OwnerPercent+tr.RenterPercent).min(),2), round((tr.OwnerPercent+tr.RenterPercent).max(),2))
print("RentIndex missing (train):", tr.RentIndex.isna().sum(), "test:", te.RentIndex.isna().sum())
print("states missing rent (train):", tr[tr.RentIndex.isna()].State.value_counts().to_dict())
print("\ntarget by state (train):")
print(tr.groupby('State').AffordabilityPercentageTrue.agg(['count','mean','std']).sort_values('count',ascending=False).round(1).to_string())
seen = set(a.index)
newte = te[~te.Zip.isin(seen)]
print("\nnew zips in test:", len(newte), " seen in test:", len(te)-len(newte))
print("new zip states:", newte.State.value_counts().to_dict())
# binomial noise check: relationship between |y6-y5| and sample size
j['n'] = tr[tr.mon==5].set_index('Zip').SampleSize
q = pd.qcut(j['n'], 5, labels=False)
print("\n|y6-y5| by SampleSize quintile:\n", j.assign(q=q, ad=d.abs()).groupby('q').agg(n=('ad','size'), mean_abs=('ad','mean')).round(3))
