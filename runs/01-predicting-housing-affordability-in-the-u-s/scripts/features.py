import numpy as np
import pandas as pd

NUM_FEATS = ['SampleSize','RentIndex','AnnualMedianHouseholdIncome','OwnerPercent','RenterPercent',
             'MedianHomeLengthofResidence','AvgNumberofChildren','AvgNumberofAdults','AvgHouseholdSize',
             'AvgGenerationsinHousehold','YoungAdultinHouseholdPercent','SeniorAdultinHouseholdPercent']
CAT_FEATS = ['State','Metro']
TARGET = 'AffordabilityPercentageTrue'

def add_derived(df):
    df = df.copy()
    inc = df['AnnualMedianHouseholdIncome']
    rent = df['RentIndex']
    df['RentMonthly_x12'] = rent * 12.0
    df['RentIncomeRatio'] = (rent * 12.0) / inc.replace(0, np.nan)
    df['RentIncomeRatio30'] = df['RentIncomeRatio'] * 30.0
    df['LogRent'] = np.log1p(rent.clip(lower=0))
    df['LogIncome'] = np.log1p(inc.clip(lower=0))
    df['OwnerXIncome'] = df['OwnerPercent'] / 100.0 * inc
    df['InvSqrtN'] = 1.0 / np.sqrt(df['SampleSize'])
    df['LogN'] = np.log1p(df['SampleSize'])
    df['ChildrenXIncome'] = df['AvgNumberofChildren'] * inc
    df['AdultsXIncome'] = df['AvgNumberofAdults'] * inc
    df['ResidenceXOwner'] = df['MedianHomeLengthofResidence'] * df['OwnerPercent']
    df['SeniorXIncome'] = df['SeniorAdultinHouseholdPercent'] * inc
    return df

def build(df, with_month=False):
    d = add_derived(df)
    cols = NUM_FEATS + ['RentMonthly_x12','RentIncomeRatio','RentIncomeRatio30','LogRent','LogIncome',
                        'OwnerXIncome','InvSqrtN','LogN','ChildrenXIncome','AdultsXIncome',
                        'ResidenceXOwner','SeniorXIncome']
    cols = [c for c in cols if c in d.columns]
    X = d[cols].copy()
    meta = d[['Zip','State','Month','SampleSize']]
    return X, meta
