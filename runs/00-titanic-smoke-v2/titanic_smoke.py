"""Titanic smoke end-to-end: simple RandomForest baseline, holdout validation.

Single source for local runs and the Kaggle private CPU notebook.
Data dir: /kaggle/input/competitions/titanic on Kaggle, ./data locally.
Output dir: /kaggle/working on Kaggle, ./outputs locally.
Deterministic (seed 42). CPU only.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

SEED = 42
FEATURES = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare",
            "Embarked", "Title", "FamilySize", "IsAlone", "CabinDeck"]


def add_features(df):
    df = df.copy()
    df["Title"] = df["Name"].str.extract(r", (\w+)\.", expand=False)
    df.loc[df["Title"].isin(["Mlle", "Ms"]), "Title"] = "Miss"
    df.loc[df["Title"].isin(["Mme"]), "Title"] = "Mrs"
    rare = df["Title"].value_counts()
    df.loc[~df["Title"].isin(rare[rare > 4].index), "Title"] = "Rare"
    df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
    df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
    df["CabinDeck"] = df["Cabin"].str[0]
    return df


def make_pipeline():
    return Pipeline([
        ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestClassifier(n_estimators=400, max_depth=6,
                                         min_samples_leaf=3, random_state=SEED,
                                         n_jobs=1)),
    ])


def resolve_paths():
    kaggle_data = Path("/kaggle/input/competitions/titanic")
    if kaggle_data.is_dir():
        data_dir, out_dir = kaggle_data, Path("/kaggle/working")
    else:
        here = Path(__file__).resolve().parent
        data_dir, out_dir = here / "data", here / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    return data_dir, out_dir


def main():
    data_dir, out_dir = resolve_paths()
    train = pd.read_csv(data_dir / "train.csv")
    test = pd.read_csv(data_dir / "test.csv")
    y = train["Survived"].astype(int)

    xtr = add_features(train)[FEATURES]
    xte = add_features(test)[FEATURES]

    idx_tr, idx_val = train_test_split(np.arange(len(train)), test_size=0.2,
                                       random_state=SEED, stratify=y)
    pipe = make_pipeline()
    pipe.fit(xtr.iloc[idx_tr], y.iloc[idx_tr])
    val_acc = float((pipe.predict(xtr.iloc[idx_val]) == y.iloc[idx_val]).mean())
    print(f"holdout (80/20, seed {SEED}) validation accuracy = {val_acc:.4f}")

    pipe.fit(xtr, y)
    proba = pipe.predict_proba(xte)[:, 1]
    submission = pd.DataFrame({
        "PassengerId": test["PassengerId"].astype(int),
        "Survived": (proba >= 0.5).astype(int),
    })
    sub_path = out_dir / "submission_smoke_v2.csv"
    submission.to_csv(sub_path, index=False)

    report = {
        "script": "titanic_smoke.py",
        "seed": SEED,
        "model": "RandomForestClassifier(400, depth=6, leaf>=3)",
        "features": FEATURES,
        "validation_protocol": "stratified 80/20 holdout",
        "holdout_accuracy": round(val_acc, 4),
        "n_train": len(train),
        "n_test": len(test),
        "predicted_survival_rate": round(float(submission["Survived"].mean()), 4),
    }
    (out_dir / "validation_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print(f"wrote {sub_path} ({len(submission)} rows)")


if __name__ == "__main__":
    main()
