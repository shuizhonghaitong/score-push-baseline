"""Build submission CSV for the cross-subject test set.

Order: test files sorted by filename; epochs in file order (cue_start markers).
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

sys.path.insert(0, '.')
from load_data import iter_dataset, load_trial_windows
from features import trial_features


def train_model(Xtr, ytr):
    models = [make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=3000))
              for C in (0.1, 0.3, 1.0)]
    for m in models:
        m.fit(Xtr, ytr)
    return models


def predict(models, X):
    return np.mean([m.predict_proba(X)[:, 1] for m in models], axis=0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--data', required=True)
    p.add_argument('--train-feats', required=True)
    p.add_argument('--train-labels', required=True)
    p.add_argument('--out', required=True)
    args = p.parse_args()

    Xtr = np.load(args.train_feats)
    ytr = np.load(args.train_labels)
    models = train_model(Xtr, ytr)

    files = []
    for subject, fname, path in iter_dataset(args.data, 'test'):
        files.append((fname, path))
    files.sort(key=lambda t: t[0])
    trials_all = []
    for fname, path in files:
        t = load_trial_windows(path)
        print(fname, len(t), 'trials')
        trials_all.extend(t)
    Xt = np.stack([trial_features(t) for t in trials_all])
    proba = predict(models, Xt)

    rows = [{'ID': str(i), 'TARGET': 'move' if proba[i] > 0.5 else 'rest'}
            for i in range(len(trials_all))]
    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(f'wrote {args.out}: {len(df)} rows, move={(df.TARGET == "move").mean():.3f}')
    np.save(args.out.replace('.csv', '_proba.npy'), proba)


if __name__ == '__main__':
    main()
