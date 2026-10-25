"""Subject-grouped cross-validation over all training subjects."""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np

from load_data import iter_dataset, load_trial_windows
from features import trial_features, FEATURE_NAMES


def build_train(root: str, verbose=False):
    """Returns list of (subject, feats (n_trials, F), labels (n_trials,))."""
    per_subject = {}
    for subject, fname, path in iter_dataset(root, 'train'):
        trials = load_trial_windows(path)
        for t in trials:
            label = 1 if t['label'] == 'move' else 0
            per_subject.setdefault(subject, []).append((trial_features(t), label))
    out = []
    for subject in sorted(per_subject):
        feats = np.stack([f for f, _ in per_subject[subject]])
        labels = np.array([l for _, l in per_subject[subject]])
        if verbose:
            print(subject, feats.shape, labels.mean())
        out.append((subject, feats, labels))
    return out


def grouped_cv(root: str, make_model, n_folds=5, seed=0):
    data = build_train(root, verbose=True)
    subjects = [d[0] for d in data]
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(subjects))
    folds = np.array_split(order, n_folds)
    fold_accs = []
    for k, fold in enumerate(folds):
        val_idx = set(fold)
        val = [data[i] for i in fold if i in val_idx]
        tr = [data[i] for i in range(len(data)) if i not in val_idx]
        Xtr = np.concatenate([d[1] for d in tr])
        ytr = np.concatenate([d[2] for d in tr])
        Xva = np.concatenate([d[1] for d in val])
        yva = np.concatenate([d[2] for d in val])
        model = make_model()
        t0 = time.time()
        model.fit(Xtr, ytr)
        pred = model.predict(Xva)
        acc = (pred == yva).mean()
        fold_accs.append(acc)
        print(f'fold {k}: subjects={[data[i][0] for i in fold]} acc={acc:.4f} '
              f'pred_move={pred.mean():.3f} true_move={yva.mean():.3f} ({time.time()-t0:.1f}s)')
    print(f'MEAN acc = {np.mean(fold_accs):.4f} +/- {np.std(fold_accs):.4f}')
    return np.mean(fold_accs)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', required=True)
    args = p.parse_args()
    data = build_train(args.root, verbose=True)
    X = np.concatenate([d[1] for d in data])
    y = np.concatenate([d[2] for d in data])
    print('total trials:', X.shape, 'move fraction:', y.mean())
    np.save('outputs/train_feats.npy', X)
    np.save('outputs/train_labels.npy', y)
    np.save('outputs/train_subjects.npy', np.repeat(
        [d[0] for d in data], [d[1].shape[0] for d in data]))
