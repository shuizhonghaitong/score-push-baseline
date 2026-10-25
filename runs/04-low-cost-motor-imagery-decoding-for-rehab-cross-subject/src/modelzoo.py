from __future__ import annotations

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import make_pipeline
import xgboost as xgb
import lightgbm as lgb


def make_model(kind: str, **kw):
    if kind == 'lr':
        return make_pipeline(StandardScaler(),
                             LogisticRegression(C=kw.get('C', 1.0), max_iter=2000))
    if kind == 'svm':
        return make_pipeline(StandardScaler(),
                             CalibratedClassifierCV(
                                 SVC(kernel=kw.get('kernel', 'rbf'),
                                     C=kw.get('C', 1.0),
                                     class_weight=kw.get('cw', 'balanced')),
                                 method='sigmoid', cv=3))
    if kind == 'xgb':
        return xgb.XGBClassifier(
            n_estimators=kw.get('n', 300), learning_rate=0.05, max_depth=3,
            subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
            eval_metric='logloss', n_jobs=8, random_state=0)
    if kind == 'lgbm':
        return lgb.LGBMClassifier(
            n_estimators=kw.get('n', 300), learning_rate=0.05, num_leaves=15,
            subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
            n_jobs=8, random_state=0, verbose=-1)
    raise ValueError(kind)


class Ensemble:
    def __init__(self, kinds):
        self.models = [make_model(k, **p) for k, p in kinds]

    def fit(self, X, y):
        for m in self.models:
            m.fit(X, y)

    def predict_proba(self, X):
        return np.mean([m.predict_proba(X)[:, 1] for m in self.models], axis=0)

    def predict(self, X):
        return (self.predict_proba(X) > 0.5).astype(int)
