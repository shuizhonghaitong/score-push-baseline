"""Aggregate features, LOOCV model selection, produce submission."""
import os, sys, json, time
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge, BayesianRidge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import ElasticNet
from sklearn.multiclass import OneVsRestClassifier  # noqa
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import MultiTaskElasticNet, MultiTaskLasso
from sklearn.cross_decomposition import PLSRegression

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C


# ---------------- Y transforms ----------------
class CumulativeTransform:
    name = "cum"
    def encode(self, Y): return Y.copy()
    def decode(self, Yt): return C.enforce_rows(Yt)


class DeltaTransform:
    name = "delta"
    def encode(self, Y): return np.diff(Y, axis=1, prepend=0)
    def decode(self, Dt):
        D = np.clip(Dt, 0, None)
        s = D.sum(axis=1, keepdims=True); s = np.where(s <= 1e-9, 1.0, s)
        F = np.cumsum(D / s * 100.0, axis=1)
        return C.enforce_rows(F)


TRANSFORMS = [CumulativeTransform(), DeltaTransform()]


def make_models():
    return {
        "ridge_5": lambda: Ridge(alpha=5.0),
        "ridge_15": lambda: Ridge(alpha=15.0),
        "ridge_40": lambda: Ridge(alpha=40.0),
        "ridge_120": lambda: Ridge(alpha=120.0),
        "ridge_400": lambda: Ridge(alpha=400.0),
        "enet_0.05": lambda: MultiOutputRegressor(ElasticNet(alpha=0.05, l1_ratio=0.5, max_iter=20000, tol=1e-4)),
        "enet_0.1_l03": lambda: MultiOutputRegressor(ElasticNet(alpha=0.1, l1_ratio=0.3, max_iter=20000, tol=1e-4)),
        "mtlasso_0.05": lambda: MultiTaskLasso(alpha=0.05, max_iter=20000, tol=1e-4),
        "mtlasso_0.1": lambda: MultiTaskLasso(alpha=0.1, max_iter=20000, tol=1e-4),
        "mtenet_0.05": lambda: MultiTaskElasticNet(alpha=0.05, l1_ratio=0.5, max_iter=20000, tol=1e-4),
        "bayesridge": lambda: MultiOutputRegressor(BayesianRidge()),
        "pls_3": lambda: PLSRegression(n_components=3),
        "pls_5": lambda: PLSRegression(n_components=5),
        "svr_rbf": lambda: MultiOutputRegressor(SVR(C=10.0, epsilon=0.1, kernel="rbf")),
    }


class YScaled:
    def __init__(self, fn, tf):
        self.fn = fn; self.tf = tf
    def fit(self, X, Y):
        Yt = self.tf.encode(Y)
        self.sc = StandardScaler(); Ys = self.sc.fit_transform(Yt)
        self.m = self.fn(); self.m.fit(X, Ys); return self
    def predict(self, X):
        p = self.m.predict(X)
        if p.ndim == 1: p = p.reshape(1, -1)
        return self.tf.decode(self.sc.inverse_transform(p))


def loocv_oof(X, Y, fn, tf):
    n = X.shape[0]; oof = np.zeros_like(Y)
    for i in range(n):
        tr = np.ones(n, bool); tr[i] = False
        m = YScaled(fn, tf); m.fit(X[tr], Y[tr])
        oof[i] = m.predict(X[i:i+1])[0]
    return oof


def loocv_score(X, Y, fn, tf):
    return C.emd_trap_mean(Y, loocv_oof(X, Y, fn, tf))


def aggregate(hc_tr, hc_te, hc_keys, hc_tr_ids, hc_te_ids, dino_tr, dino_te):
    """Group per-photo rows by sample id into per-sample mean/std."""
    import common as C2
    def agg(a, ids, keys):
        df = pd.DataFrame(a, columns=keys)
        df["sid"] = ids
        mean = df.groupby("sid")[keys].mean().add_suffix("_m")
        std = df.groupby("sid")[keys].std().add_suffix("_s").fillna(0.0)
        return pd.concat([mean, std], axis=1).reset_index()
    out = {}
    out["hc_tr"] = agg(hc_tr, hc_tr_ids, hc_keys); out["hc_te"] = agg(hc_te, hc_te_ids, hc_keys)
    if dino_tr is not None:
        dk = [f"dino_{i}" for i in range(dino_tr.shape[1])]
        out["dino_tr"] = pd.DataFrame(dino_tr, columns=dk); out["dino_tr"]["sid"] = hc_tr_ids
        out["dino_te"] = pd.DataFrame(dino_te, columns=dk); out["dino_te"]["sid"] = hc_te_ids
        # per-sample mean of dino
        out["dino_tr"] = out["dino_tr"].groupby("sid", as_index=False)[dk].mean()
        out["dino_te"] = out["dino_te"].groupby("sid", as_index=False)[dk].mean()
    return out


def build_dataset():
    state = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "state")
    hc_file = os.environ.get("FEAT_HC", "feat_handcrafted_tl1600.npz")
    dd_file = os.environ.get("FEAT_DINO", "feat_dino_s518.npz")
    hc = np.load(f"{state}/{hc_file}", allow_pickle=True)
    dd = np.load(f"{state}/{dd_file}", allow_pickle=True)
    hc_tr, hc_te, keys = hc["tr"], hc["te"], list(hc["keys"])
    hc_tr_ids, hc_te_ids = hc["tr_ids"], hc["te_ids"]
    dino_tr, dino_te = dd["tr"], dd["te"]
    A = aggregate(hc_tr, hc_te, keys, hc_tr_ids, hc_te_ids, dino_tr, dino_te)
    lab = pd.read_csv(f"{C.DATA_DIR}/Training_labels_updated.csv")
    sub = pd.read_csv(f"{C.DATA_DIR}/sample_submission.csv")
    tr = A["hc_tr"].merge(A["dino_tr"], on="sid", how="inner").merge(lab, left_on="sid", right_on="sample_id", how="inner")
    te = A["hc_te"].merge(A["dino_te"], on="sid", how="inner")
    return tr, te, sub


def main():
    t0 = time.time()
    tr, te, sub = build_dataset()
    tr = tr.sort_values("sample_id").reset_index(drop=True)
    te = te.merge(sub[["sample_id"]], left_on="sid", right_on="sample_id", how="inner")
    Y = tr[C.SUPPORT_COLS].values.astype(float)
    tr_ids = tr["sample_id"].values
    te_ids = te["sample_id"].values
    print(f"train samples: {len(tr)}, test samples: {len(te)}", flush=True)
    hc_cols = [c for c in A_cols(tr, "hc")]
    dino_cols = [c for c in A_cols(tr, "dino")]
    feature_sets = {
        "hc": hc_cols, "dino": dino_cols, "hc+dino": hc_cols + dino_cols,
    }
    models = make_models()
    results = []
    for fs, cols in feature_sets.items():
        Xtr = tr[cols].values.astype(float); Xte = te[cols].values.astype(float)
        for mname, fn in models.items():
            for tf in TRANSFORMS:
                s = loocv_score(Xtr, Y, fn, tf)
                results.append((s, fs, mname, tf.name))
                print(f"  {fs:8s} {mname:14s} {tf.name:6s} LOOCV={s:.4f}", flush=True)
    results.sort(key=lambda r: r[0])
    print("\n=== TOP 15 ===")
    for r in results[:15]:
        print(f"  {r[1]:8s} {r[2]:14s} {r[3]:6s} {r[0]:.4f}")
    print(f"\ntotal {time.time()-t0:.0f}s")
    # save results
    state = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "state")
    with open(f"{state}/loo_results.json", "w") as f:
        json.dump([{"score": r[0], "fs": r[1], "model": r[2], "tf": r[3]} for r in results], f, indent=1)
    return results


def A_cols(tr, prefix):
    return [c for c in tr.columns if (prefix == "hc" and (c.endswith("_m") or c.endswith("_s")))
            or (prefix == "dino" and c.startswith("dino_"))]


if __name__ == "__main__":
    main()
