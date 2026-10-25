"""V2: multi-scale hc features, per-photo PSD prediction, LOOS-CV, ensembles."""
import os, sys, json, time
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.neighbors import KNeighborsRegressor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

STATE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "state")
SCALES = [800, 1600, 3200]


class Cum:
    name = "cum"
    @staticmethod
    def enc(Y): return Y.copy()
    @staticmethod
    def dec(Yt): return C.enforce_rows(Yt)


class Delta:
    name = "delta"
    @staticmethod
    def enc(Y): return np.diff(Y, axis=1, prepend=0)
    @staticmethod
    def dec(Dt):
        D = np.clip(Dt, 0, None)
        s = D.sum(axis=1, keepdims=True); s = np.where(s <= 1e-9, 1.0, s)
        return C.enforce_rows(np.cumsum(D / s * 100.0, axis=1))


TF = {c.name: c for c in (Cum(), Delta())}


def load_photos():
    d = {}
    ids, paths = C.list_photos(C.TRAIN_IMG_DIR, True), C.list_photos(C.TEST_IMG_DIR, False)
    for s in SCALES:
        f = np.load(f"{STATE}/feat_handcrafted_cal_tl{s}.npz", allow_pickle=True)
        d[f"hc{s}"] = (f["tr"].astype(float), f["te"].astype(float))
    dd = np.load(f"{STATE}/feat_dino_s518.npz", allow_pickle=True)
    d["dino"] = (dd["tr"].astype(float), dd["te"].astype(float))
    keys800 = list(np.load(f"{STATE}/feat_handcrafted_cal_tl800.npz", allow_pickle=True)["keys"])
    lab = pd.read_csv(f"{C.DATA_DIR}/Training_labels_updated.csv")
    tr_ids = [sid for _, sid, _ in ids]
    te_ids = [sid for _, sid, _ in C.list_photos(C.TEST_IMG_DIR, False)]
    Y = {r["sample_id"]: r[C.SUPPORT_COLS].values.astype(float) for _, r in lab.iterrows()}
    return d, tr_ids, te_ids, Y, keys800


def photo_matrix(d, use_dino=False):
    tr = np.concatenate([d[f"hc{s}"][0] for s in SCALES], axis=1)
    te = np.concatenate([d[f"hc{s}"][1] for s in SCALES], axis=1)
    if use_dino:
        tr = np.concatenate([tr, d["dino"][0]], axis=1)
        te = np.concatenate([te, d["dino"][1]], axis=1)
    return tr, te


def sample_matrix(Xp, ids):
    """mean + std of photo features per sample."""
    df = pd.DataFrame(Xp)
    df["sid"] = ids
    m = df.groupby("sid").mean(numeric_only=True).add_suffix("_m")
    st = df.groupby("sid").std(numeric_only=True).add_suffix("_s").fillna(0.0)
    return pd.concat([m, st], axis=1), m.index.tolist()


def fit_predict_ridge(Xtr, Ytr, Xte, alpha, tfn):
    sc = StandardScaler(); Ys = sc.fit_transform(TF[tfn].enc(Ytr))
    m = Ridge(alpha=alpha); m.fit(sc.fit_transform(Xtr) if False else Xtr, Ys)
    # note: ridge on raw X is what v1 used; keep raw X for consistency
    p = TF[tfn].dec(sc.inverse_transform(m.predict(Xte)))
    return p


def fit_predict_pls(Xtr, Ytr, Xte, comp, tfn):
    m = PLSRegression(n_components=comp)
    m.fit(Xtr, TF[tfn].enc(Ytr))
    return TF[tfn].dec(m.predict(Xte))


def fit_predict_knn(Xtr, Ytr, Xte, k, tfn):
    sc = StandardScaler(); Xs = sc.fit_transform(Xtr); Xts = sc.transform(Xte)
    m = KNeighborsRegressor(n_neighbors=k, weights="distance", metric="cosine")
    m.fit(Xs, TF[tfn].enc(Ytr))
    return TF[tfn].dec(m.predict(Xts))


def photo_cv(Xp_tr, ids, Y, kind):
    """LOOS at photo level: predict each photo, average per sample."""
    sids = sorted(set(ids))
    ids_a = np.array(ids)
    oof = {}
    for s in sids:
        trm = ids_a != s; tem = ~trm
        Xtr, Xte = Xp_tr[trm], Xp_tr[tem]
        Ytr = np.array([Y[i] for i in ids_a[trm]])
        if kind[0] == "ridge":
            p = fit_predict_ridge(Xtr, Ytr, Xte, kind[1], kind[2])
        elif kind[0] == "pls":
            p = fit_predict_pls(Xtr, Ytr, Xte, kind[1], kind[2])
        else:
            p = fit_predict_knn(Xtr, Ytr, Xte, kind[1], kind[2])
        oof[s] = p.mean(axis=0)
    Ys = np.array([Y[s] for s in sids]); Ps = np.array([oof[s] for s in sids])
    return C.emd_trap_mean(Ys, Ps), Ps


def sample_cv(Xp_tr, ids, Y, kind):
    """LOOS with sample-aggregated features."""
    sids = sorted(set(ids))
    ids_a = np.array(ids)
    oof = {}
    for s in sids:
        trm = ids_a != s
        Xs_tr, _ = sample_matrix(Xp_tr[trm], list(ids_a[trm]))
        Xs_te_all, _ = sample_matrix(Xp_tr, ids)
        Xs_te = Xs_te_all[Xs_te_all.index == s].reset_index(drop=True)
        Xs_tr = Xs_tr.reset_index(drop=True)
        Ytr = np.array([Y[i] for i in sids if i != s])
        if kind[0] == "ridge":
            p = fit_predict_ridge(Xs_tr.values, Ytr, Xs_te.values, kind[1], kind[2])[0]
        elif kind[0] == "pls":
            p = fit_predict_pls(Xs_tr.values, Ytr, Xs_te.values, kind[1], kind[2])[0]
        else:
            p = fit_predict_knn(Xs_tr.values, Ytr, Xs_te.values, kind[1], kind[2])[0]
        oof[s] = p
    Ys = np.array([Y[s] for s in sids]); Ps = np.array([oof[s] for s in sids])
    return C.emd_trap_mean(Ys, Ps), Ps


def main():
    t0 = time.time()
    d, tr_ids, te_ids, Y, keys = load_photos()
    Xp_tr, Xp_te = photo_matrix(d, use_dino=False)
    Xpd_tr, Xpd_te = photo_matrix(d, use_dino=True)
    print(f"photo features: hc {Xp_tr.shape}, +dino {Xpd_tr.shape}", flush=True)
    kinds = (
        [("ridge", a, tf) for a in (5, 15, 40, 120) for tf in ("cum", "delta")]
        + [("pls", c, tf) for c in (3, 5) for tf in ("cum", "delta")]
        + [("knn", k, tf) for k in (3, 5, 8) for tf in ("cum", "delta")]
    )
    results = []
    for name, Xp in (("hc", Xp_tr), ("hc+dino", Xpd_tr)):
        for kind in kinds:
            if name == "hc+dino" and kind[0] == "knn" and kind[1] < 5:
                continue
            score, Ps = photo_cv(Xp, tr_ids, Y, kind)
            results.append((score, name, kind, Ps))
            print(f"  photo {name:8s} {str(kind):28s} LOOS={score:.4f}", flush=True)
    # sample-level models on hc only
    for kind in kinds[:8]:
        score, Ps = sample_cv(Xp_tr, tr_ids, Y, kind)
        results.append((score, "sample_hc", kind, Ps))
        print(f"  sample {str(kind):28s} LOOS={score:.4f}", flush=True)
    results.sort(key=lambda r: r[0])
    print("\n=== TOP 15 ===")
    for r in results[:15]:
        print(f"  {r[1]:10s} {str(r[2]):28s} {r[0]:.4f}")
    out = [{"score": r[0], "set": r[1], "kind": list(r[2])} for r in results[:20]]
    json.dump(out, open(f"{STATE}/loo2_top20.json", "w"), indent=1)
    # save OOF PSDs of top-10 for ensemble weighting
    top = results[:10]
    oof = np.stack([r[3] for r in top])
    np.savez_compressed(f"{STATE}/oof_top10.npz",
                        oof=oof,
                        names=np.array([f"{r[1]}|{r[2][0]}{r[2][1]}|{r[2][2]}" for r in top]))
    print(f"total {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
