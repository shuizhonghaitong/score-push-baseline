"""Extract and cache per-photo features for train + test."""
import os, sys, time, glob
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C
import features as F

STATE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "state")
os.makedirs(STATE, exist_ok=True)


def extract_all(target_long=1600, use_dino=True, dino_size=518, force=False):
    ppm_df = C.load_ppm_table()
    train_photos = C.list_photos(C.TRAIN_IMG_DIR, True)
    test_photos = C.list_photos(C.TEST_IMG_DIR, False)
    print(f"train photos: {len(train_photos)}, test photos: {len(test_photos)}")

    def extract_hand(path, cam):
        ppm, table_long, src = C.lookup_ppm(cam, ppm_df)
        feats = F.extract_handcrafted(path, ppm, table_long, target_long=target_long)
        if feats is None:
            return None
        return feats

    # ---- handcrafted ----
    hc_cache = f"{STATE}/feat_handcrafted_cal_tl{target_long}.npz"
    if use_dino:
        dino_cache = f"{STATE}/feat_dino_s{dino_size}.npz"
    if os.path.exists(hc_cache) and not force:
        d = np.load(hc_cache, allow_pickle=True)
        print("loaded cached handcrafted features")
    else:
        t0 = time.time()
        tr_rows, te_rows = [], []
        tr_ids, te_ids = [], []
        for i, (p, sid, cam) in enumerate(train_photos):
            f = extract_hand(p, cam)
            if f is not None:
                tr_rows.append(f); tr_ids.append(sid)
        for i, (p, sid, cam) in enumerate(test_photos):
            f = extract_hand(p, cam)
            if f is not None:
                te_rows.append(f); te_ids.append(sid)
        keys = sorted(tr_rows[0].keys())
        tr_arr = np.array([[r[k] for k in keys] for r in tr_rows], float)
        te_arr = np.array([[r[k] for k in keys] for r in te_rows], float)
        np.savez_compressed(hc_cache, keys=np.array(keys), tr=tr_arr, tr_ids=np.array(tr_ids),
                            te=te_arr, te_ids=np.array(te_ids))
        print(f"handcrafted features: {len(keys)} dims; train {tr_arr.shape} test {te_arr.shape} in {time.time()-t0:.0f}s")

    # ---- dino ----
    if use_dino:
        if os.path.exists(dino_cache) and not force:
            d = np.load(dino_cache, allow_pickle=True)
            print("loaded cached dino features")
        else:
            t0 = time.time()
            model, tf = __import__('dino').get_dino(dino_size)
            tr_d = np.zeros((len(train_photos), 768)); te_d = np.zeros((len(test_photos), 768))
            for i, (p, sid, cam) in enumerate(train_photos):
                tr_d[i] = __import__('dino').dino_embed(p, model, tf, dino_size)
            for i, (p, sid, cam) in enumerate(test_photos):
                te_d[i] = __import__('dino').dino_embed(p, model, tf, dino_size)
            np.savez_compressed(dino_cache, tr=tr_d, te=te_d)
            print(f"dino features: train {tr_d.shape} test {te_d.shape} in {time.time()-t0:.0f}s")

    # load into memory
    hc = np.load(hc_cache, allow_pickle=True)
    if use_dino:
        dd = np.load(dino_cache, allow_pickle=True)
    # align test ids with photo list order (they're in same order since we iterate same list)
    return {
        "hc_keys": hc["keys"], "hc_tr": hc["tr"], "hc_tr_ids": hc["tr_ids"],
        "hc_te": hc["te"], "hc_te_ids": hc["te_ids"],
        "dino_tr": dd["tr"] if use_dino else None,
        "dino_te": dd["te"] if use_dino else None,
    }


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--target-long", type=int, default=1600)
    ap.add_argument("--no-dino", action="store_true")
    ap.add_argument("--dino-size", type=int, default=518)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    extract_all(a.target_long, not a.no_dino, a.dino_size, a.force)
    print("done")
