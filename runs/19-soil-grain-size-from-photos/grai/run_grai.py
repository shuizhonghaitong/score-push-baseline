"""Run the public GRAI CNN (soranz84/grai) on photos; predict PSD via Weibull (b, c)."""
import os, sys, glob, math, time
import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
import common as C

import tensorflow as tf
from PIL import Image

SUPPORT = C.SUPPORT_MM

def cam_scale(w, h, cam):
    """Reproduce the demo's camera scaling (standardizes to ~4.6 ppm), both orientations."""
    lo, hi = min(w, h), max(w, h)
    # demo uses width; we use the long side to be orientation-robust
    if hi in (1800, 4000):            # Motorola Edge file/original
        return 720 / 1800 if hi == 4000 else 1.0
    if hi in (6936, 9248):            # Samsung A52
        return 1600 / 9248 if hi == 9248 else 1.0
    if hi in (3024, 4032):            # iPhone 14
        return 552 / 1673 if hi == 4032 else 1.0
    if hi in (4284, 5712):            # iPhone 16 (not in demo; match ~4.6 ppm)
        return 4.6 / 19.525 if hi == 5712 else 1.0
    return 1.0


def preprocess(path, scale):
    img = Image.open(path).convert("RGB")
    w, h = img.size
    if scale != 1.0:
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.Resampling.LANCZOS)
    w, h = img.size
    left = (w - 400) // 2
    upper = (h - 400) // 2
    img = img.crop((left, upper, left + 400, upper + 400))
    arr = np.array(img).astype(np.float32)
    arr = (arr - 127.5) / 127.5
    return arr


def weibull_psds(bc):
    """bc: (n,2) [lnb, c] -> (n,11) percent finer at SUPPORT."""
    b = np.exp(bc[:, 0:1])
    c = np.clip(bc[:, 1:2], 1e-3, None)
    d = SUPPORT[None, :] / b
    F = (1.0 - np.exp(-(d ** c))) * 100.0
    return F


def main():
    model = tf.keras.models.load_model("grai/CNN.h5", compile=False)
    ppm_df = C.load_ppm_table()
    tr_photos = C.list_photos(C.TRAIN_IMG_DIR, True)
    te_photos = C.list_photos(C.TEST_IMG_DIR, False)
    out = {}
    for tag, photos in (("tr", tr_photos), ("te", te_photos)):
        arrs, ids = [], []
        for p, sid, cam in photos:
            w0, h0 = Image.open(p).size
            scale = cam_scale(w0, h0, cam)
            arrs.append(preprocess(p, scale))
            ids.append(sid)
        X = np.stack(arrs).astype(np.float32)
        pred = model.predict(X, verbose=0)
        if isinstance(pred, (list, tuple)):
            bc = np.column_stack([np.asarray(pred[0]).ravel(), np.asarray(pred[1]).ravel()])
        else:
            bc = np.asarray(pred).astype(float)
        F = weibull_psds(bc)
        out[tag] = (np.array(ids), bc, F)
        print(f"{tag}: {len(photos)} photos, bc range lnb [{bc[:,0].min():.3f},{bc[:,0].max():.3f}] c [{bc[:,1].min():.3f},{bc[:,1].max():.3f}")
    np.savez_compressed("state/grai_preds.npz",
                        tr_ids=out["tr"][0], tr_bc=out["tr"][1], tr_F=out["tr"][2],
                        te_ids=out["te"][0], te_bc=out["te"][1], te_F=out["te"][2])
    print("saved state/grai_preds.npz")

if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"done in {time.time()-t0:.0f}s")
