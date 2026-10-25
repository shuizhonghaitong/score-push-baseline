"""Reproduce the v5 IndexError locally: exact functions from notebooks/train-dl/train_dl.py
run on synthetic test images (800x1200x3 TIF + variants) with a random Unet saved fp16."""
import os, io, time, math, sys, traceback, tempfile
import numpy as np
import pandas as pd
from PIL import Image
import cv2

SRC = os.path.join(os.path.dirname(__file__), "..", "..", "notebooks", "train-dl", "train_dl.py")
code = open(SRC).read()

def grab(name, nxt):
    i = code.index(f"def {name}(")
    j = code.index(f"\ndef {nxt}") if nxt.startswith("skeletonize") or nxt.startswith("two_lines") or nxt.startswith("fasc_lines") or nxt.startswith("decode_one") or nxt.startswith("predict_mask") else code.index(f"\n{nxt}")
    return code[i:j]

# extract verbatim function bodies
funcs = ""
for name, nxt in [("decode_hardware_scale", "decode_one"), ("decode_one", "skeletonize"),
                  ("skeletonize", "two_lines"), ("two_lines", "fasc_lines"),
                  ("fasc_lines", "models_apo")]:
    funcs += grab(name, nxt)
print("extracted", [n for n, _ in [("decode_hardware_scale",0),("decode_one",0),("skeletonize",0),("two_lines",0),("fasc_lines",0)]], file=sys.stderr)

import torch
DEVICE = "cpu"
SZ = 512

ns = {"torch": torch, "cv2": cv2, "np": np, "os": os, "math": math, "DEVICE": DEVICE, "SZ": SZ, "Image": Image}
i = code.index("def predict_mask("); j = code.index("\nrows = []")
exec(code[i:j], ns)
ns.update({k: v for k, v in ns.items()})
exec(funcs, ns)
for k, v in ns.items():
    if callable(v) and k in ("decode_hardware_scale","decode_one","skeletonize","two_lines","fasc_lines","predict_mask"):
        globals()[k] = v

# ---------------- synthetic test images ----------------
def make_test_tif(path, h=800, w=1200, ch=3, pages=1, style="right"):
    """mimic umud test tif: gray ultrasound bg + 2 horizontal bright lines + diagonal fascicles + ruler"""
    rng = np.random.default_rng(0)
    frames = []
    for p in range(pages):
        a = rng.integers(0, 60, (h, w), dtype=np.uint8)
        # superficial + deep aponeurosis: bright slightly-tilted lines
        for y0, tilt in [(300, 0.02), (420, -0.01)]:
            for x in range(w):
                y = int(y0 + tilt * (x - w / 2) + p)
                if 0 <= y < h:
                    a[max(0, y-2):y+3, x] = 230
        # fascicles: diagonal bright streaks between the two lines
        for i in range(14):
            x0 = 60 + i * 78
            for s in range(0, 130):
                x = x0 + int(s * 0.55)
                y = 300 + s
                if x < w and y < h:
                    a[max(0, y-1):y+2, x] = np.clip(a[y, x] + 120, 0, 255)
        # ruler: right-side ticks (value 175) if style right
        if style == "right" and w >= 1160:
            a[87, 1147:1157] = 175
            col = 40
            for k in range(7):
                y = 40 + k * 113
                a[y, 1147:1157] = 175
                a[y-2:y+3, 1150] = 200
        frames.append(a)
    if pages == 1:
        a = frames[0]
        if ch == 3:
            a = np.stack([a] * 3, axis=-1)
        Image.fromarray(a).save(path)
    else:
        ims = []
        for f in frames:
            ff = f if ch == 1 else np.stack([f] * 3, axis=-1)
            ims.append(Image.fromarray(ff))
        ims[0].save(path, save_all=True, append_images=ims[1:])
    return path

# random unet saved fp16 (same as train_task)
import segmentation_models_pytorch as smp
m = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=1, classes=1).to(DEVICE).eval()
tdir = tempfile.mkdtemp()
cks = []
for k in range(3):
    c = os.path.join(tdir, f"fold{k}_apo.pt")
    torch.save({kk: vv.half() for kk, vv in m.state_dict().items()}, c)
    cks.append(c)
models = []
for c in cks:
    mm = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=1, classes=1).to(DEVICE).eval()
    mm.load_state_dict(torch.load(c, map_location="cpu", weights_only=False), strict=True)
    models.append(mm)
print("model loaded (random weights, fp16 ckpt)", file=sys.stderr)

tmp = tempfile.mkdtemp()
cases = [
    ("t800_1200x3", make_test_tif(os.path.join(tmp, "IMG_00001.tif"), 800, 1200, 3)),
    ("t800_1200gray", make_test_tif(os.path.join(tmp, "IMG_00002.tif"), 800, 1200, 1)),
    ("t644_1088x3", make_test_tif(os.path.join(tmp, "IMG_00003.tif"), 644, 1088, 3)),
    ("t512_512x3", make_test_tif(os.path.join(tmp, "IMG_00004.png"), 512, 512, 3)),
    ("t800_1200x3_pages5", make_test_tif(os.path.join(tmp, "IMG_00005.tif"), 800, 1200, 3, pages=5)),
]
for tag, path in cases:
    print("=" * 20, tag, path, file=sys.stderr)
    rec = {"image_id": os.path.basename(path)}
    try:
        px_per_cm, l, t, r, b = decode_one(path)
        with Image.open(path) as img:
            a = np.asarray(img)
        print("  shape:", a.shape, "px_per_cm:", px_per_cm, "bbox:", (l, t, r, b), file=sys.stderr)
        gray = a if a.ndim == 2 else a.mean(axis=-1).astype(np.float32)
        ap_prob = predict_mask(models, gray)
        fa_prob = predict_mask(models, gray)
        ap_mask = (ap_prob > 0.35).astype(np.uint8)
        fa_mask = (fa_prob > 0.30).astype(np.uint8)
        lines = two_lines(ap_mask)
        mt_raw = pa_raw = None
        if lines is not None:
            s_sup, cx1, cy1 = lines[0]
            s_deep, cx2, cy2 = lines[1]
            mid_x = gray.shape[1] / 2.0
            y_sup = s_sup * (mid_x - cx1) + cy1
            y_deep = s_deep * (mid_x - cx2) + cy2
            mt_raw = abs(y_deep - y_sup) * (10.0 / px_per_cm)
            cands = fasc_lines(fa_mask)
            if cands:
                central = [c for c in cands if c[2]]
                pool = central if len(central) >= 3 else cands
                pool = sorted(pool, key=lambda c: c[1], reverse=True)[:3]
                tw = sum(c[1] for c in pool)
                pa_raw = sum(math.degrees(math.atan(abs((c[0] - s_deep) / (1.0 + c[0] * s_deep)))) * c[1] for c in pool) / tw
        rec.update(mt_raw=mt_raw, pa_raw=pa_raw, ok_dl=(mt_raw is not None))
        print("  OK:", rec, file=sys.stderr)
    except Exception as e:
        print("  EXC:", type(e).__name__, e, file=sys.stderr)
        traceback.print_exc()
print("DONE", file=sys.stderr)
