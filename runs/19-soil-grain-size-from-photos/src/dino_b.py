"""DINOv2 base (ViT-B/14) features for train+test photos, cached to state/feat_dino_b518.npz."""
import os, sys, time
import numpy as np
import torch
import timm
import torchvision.transforms as T
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common as C

STATE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "state")
out = f"{STATE}/feat_dino_b518.npz"

torch.set_num_threads(8)
model = timm.create_model("vit_base_patch14_dinov2.lvd142m", pretrained=True, num_classes=0)
model.eval()
tf = T.Compose([T.Resize(518), T.CenterCrop(518), T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.224])])

def embed(path):
    img = Image.open(path).convert("RGB")
    x = tf(img).unsqueeze(0)
    with torch.no_grad():
        f = model.forward_features(x).squeeze(0)
    return np.concatenate([f[0].numpy(), f[1:].mean(0).numpy()]).astype(float)

tr_photos = C.list_photos(C.TRAIN_IMG_DIR, True)
te_photos = C.list_photos(C.TEST_IMG_DIR, False)
tr_d = np.zeros((len(tr_photos), 1536)); te_d = np.zeros((len(te_photos), 1536))
t0 = time.time()
for i, (p, s, c) in enumerate(tr_photos):
    tr_d[i] = embed(p)
    if i % 20 == 0: print(f"tr {i}/{len(tr_photos)} {time.time()-t0:.0f}s", flush=True)
for i, (p, s, c) in enumerate(te_photos):
    te_d[i] = embed(p)
np.savez_compressed(out, tr=tr_d, te=te_d)
print(f"done {tr_d.shape} {te_d.shape} in {time.time()-t0:.0f}s", flush=True)
