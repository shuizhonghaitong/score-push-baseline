"""Convert notebook output clip_embeddings.npz into local files."""
import os, sys
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
src = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, "outputs/clip_embeddings.npz")
d = np.load(src, allow_pickle=True)
np.savez_compressed(os.path.join(ROOT, "outputs/clip_train.npz"), emb=d["tr"])
np.savez_compressed(os.path.join(ROOT, "outputs/clip_test.npz"), emb=d["te"])
np.savez_compressed(os.path.join(ROOT, "outputs/clip_qtext_train.npz"), emb=d["q"], qids=d["qids"])
if "tr2" in d.files:
    np.savez_compressed(os.path.join(ROOT, "outputs/dino_train.npz"), emb=d["tr2"])
    np.savez_compressed(os.path.join(ROOT, "outputs/dino_test.npz"), emb=d["te2"])
print("ok", d["tr"].shape, d["te"].shape, d["q"].shape, "dino:", d["tr2"].shape if "tr2" in d.files else None)
