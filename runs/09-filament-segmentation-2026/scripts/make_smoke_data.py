"""Build a small local smoke-test dataset mimicking the Kaggle competition layout,
from the esairlab MAGFiLO_1.0 COCO annotations (same observation format).

Layout:
  OUT/MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json
  OUT/MAGFiLO_1.0_Kaggle_2026/train/train_images/<stem>.jpeg   (symlinks)
  OUT/MAGFiLO_1.0_Kaggle_2026/test/test_images/<stem>.jpeg     (symlinks)
"""
import json
import sys
from pathlib import Path

RUN = Path(__file__).resolve().parent.parent
SRC_JSON = RUN / "data/extracted/MAGFiLO_1.0_Annotations.json"
IMG_DIR = RUN / "data/extracted/MAGFiLO_1.0_Images"
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/fil_smoke")
N_TRAIN_OBS = int(sys.argv[2]) if len(sys.argv) > 2 else 24
N_TEST_OBS = int(sys.argv[3]) if len(sys.argv) > 3 else 4

d = json.loads(SRC_JSON.read_text())
by_stem: dict[str, list] = {}
for img in d["images"]:
    stem = Path(img["file_name"]).stem
    by_stem.setdefault(stem, []).append(img)
stems = sorted(by_stem)
# require images present on disk
stems = [s for s in stems if (IMG_DIR / f"{s}.jpeg").exists()]
print(f"obs with local images: {len(stems)}")
train_stems = stems[:N_TRAIN_OBS]
test_stems = stems[N_TRAIN_OBS:N_TRAIN_OBS + N_TEST_OBS]
assert len(test_stems) == N_TEST_OBS

ann_img_ids = {img["id"] for s in train_stems for img in by_stem[s]}
mini = {
    "info": d.get("info", {}),
    "licenses": d.get("licenses", []),
    "categories": d.get("categories", []),
    "images": [img for img in d["images"] if img["id"] in ann_img_ids],
    "annotations": [a for a in d["annotations"] if a["image_id"] in ann_img_ids],
}

root = OUT / "MAGFiLO_1.0_Kaggle_2026"
tr_dir = root / "train"
te_dir = root / "test"
tr_img = tr_dir / "train_images"
te_img = te_dir / "test_images"
for p in (tr_img, te_img):
    p.mkdir(parents=True, exist_ok=True)
(tr_dir / "MAGFiLO_1.0_Annotations_kaggle2026_train.json").write_text(json.dumps(mini))
for s in train_stems:
    link = tr_img / f"{s}.jpeg"
    if not link.exists():
        link.symlink_to(IMG_DIR / f"{s}.jpeg")
for s in test_stems:
    link = te_img / f"{s}.jpeg"
    if not link.exists():
        link.symlink_to(IMG_DIR / f"{s}.jpeg")
print(f"smoke data at {root}: train_obs={len(train_stems)} test_obs={len(test_stems)} "
      f"records={len(mini['images'])} anns={len(mini['annotations'])}")
