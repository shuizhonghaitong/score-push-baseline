#!/usr/bin/env python3
"""Verify the mirror zip, extract it, and build the official Kaggle 2026 layout
under data/official/MAGFiLO_1.0_Kaggle_2026 via symlinks. Prints a sanity report."""
import hashlib, json, os, sys, zipfile
from pathlib import Path

RUN = Path(__file__).resolve().parents[1]
ZIP = Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/mirror_parts/mirror_full.zip")
EXPECT_MD5 = "1676858dd25eaf583a7aa9d460f444d5"
EXPECT_SIZE = 710546369
EXTRACT = RUN / "data" / "mirror-kh"
OFFICIAL = RUN / "data" / "official" / "MAGFiLO_1.0_Kaggle_2026"

def main():
    size = ZIP.stat().st_size
    if size != EXPECT_SIZE:
        raise SystemExit(f"size mismatch: {size} != {EXPECT_SIZE}")
    h = hashlib.md5()
    with open(ZIP, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    md5 = h.hexdigest()
    if md5 != EXPECT_MD5:
        raise SystemExit(f"md5 mismatch: {md5} != {EXPECT_MD5}")
    print(f"[zip] size={size} md5={md5} OK")

    if not (EXTRACT / "images" / "test").exists():
        with zipfile.ZipFile(ZIP) as zf:
            zf.extractall(EXTRACT)
    train_imgs = sorted((EXTRACT / "images" / "train").glob("*.jpeg"))
    test_imgs = sorted((EXTRACT / "images" / "test").glob("*.jpeg"))
    print(f"[extract] train_images={len(train_imgs)} test_images={len(test_imgs)}")

    OFFICIAL.mkdir(parents=True, exist_ok=True)
    (OFFICIAL / "train").mkdir(exist_ok=True)
    (OFFICIAL / "test").mkdir(exist_ok=True)
    def link(src: Path, dst: Path):
        if dst.is_symlink() or dst.exists():
            if dst.is_symlink() or dst.is_file():
                dst.unlink()
            elif dst == (OFFICIAL / "train" / "train_images") or dst == (OFFICIAL / "test" / "test_images"):
                import shutil
                shutil.rmtree(dst)
        dst.symlink_to(src.resolve())
    link(EXTRACT / "images" / "train", OFFICIAL / "train" / "train_images")
    link(EXTRACT / "images" / "test", OFFICIAL / "test" / "test_images")
    link(EXTRACT / "annotations" / "magfilo_train.json",
         OFFICIAL / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json")

    from pycocotools.coco import COCO
    coco = COCO(str(OFFICIAL / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"))
    n_imgs = len(coco.imgs)
    n_ann = len(coco.anns)
    cats = coco.getCatIds()
    ann0 = coco.anns[next(iter(coco.anns))]
    seg_kind = "RLE" if isinstance(ann0.get("segmentation"), dict) else "polygon"
    missing = [i for i in coco.imgs if not (OFFICIAL / "train" / "train_images" / coco.imgs[i]["file_name"]).exists()]
    print(f"[coco] train images={n_imgs} anns={n_ann} cats={cats} seg={seg_kind} missing_imgs={len(missing)}")
    if missing:
        print("  missing sample:", missing[:5])
    stems = {Path(coco.imgs[i]["file_name"]).stem for i in coco.imgs}
    test_stems = {p.stem for p in test_imgs}
    print(f"[overlap] train stems={len(stems)} test stems={len(test_stems)} train&test={len(stems & test_stems)}")
    print("[layout] official tree ready at", OFFICIAL)

if __name__ == "__main__":
    main()
