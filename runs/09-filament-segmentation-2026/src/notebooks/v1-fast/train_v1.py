"""Solar Filament Segmentation Challenge 2026 - pipeline v1 (memory-safe).

Two complementary models are trained on the official training split:

1. ``unet``  - semantic U-Net (ResNet34 encoder, smp).  Instances come from
   connected components of the thresholded map.  4-flip TTA at inference.
2. ``mrcnn`` - instance Mask R-CNN (ResNet50-FPN-v2, COCO pretrain).

Memory model (important): instance masks are kept as **base64 COCO RLE
strings**, never as resident pixel arrays.  Pixels are materialised only
transiently, one observation at a time (post-processing, PQ evaluation,
merging).  This keeps peak RAM ~4-6 GB even on a 16 GB P100 container.

Validation uses the *official* panoptic-quality implementation (grouped by
annotator-image row; TP iff IoU > 0.5, one-to-many; PQ = sum(TP IoU) /
(|TP| + 0.5|FP| + 0.5|FN|)).

Outputs (in /kaggle/working):
- ``submission.csv`` + ``submission_{unet,mrcnn,merge}.csv``
- ``run_report.json``
- ``val_stems.json``
- ``val_predictions_{unet,mrcnn,merge}.npz``  per-stem keys -> b64 RLE str arrays
- ``best_unet.pt`` / ``best_mrcnn.pt``
"""
from __future__ import annotations

import base64
import json
import os
import random
import shutil
import sys
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

IMG = 2048          # native image size (fixed by the competition)
WORK = Path(os.environ.get("FILAMENT_WORK_DIR", "/kaggle/working"))
try:
    WORK.mkdir(parents=True, exist_ok=True)
except OSError:  # local smoke tests
    WORK = Path.cwd() / "working_local"
    WORK.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------- #
# 0. Reproducibility / helpers
# --------------------------------------------------------------------------- #
def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def _now() -> float:
    return time.time()


# --------------------------------------------------------------------------- #
# 1. Data loading
# --------------------------------------------------------------------------- #
def find_competition_root() -> Path:
    env_root = os.environ.get("FILAMENT_DATA_ROOT")
    if env_root:
        return Path(env_root)
    base = Path("/kaggle/input")
    anns = list(base.rglob("MAGFiLO_1.0_Annotations_kaggle2026_train.json"))
    if len(anns) != 1:
        raise RuntimeError(f"expected exactly 1 train annotation file, found {anns}")
    return anns[0].parent.parent  # MAGFiLO_1.0_Kaggle_2026


@dataclass
class Record:
    """One annotator-image record (a COCO 'image' entry).

    Images are lazily cached; GT masks are *never* cached (recomputed from
    the in-memory COCO RLE/polygon via annToMask) to bound RAM.
    """
    record_id: str
    stem: str
    img_path: Path
    ann_items: list
    _image: "np.ndarray | None" = field(default=None, repr=False)
    _coco: object = field(default=None, repr=False)

    @property
    def image(self) -> np.ndarray:
        if self._image is None:
            self._image = cv2.imread(str(self.img_path), cv2.IMREAD_GRAYSCALE)
            if self._image is None or self._image.shape[:2] != (IMG, IMG):
                raise RuntimeError(f"bad image {self.img_path}")
        return self._image

    @property
    def masks(self) -> list:
        return [self._coco.annToMask(a) > 0 for a in self.ann_items]


def load_all(root: Path) -> tuple[list[Record], list[str]]:
    from pycocotools.coco import COCO

    ann_path = root / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"
    coco = COCO(str(ann_path))
    records: list[Record] = []
    for rec_id, img_info in coco.imgs.items():
        stem = Path(img_info["file_name"]).stem
        img_path = root / "train" / "train_images" / img_info["file_name"]
        records.append(Record(record_id=str(rec_id), stem=stem, img_path=img_path,
                              ann_items=list(coco.imgToAnns[rec_id]), _coco=coco))
    first = records[0].image
    n_ann = sum(len(r.ann_items) for r in records)
    obs = len({r.stem for r in records})
    test_files = sorted(p.stem for p in (root / "test" / "test_images").glob("*.jpeg"))
    print(f"[data] records={len(records)} observations={obs} annotations={n_ann} "
          f"test_images={len(test_files)} first_image={first.shape}", flush=True)
    return records, test_files


def grouped_split(records: list[Record], val_frac: float = 0.2, seed: int = 42) -> tuple[list, list]:
    """Split by physical observation so annotator variants stay together."""
    stems = sorted({r.stem for r in records})
    rng = random.Random(seed)
    rng.shuffle(stems)
    n_val = max(1, int(round(len(stems) * val_frac)))
    val_stems = set(stems[:n_val])
    train = [r for r in records if r.stem not in val_stems]
    val = [r for r in records if r.stem in val_stems]
    print(f"[split] obs={len(stems)} train_obs={len(stems) - n_val} val_obs={n_val} "
          f"train_rec={len(train)} val_rec={len(val)}", flush=True)
    return train, val


# --------------------------------------------------------------------------- #
# 2. RLE storage + official panoptic quality (streaming, one row at a time)
# --------------------------------------------------------------------------- #
def mask_to_rle_b64(mask: np.ndarray) -> str:
    from pycocotools import mask as mu
    rle = mu.encode(np.asfortranarray(mask.astype(np.uint8)))
    counts = rle["counts"]
    if isinstance(counts, bytes):
        return base64.b64encode(counts).decode("ascii")
    return str(counts)


def masks_to_rles(masks: list[np.ndarray]) -> list[str]:
    return [mask_to_rle_b64(m) for m in masks if m.any()]


def upsample_masks(masks: list[np.ndarray], shape: tuple[int, int]) -> list[np.ndarray]:
    out = []
    for m in masks:
        if m.shape[:2] == shape:
            out.append(m)
        else:
            out.append(cv2.resize(m.astype(np.uint8), shape, interpolation=cv2.INTER_NEAREST).astype(bool))
    return out


def rle_b64_to_mask(s: str) -> np.ndarray:
    from pycocotools import mask as mu
    try:
        counts = base64.b64decode(s)
    except Exception:
        counts = s
    return (mu.decode({"size": [IMG, IMG], "counts": counts}) > 0)


def rles_to_masks(rles: list[str]) -> list[np.ndarray]:
    return [rle_b64_to_mask(r) for r in rles]


def pq_stream(records: list[Record], pred_rles_by_stem: dict[str, list[str]],
              iou_threshold: float = 0.5) -> dict:
    """Official PQ, evaluated row by row so only one row of pixels is live.

    One row per annotator record; predictions grouped by physical stem
    (identical to the competition Self-Evaluation grouping).
    """
    tp_sum = 0.0
    tp = fp = fn = 0
    for rec in records:
        gt = rec.masks  # list of (IMG,IMG) bool, transient
        pred_rles = pred_rles_by_stem.get(rec.stem, [])
        pred = rles_to_masks(pred_rles)  # transient
        n_gt, n_pred = len(gt), len(pred)
        if n_gt == 0:
            fp += n_pred
            continue
        if n_pred == 0:
            fn += n_gt
            continue
        gtf = np.stack(gt).reshape(n_gt, -1).astype(np.float32)
        ptf = np.stack(pred).reshape(n_pred, -1).astype(np.float32)
        inter = gtf @ ptf.T
        union = gtf.sum(1, keepdims=True) + ptf.sum(1, keepdims=True).T - inter
        iou = np.where(union > 0, inter / np.maximum(union, 1), 0.0)
        hit = iou > iou_threshold
        tp_iou = iou[hit]
        tp += int(hit.sum())
        tp_sum += float(tp_iou.sum())
        fp += int((hit.sum(axis=0) == 0).sum())
        fn += int((hit.sum(axis=1) == 0).sum())
        del gt, pred, gtf, ptf, inter, union, iou, hit
    denom = tp + 0.5 * fp + 0.5 * fn
    return {"pq": tp_sum / denom if denom > 0 else 0.0, "tp": tp, "fp": fp, "fn": fn,
            "mean_tp_iou": tp_sum / tp if tp else 0.0}


# --------------------------------------------------------------------------- #
# 3. Instance post-processing (pixels in, pixels out, per-observation)
# --------------------------------------------------------------------------- #
def remove_small(mask: np.ndarray, min_px: int) -> np.ndarray:
    n, lab = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    out = np.zeros_like(mask)
    for i in range(1, n):
        if (lab == i).sum() >= min_px:
            out[lab == i] = 1
    return out


def instances_from_semantic(mask: np.ndarray, min_px: int = 64) -> list[np.ndarray]:
    n, lab = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    out = []
    for i in range(1, n):
        part = lab == i
        if part.sum() >= min_px:
            out.append(part)
    return out


def merge_overlapping(masks: list[np.ndarray]) -> list[np.ndarray]:
    """Merge instances that share any pixel.  O(n) 2048-passes via a label
    image instead of O(n^2) pairwise ANDs."""
    masks = [m for m in masks if m.any()]
    if len(masks) < 2:
        return masks
    H, W = masks[0].shape[:2]
    L = np.zeros((H, W), np.int32)
    for i, m in enumerate(masks):
        L[m] = i + 1
    n = len(masks)
    parent = list(range(n))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i, m in enumerate(masks):
        labels = np.unique(L[m])
        for lab in labels:
            j = int(lab) - 1
            if j != i:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[ri] = rj
    groups: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)
    out = []
    for members in groups.values():
        if len(members) == 1:
            out.append(masks[members[0]])
        else:
            out.append(np.logical_or.reduce([masks[k] for k in members]))
    return out


def mrcnn_instances(raw_masks: np.ndarray, scores: np.ndarray, score_thr: float,
                    mask_thr: float, min_px: int = 32) -> list[np.ndarray]:
    """torchvision masks (n, H, W) float probs -> bool instance list."""
    keep = scores >= score_thr
    insts = []
    for k in np.flatnonzero(keep):
        m = (raw_masks[k] >= mask_thr)
        if m.any():
            m = remove_small(m, min_px)
            if m.any():
                insts.append(m)
    return merge_overlapping(insts)


def merge_rles(rles_a: list[str], rles_b: list[str]) -> list[str]:
    masks = rles_to_masks(list(rles_a) + list(rles_b))
    if not masks:
        return []
    merged = merge_overlapping(masks)
    return masks_to_rles(merged)


def merged_candidates_rles(a: dict[str, list[str]], b: dict[str, list[str]]) -> dict[str, list[str]]:
    out = {}
    for stem in a:
        out[stem] = merge_rles(a.get(stem, []), b.get(stem, []))
    return out


def rles_to_rows(pred_rles_by_stem: dict[str, list[str]]) -> pd.DataFrame:
    rows = []
    for stem in sorted(pred_rles_by_stem):
        for k, r in enumerate(pred_rles_by_stem[stem], start=1):
            rows.append({"filament_id": f"{stem}_{k}", "segmentation_rle": r})
    return pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])


# --------------------------------------------------------------------------- #
# 4. U-Net (semantic)
# --------------------------------------------------------------------------- #
def train_unet(train: list[Record], val: list[Record], epochs: int, bs: int, size: int,
               device: "torch.device") -> "torch.nn.Module":
    import torch.nn as nn
    from torch.utils.data import Dataset, DataLoader

    class SemanticDS(Dataset):
        def __init__(self, records, augment):
            self.records = records
            self.augment = augment
            self.tr = None
            if augment:
                import albumentations as A
                self.tr = A.Compose([
                    A.RandomBrightnessContrast(p=0.3),
                    A.OneOf([A.HorizontalFlip(p=1.0), A.VerticalFlip(p=1.0),
                             A.Transpose(p=1.0), A.ShiftScaleRotate(scale_limit=0.05,
                             rotate_limit=90, border_mode=cv2.BORDER_CONSTANT, p=1.0)]),
                ])

        def __len__(self):
            return len(self.records)

        def __getitem__(self, idx):
            rec = self.records[idx]
            image = cv2.resize(rec.image, (size, size), interpolation=cv2.INTER_AREA)
            target = np.zeros((size, size), np.uint8)
            for m in rec.masks:
                target |= cv2.resize(m.astype(np.uint8), (size, size), interpolation=cv2.INTER_NEAREST)
            if self.tr is not None:
                r = self.tr(image=image, mask=target)
                image, target = r["image"], r["mask"]
            image = (image.astype(np.float32) / 255.0 - 0.5) / 0.5
            return (torch.from_numpy(image).unsqueeze(0).repeat(3, 1, 1),
                    torch.from_numpy(target.astype(np.float32)))

    def dice_bce(logits, target, smooth=1.0):
        logits = logits.squeeze(1)
        prob = torch.sigmoid(logits)
        bce = nn.functional.binary_cross_entropy_with_logits(logits, target)
        inter = (prob * target).sum(dim=(1, 2))
        union = prob.sum(dim=(1, 2)) + target.sum(dim=(1, 2))
        dice = 1.0 - (2.0 * inter + smooth) / (union + smooth)
        return bce + dice.mean()

    try:
        import segmentation_models_pytorch as smp
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "segmentation_models_pytorch"])
        import segmentation_models_pytorch as smp
    model = smp.Unet("resnet34", in_channels=3, classes=1, encoder_weights="imagenet").to(device)
    ds_tr, ds_val = SemanticDS(train, True), SemanticDS(val, False)
    ld_tr = DataLoader(ds_tr, batch_size=bs, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    ld_val = DataLoader(ds_val, batch_size=bs, shuffle=False, num_workers=2, pin_memory=True)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * len(ld_tr))
    use_amp = str(device).startswith("cuda")
    scaler = torch.amp.GradScaler("cuda") if use_amp else None
    best_val, best_state, best_epoch = -1.0, None, 0
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        tot, n_it = 0.0, 0
        for x, y in ld_tr:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            if use_amp:
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    loss = dice_bce(model(x), y)
                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
            else:
                loss = dice_bce(model(x), y)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            sched.step()
            tot += loss.item()
            n_it += 1
        model.eval()
        tp = fp = fn = 0
        with torch.inference_mode():
            for x, y in ld_val:
                x = x.to(device, non_blocking=True)
                if use_amp:
                    with torch.amp.autocast("cuda", dtype=torch.float16):
                        logits = model(x)
                else:
                    logits = model(x)
                p = (torch.sigmoid(logits) > 0.5).float().squeeze(1)
                tp += (p * y.to(device)).sum().item()
                fp += (p * (1 - y.to(device))).sum().item()
                fn += ((1 - p) * y.to(device)).sum().item()
        val_dice = 2 * tp / max(1, 2 * tp + fp + fn)
        print(f"[unet] epoch {epoch}/{epochs} loss={tot / max(1, n_it):.4f} val_fg_dice={val_dice:.4f} "
              f"t={time.time() - t0:.0f}s", flush=True)
        if val_dice > best_val:
            best_val, best_epoch = val_dice, epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
        torch.save(best_state, WORK / "best_unet.pt")
    print(f"[unet] done best_epoch={best_epoch} best_val_dice={best_val:.4f}", flush=True)
    return model


@torch.inference_mode()
def _unet_prob_at(model, image: np.ndarray, size: int, device) -> np.ndarray:
    """Probability map at (size, size) for one orientation."""
    x = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    x = (x - 0.5) / 0.5
    t = torch.from_numpy(x).unsqueeze(0).repeat(3, 1, 1).unsqueeze(0).to(device)
    if str(device).startswith("cuda"):
        with torch.amp.autocast("cuda", dtype=torch.float16):
            logits = model(t)
    else:
        logits = model(t)
    return torch.sigmoid(logits)[0, 0].cpu().numpy()


@torch.inference_mode()
def unet_predict(model, image: np.ndarray, size: int, device, thr: float = 0.5,
                 tta: bool = True) -> np.ndarray:
    """2048 bool mask, averaged over 4 flips (or single orientation), with
    linear probability upsampling before thresholding."""
    if not tta:
        p = _unet_prob_at(model, image, size, device)
        p_big = cv2.resize(p, (IMG, IMG), interpolation=cv2.INTER_LINEAR)
        return p_big > thr
    acc = np.zeros((size, size), np.float32)
    acc += _unet_prob_at(model, image, size, device)
    acc += _unet_prob_at(model, image[:, ::-1], size, device)[:, ::-1]
    acc += _unet_prob_at(model, image[::-1, :], size, device)[::-1, :]
    acc += _unet_prob_at(model, image[::-1, ::-1], size, device)[::-1, ::-1]
    p = acc / 4.0
    p_big = cv2.resize(p, (IMG, IMG), interpolation=cv2.INTER_LINEAR)
    return p_big > thr


@torch.inference_mode()
def unet_full_val_rles(model, val: list[Record], size: int, device, thr: float = 0.5) -> dict:
    out = {}
    for i, rec in enumerate(val):
        m = unet_predict(model, rec.image, size, device, thr)
        insts = instances_from_semantic(m, 64)
        out[rec.stem] = masks_to_rles(insts)
        if (i + 1) % 20 == 0:
            print(f"[unet-val] {i + 1}/{len(val)}", flush=True)
    return out


# --------------------------------------------------------------------------- #
# 5. Mask R-CNN (instance)
# --------------------------------------------------------------------------- #
def build_mrcnn(size: int) -> "torch.nn.Module":
    from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
    from torchvision.models.detection.anchor_utils import AnchorGenerator
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
    from torchvision.ops import MultiScaleRoIAlign

    model = maskrcnn_resnet50_fpn_v2(pretrained=True, pretrained_backbone=True)
    box_in = model.roi_heads.box_predictor.cls_score.in_features
    mask_in = model.roi_heads.mask_predictor.conv5_mask.in_channels
    mask_hid = model.roi_heads.mask_predictor.conv5_mask.out_channels
    model.roi_heads.box_predictor = FastRCNNPredictor(box_in, 2)
    model.roi_heads.mask_predictor = MaskRCNNPredictor(mask_in, mask_hid, 2)
    model.rpn.anchor_generator = AnchorGenerator(sizes=((16,), (32,), (64,), (128,), (256,)),
                                                 aspect_ratios=((0.5, 1.0, 2.0),) * 5)
    model.rpn.stride = [8, 16, 32, 64, 128]
    model.roi_heads.mask_roi_pool = MultiScaleRoIAlign(featmap_names=["0", "1", "2", "3"],
                                                       output_size=28, sampling_ratio=2)
    model.transform.min_size = (size,)
    model.transform.max_size = size
    return model


class MRCNNDataset:
    """torchvision-compatible iterable over COCO-style instance targets."""

    def __init__(self, records: list[Record], size: int, augment: bool):
        self.records = records
        self.size = size
        self.augment = augment

    def __len__(self):
        return len(self.records)

    def _augment_image_mask(self, image: np.ndarray, masks: list[np.ndarray]):
        k = random.randrange(4)
        image = np.rot90(image, k).copy()
        masks = [np.rot90(m, k).copy() for m in masks]
        op = random.randrange(3)
        if op == 0:
            image = image[:, ::-1].copy()
            masks = [m[:, ::-1].copy() for m in masks]
        elif op == 1:
            image = image[::-1, :].copy()
            masks = [m[::-1, :].copy() for m in masks]
        if random.random() < 0.5:
            alpha = 1.0 + random.uniform(-0.15, 0.15)
            beta = random.uniform(-18, 18)
            image = np.clip(image.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
        return image, masks

    def __getitem__(self, idx):
        rec = self.records[idx]
        image = cv2.resize(rec.image, (self.size, self.size), interpolation=cv2.INTER_AREA)
        masks = [cv2.resize(m.astype(np.uint8), (self.size, self.size), interpolation=cv2.INTER_NEAREST) > 0
                 for m in rec.masks]
        if self.augment:
            image, masks = self._augment_image_mask(image, masks)
        boxes, masks_out, areas = [], [], []
        for ms in masks:
            ys, xs = np.where(ms)
            if len(ys) < 8:
                continue
            x0, x1 = xs.min(), xs.max()
            y0, y1 = ys.min(), ys.max()
            w, h = x1 - x0, y1 - y0
            if w < 3 or h < 3:
                continue
            boxes.append([x0, y0, x1 + 1, y1 + 1])  # xyxy (torchvision target format)
            masks_out.append(ms)
            areas.append(w * h)
        if not boxes:
            boxes = [[0, 0, 1, 1]]
            masks_out = [np.zeros((self.size, self.size), bool)]
            areas = [0.0]
        image_t = torch.from_numpy(np.stack([image] * 3, axis=-1).transpose(2, 0, 1).copy()).float() / 255.0
        target = {
            "boxes": torch.tensor(boxes, dtype=torch.float32),
            "masks": torch.tensor(np.stack(masks_out).astype(np.uint8)).float(),
            "image_id": torch.tensor([idx]),
            "area": torch.tensor(areas, dtype=torch.float32),
            "labels": torch.ones(len(boxes), dtype=torch.int64),
            "iscrowd": torch.zeros(len(boxes), dtype=torch.int64),
        }
        return image_t, target


def train_mrcnn(train: list[Record], val: list[Record], epochs: int, bs: int, size: int,
                device: "torch.device") -> "torch.nn.Module":
    from torch.utils.data import DataLoader

    ds = MRCNNDataset(train, size, augment=True)
    ld = DataLoader(ds, batch_size=bs, shuffle=True, num_workers=4, pin_memory=True,
                    collate_fn=lambda b: (list(zip(*b))[0], list(zip(*b))[1]))
    model = build_mrcnn(size).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.MultiStepLR(opt,
                                                 milestones=sorted({max(1, round(epochs * 0.7)), max(2, round(epochs * 0.9))}),
                                                 gamma=0.1)
    scaler = torch.amp.GradScaler("cuda")
    best_metric, best_state, best_epoch = -1.0, None, 0
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        tot, n_it = 0.0, 0
        for images, targets in ld:
            images = [im.to(device, non_blocking=True) for im in images]
            targets = [{k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
                        for k, v in t.items()} for t in targets]
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.float16):
                losses = model(images, targets)
                loss = sum(losses.values())
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot += loss.item()
            n_it += 1
            if n_it % 50 == 0:
                lv = {k: f"{v.item():.3f}" for k, v in list(losses.items())[:6]}
                print(f"[mrcnn] e{epoch} it{n_it}/{len(ld)} loss={tot / n_it:.4f} {lv} t={time.time() - t0:.0f}s",
                      flush=True)
        model.eval()
        metric = mrcnn_val_proxy(model, val, size, device)
        print(f"[mrcnn] epoch {epoch}/{epochs} loss={tot / max(1, n_it):.4f} {metric} "
              f"t={time.time() - t0:.0f}s", flush=True)
        if metric["f1"] > best_metric:
            best_metric, best_epoch = metric["f1"], epoch
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        sched.step()
    if best_state is not None:
        model.load_state_dict(best_state)
        torch.save(best_state, WORK / "best_mrcnn.pt")
    print(f"[mrcnn] done best_epoch={best_epoch} best_f1={best_metric:.4f}", flush=True)
    return model


@torch.inference_mode()
def mrcnn_detect(model, image: np.ndarray, size: int, device, score_thr=0.5, mask_thr=0.5):
    x = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
    if x.ndim == 2:
        x = np.stack([x] * 3, axis=-1)
    t = torch.from_numpy(x.transpose(2, 0, 1).copy()).float() / 255.0
    t = t.to(device)
    with torch.amp.autocast("cuda", dtype=torch.float16):
        out = model([t])[0]
    scores = out["scores"].cpu().numpy()
    masks = out["masks"][:, 0].cpu().numpy()
    return masks, scores


@torch.inference_mode()
def mrcnn_val_proxy(model, val: list[Record], size: int, device, score_thr=0.5, mask_thr=0.5) -> dict:
    """Fast per-observation proxy: instance F1@IoU>0.5 (greedy) + fg dice."""
    tp = fp = fn = 0
    dice_tp = dice_fp = dice_fn = 0
    for rec in val:
        det_masks, det_scores = mrcnn_detect(model, rec.image, size, device, score_thr, mask_thr)
        pred = upsample_masks(mrcnn_instances(det_masks, det_scores, score_thr, mask_thr), (IMG, IMG))
        gt = rec.masks
        if gt and pred:
            gt_a = np.array([g.sum() for g in gt])
            p_a = np.array([p.sum() for p in pred])
            inter = np.zeros((len(gt), len(pred)))
            for i, g in enumerate(gt):
                for j, p in enumerate(pred):
                    inter[i, j] = (g & p).sum()
            union = gt_a[:, None] + p_a[None, :] - inter
            iou = np.where(union > 0, inter / np.maximum(union, 1), 0)
            matched_g, matched_p = set(), set()
            order = np.argsort(-iou.ravel())
            for k in order:
                i, j = divmod(int(k), len(pred))
                if iou[i, j] > 0.5 and i not in matched_g and j not in matched_p:
                    matched_g.add(i)
                    matched_p.add(j)
                    tp += 1
            fp += len(pred) - len(matched_p)
            fn += len(gt) - len(matched_g)
        else:
            fp += len(pred)
            fn += len(gt)
        union_gt = np.logical_or.reduce(gt).astype(bool) if gt else np.zeros((IMG, IMG), bool)
        union_pred = np.logical_or.reduce(pred).astype(bool) if pred else np.zeros((IMG, IMG), bool)
        dice_tp += (union_gt & union_pred).sum()
        dice_fp += (union_pred & ~union_gt).sum()
        dice_fn += (union_gt & ~union_pred).sum()
    f1 = 2 * tp / max(1, 2 * tp + fp + fn)
    dice = 2 * dice_tp / max(1, dice_tp + dice_fp + dice_fn)
    return {"f1": round(float(f1), 4), "fg_dice": round(float(dice), 4)}


@torch.inference_mode()
def mrcnn_full_val_rles(model, val: list[Record], size: int, device,
                        score_thr: float, mask_thr: float) -> dict:
    out = {}
    for i, rec in enumerate(val):
        det_masks, det_scores = mrcnn_detect(model, rec.image, size, device, score_thr, mask_thr)
        insts = mrcnn_instances(det_masks, det_scores, score_thr, mask_thr)
        out[rec.stem] = masks_to_rles(upsample_masks(insts, (IMG, IMG)))
        if (i + 1) % 20 == 0:
            print(f"[mrcnn-val] {i + 1}/{len(val)}", flush=True)
    return out


# --------------------------------------------------------------------------- #
# 6. main
# --------------------------------------------------------------------------- #
def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--unet-epochs", type=int, default=10)
    ap.add_argument("--mrcnn-epochs", type=int, default=10)
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--unet-bs", type=int, default=8)
    args = ap.parse_args()
    # GPU kernel configuration (Kaggle script CLI args are not delivered via push);
    # FILAMENT_SMOKE=1 lets local tests use the CLI args as-is.
    if not os.environ.get("FILAMENT_SMOKE"):
        args.unet_epochs = 10
        args.mrcnn_epochs = 5
        args.size = 1024
    t_start = time.time()
    seed_everything(42)
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[cfg] {vars(args)} device={device} t_start={t_start:.0f}", flush=True)

    root = find_competition_root()
    records, test_files = load_all(root)
    train, val = grouped_split(records, args.val_frac, 42)

    # ---------------- U-Net ---------------- #
    if args.unet_epochs > 0:
        unet = train_unet(train, val, args.unet_epochs, bs=args.unet_bs, size=args.size, device=device)
    else:
        import segmentation_models_pytorch as smp
        ck = next(Path("/kaggle/input").rglob("best_unet.pt"), None)
        if ck is None:
            raise RuntimeError("best_unet.pt not found in /kaggle/input")
        unet = smp.Unet("resnet34", in_channels=3, classes=1, encoder_weights=None).to(device)
        unet.load_state_dict(torch.load(ck, map_location=device, weights_only=False))
    unet_pq, unet_val, unet_thr = None, None, 0.5
    for thr in (0.35, 0.5, 0.65):
        v = unet_full_val_rles(unet, val, args.size, device, thr=thr)
        pq = pq_stream(val, v)
        print(f"[eval] unet thr={thr} val PQ={pq['pq']:.4f} {pq}", flush=True)
        if unet_pq is None or pq["pq"] > unet_pq["pq"]:
            unet_pq, unet_val, unet_thr = pq, v, thr
        else:
            del v
    print(f"[eval] unet best thr={unet_thr} PQ={unet_pq['pq']:.4f}", flush=True)

    # ---------------- MRCNN ---------------- #
    if args.mrcnn_epochs > 0:
        mrcnn = train_mrcnn(train, val, args.mrcnn_epochs, bs=1, size=args.size, device=device)
        mrcnn_val = mrcnn_full_val_rles(mrcnn, val, args.size, device, 0.5, 0.5)
        mrcnn_pq = pq_stream(val, mrcnn_val)
        print(f"[eval] mrcnn val PQ={mrcnn_pq['pq']:.4f} {mrcnn_pq}", flush=True)
    else:
        mrcnn_val = {r.stem: [] for r in val}
        mrcnn_pq = {"pq": 0.0}

    merge_val = merged_candidates_rles(unet_val, mrcnn_val)
    merge_pq = pq_stream(val, merge_val)
    print(f"[eval] merge val PQ={merge_pq['pq']:.4f} {merge_pq}", flush=True)

    # ---------------- Test inference (RLEs, pixels freed per image) ------- #
    test_dir = root / "test" / "test_images"
    unet_test: dict[str, list[str]] = {}
    mrcnn_test: dict[str, list[str]] = {}
    for i, stem in enumerate(test_files):
        img = cv2.imread(str(test_dir / f"{stem}.jpeg"), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise RuntimeError(f"missing test image {stem}")
        m = unet_predict(unet, img, args.size, device, unet_thr)
        unet_test[stem] = masks_to_rles(instances_from_semantic(m, 64))
        del m
        if args.mrcnn_epochs > 0:
            det_masks, det_scores = mrcnn_detect(mrcnn, img, args.size, device, 0.5, 0.5)
            insts = mrcnn_instances(det_masks, det_scores, 0.5, 0.5)
            mrcnn_test[stem] = masks_to_rles(upsample_masks(insts, (IMG, IMG)))
            del det_masks
        else:
            mrcnn_test[stem] = []
        if (i + 1) % 20 == 0:
            print(f"[test] {i + 1}/{len(test_files)} t={time.time() - t_start:.0f}s", flush=True)
        del img
    merge_test = merged_candidates_rles(unet_test, mrcnn_test)

    cands = {"unet": (unet_pq, unet_test), "mrcnn": (mrcnn_pq, mrcnn_test), "merge": (merge_pq, merge_test)}
    for name, (pq, preds) in cands.items():
        df = rles_to_rows(preds)
        df.to_csv(WORK / f"submission_{name}.csv", index=False)
        print(f"[out] submission_{name}.csv rows={len(df)} valPQ={pq['pq']:.4f}", flush=True)

    best = max(cands, key=lambda k: cands[k][0]["pq"])
    shutil.copy(WORK / f"submission_{best}.csv", WORK / "submission.csv")
    print(f"[out] BEST candidate={best} valPQ={cands[best][0]['pq']:.4f}", flush=True)

    # ---------------- Report + compact val dumps (safe: RLE strings only) -- #
    report = {
        "args": vars(args),
        "n_train_records": len(train), "n_val_records": len(val),
        "n_test_images": len(test_files),
        "val_pq": {k: v[0] for k, v in cands.items()},
        "best": best,
        "seconds": round(time.time() - t_start),
        "torch": torch.__version__,
    }
    (WORK / "run_report.json").write_text(json.dumps(report, indent=1, default=str))
    val_stems = sorted({r.stem for r in val})
    (WORK / "val_stems.json").write_text(json.dumps(val_stems))

    # GT per val stem (all annotator records of that stem, in record order) as RLE strings
    gt_by_stem = {}
    for rec in val:
        gt_by_stem.setdefault(rec.stem, []).extend(masks_to_rles(rec.masks))
    try:
        parts = {}
        for i, s_ in enumerate(val_stems):
            parts[f"gt_{i:03d}"] = np.array("\n".join(gt_by_stem.get(s_, [])))
        np.savez_compressed(WORK / "val_gt.npz", stems=np.array(val_stems), **parts)
        print("[out] val_gt.npz stems=", len(val_stems), flush=True)
    except Exception as e:
        print(f"[warn] val_gt.npz failed: {e}", flush=True)

    val_pred_dicts = {"unet": unet_val, "mrcnn": mrcnn_val, "merge": merge_val}
    for name in ("unet", "mrcnn", "merge"):
        try:
            parts = {}
            for i, s in enumerate(val_stems):
                rles = val_pred_dicts[name].get(s, [])
                parts[f"pred_{i:03d}"] = np.array("\n".join(rles))  # "" = no predictions
            np.savez_compressed(WORK / f"val_predictions_{name}.npz", stems=np.array(val_stems), **parts)
            print(f"[out] val_predictions_{name}.npz stems={len(val_stems)}", flush=True)
        except Exception as e:
            print(f"[warn] val_predictions_{name}.npz failed: {e}", flush=True)
        del val_pred_dicts[name]
    print("[done]", json.dumps(report["val_pq"]), flush=True)


if __name__ == "__main__":
    main()
