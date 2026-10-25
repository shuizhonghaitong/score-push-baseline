"""Solar Filament Segmentation Challenge 2026 - pipeline v1.

Strategy
--------
Two complementary models are trained on the official training split:

1. ``unet``  - a semantic U-Net (ResNet34 encoder, smp) that predicts the
   binary "filament foreground" probability map.  Instances are extracted
   from the thresholded map with 8-connected components plus small-part
   filtering.  Cheap, smooth boundaries.

2. ``mrcnn`` - an instance Mask R-CNN (ResNet50-FPN-v2, COCO pretraining,
   small-scale anchors) that predicts one mask per filament directly.
   Handles separated/touching instances better.

Both are validated with the *official* panoptic-quality implementation
(grouped by annotator-image, TP iff IoU > 0.5, PQ = sum(TP IoU) /
(|TP| + 0.5|FP| + 0.5|FN|) over all rows) on a grouped (per physical
observation) 80/20 train/val split.

Outputs (in /kaggle/working)
----------------------------
- ``submission.csv``            best candidate by local val PQ
- ``submission_unet.csv`` / ``submission_mrcnn.csv`` / ``submission_merge.csv``
- ``val_predictions.npz``       val instance masks (all candidates) for offline analysis
- ``val_gt.npz``                val GT layers (per annotator record)
- ``run_report.json``           metrics, config, timings
- ``best_unet.pt`` / ``best_mrcnn.pt``  checkpoints
"""
from __future__ import annotations

import argparse
import base64
import sys
import json
import math
import os
import random
import shutil
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch  # noqa: E402  (module-level decorators need it)

IMG = 2048          # native image size (fixed by the competition)
WORK = Path(os.environ.get("FILAMENT_WORK_DIR", "/kaggle/working"))
try:
    WORK.mkdir(parents=True, exist_ok=True)
except OSError:  # local smoke tests
    WORK = Path.cwd() / "working_local"
    WORK.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------- #
# 0. Reproducibility
# --------------------------------------------------------------------------- #
def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    import torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


# --------------------------------------------------------------------------- #
# 1. Data loading
# --------------------------------------------------------------------------- #
def find_competition_root() -> Path:
    base = Path("/kaggle/input")
    anns = list(base.rglob("MAGFiLO_1.0_Annotations_kaggle2026_train.json"))
    if len(anns) != 1:
        raise RuntimeError(f"expected exactly 1 train annotation file, found {anns}")
    return anns[0].parent.parent  # MAGFiLO_1.0_Kaggle_2026


@dataclass
class Record:
    """One annotator-image record (a COCO 'image' entry).

    Image pixels and annotation masks are loaded lazily (disk-backed) because
    the full record set is larger than the notebook's RAM budget.
    """
    record_id: str          # e.g. "010401-20160920230134Lh"
    stem: str               # physical observation, e.g. "20160920230134Lh"
    img_path: Path
    ann_items: list
    _image: "np.ndarray | None" = field(default=None, repr=False)
    _masks: "list | None" = field(default=None, repr=False)
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
        if self._masks is None:
            self._masks = [self._coco.annToMask(a) > 0 for a in self.ann_items]
        return self._masks

    def release(self) -> None:
        self._image = None
        self._masks = None


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
    # warm the OS page cache for one image to validate the layout
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
    print(f"[split] obs={len(stems)} train_obs={len(stems)-n_val} val_obs={n_val} "
          f"train_rec={len(train)} val_rec={len(val)}", flush=True)
    return train, val


# --------------------------------------------------------------------------- #
# 2. Official panoptic-quality (faithful to the Self-Evaluation notebook)
# --------------------------------------------------------------------------- #
def _decode_counts(counts, height: int, width: int) -> np.ndarray:
    from pycocotools import mask as mu
    cands = [counts]
    if isinstance(counts, str):
        try:
            cands.append(base64.b64decode(counts))
        except Exception:
            pass
    last = None
    for c in cands:
        try:
            return (mu.decode({"size": [height, width], "counts": c}) > 0).astype(np.uint8)
        except Exception as e:  # pragma: no cover
            last = e
    raise last


def rles_to_layers(rles, height: int, width: int) -> np.ndarray:
    if not rles:
        return np.zeros((0, height, width), dtype=np.float32)
    return np.stack([_decode_counts(r, height, width) for r in rles], axis=0).astype(np.float32)


def pq_from_layers(rows: list[tuple[np.ndarray, np.ndarray]], iou_threshold: float = 0.5) -> dict:
    """rows: list of (gt_layers (n_gt,H,W), pred_layers (n_pred,H,W))."""
    tp_sum = 0.0
    tp = fp = fn = 0
    per_image_iou = []
    for gt, pred in rows:
        n_gt, n_pred = gt.shape[0], pred.shape[0]
        if n_gt == 0:
            fp += n_pred
            continue
        if n_pred == 0:
            fn += n_gt
            continue
        gtf = gt.reshape(n_gt, -1)
        ptf = pred.reshape(n_pred, -1)
        inter = gtf @ ptf.T
        gt_area = gtf.sum(axis=1, keepdims=True)
        pred_area = ptf.sum(axis=1, keepdims=True).T
        union = gt_area + pred_area - inter
        iou = np.where(union > 0, inter / np.maximum(union, 1), 0.0)
        hit = iou > iou_threshold
        tp_iou = iou[hit]
        tp += int(hit.sum())
        tp_sum += float(tp_iou.sum())
        fp += int((hit.sum(axis=0) == 0).sum())
        fn += int((hit.sum(axis=1) == 0).sum())
        per_image_iou.append(float(tp_iou.mean()) if tp_iou.size else 0.0)
    denom = tp + 0.5 * fp + 0.5 * fn
    pq = tp_sum / denom if denom > 0 else 0.0
    return {"pq": pq, "tp": tp, "fp": fp, "fn": fn,
            "mean_tp_iou": tp_sum / tp if tp else 0.0,
            "mean_per_image_tp_iou": float(np.mean(per_image_iou)) if per_image_iou else 0.0}


def val_rows(records: list[Record], pred_by_stem: dict[str, list[np.ndarray]]) -> list:
    """Official rows: one per annotator record; preds grouped by physical stem."""
    rows = []
    for rec in records:
        gt = np.stack(rec.masks, axis=0).astype(np.float32) if rec.masks else np.zeros((0, IMG, IMG), np.float32)
        pred = np.stack(pred_by_stem[rec.stem], axis=0).astype(np.float32) if pred_by_stem.get(rec.stem) else np.zeros((0, IMG, IMG), np.float32)
        rows.append((gt, pred))
    return rows


# --------------------------------------------------------------------------- #
# 3. Instance post-processing
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
            out.append(part.astype(bool))
    return out


def merge_overlapping(masks: list[np.ndarray]) -> list[np.ndarray]:
    """Merge instances that share any pixel (graph over positive overlaps)."""
    masks = [m for m in masks if m.any()]
    if len(masks) < 2:
        return masks
    flat = [m.reshape(-1) for m in masks]
    n = len(masks)
    parent = list(range(n))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    # coarse grid filter: two masks can only overlap if their bboxes intersect
    boxes = []
    for m in masks:
        ys, xs = np.where(m)
        boxes.append((ys.min(), ys.max(), xs.min(), xs.max()))
    for i in range(n):
        for j in range(i + 1, n):
            a, b = boxes[i], boxes[j]
            if a[0] > b[1] or b[0] > a[1] or a[2] > b[3] or b[2] > a[3]:
                continue
            if (masks[i] & masks[j]).any():
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[ri] = rj
    groups: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(i)
    out = []
    for members in groups.values():
        merged = np.logical_or.reduce([masks[k] for k in members])
        out.append(merged)
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


def masks_to_rows(pred_by_stem: dict[str, list[np.ndarray]]) -> pd.DataFrame:
    """submission format: filament_id = '<stem>_<k>', segmentation_rle = b64 counts."""
    from pycocotools import mask as mu
    rows = []
    for stem in sorted(pred_by_stem):
        for k, m in enumerate(pred_by_stem[stem], start=1):
            if not m.any():
                continue
            rle = mu.encode(np.asfortranarray(m.astype(np.uint8)))
            counts = rle["counts"]
            if isinstance(counts, bytes):
                counts = base64.b64encode(counts).decode("ascii")
            rows.append({"filament_id": f"{stem}_{k}", "segmentation_rle": counts})
    return pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])


# --------------------------------------------------------------------------- #
# 4. U-Net (semantic)
# --------------------------------------------------------------------------- #
def train_unet(train: list[Record], val: list[Record], epochs: int, bs: int, size: int,
               device: "torch.device") -> "nn.Module":
    import torch
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
                ], bbox_params=None)

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
    best_val, best_state, best_epoch = -1.0, None, 0
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        tot, n_it = 0.0, 0
        for x, y in ld_tr:
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            loss = dice_bce(model(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            sched.step()
            tot += loss.item()
            n_it += 1
        # quick val foreground dice
        model.eval()
        tp = fp = fn = 0
        with torch.inference_mode():
            for x, y in ld_val:
                x = x.to(device, non_blocking=True)
                p = (torch.sigmoid(model(x)) > 0.5).float().squeeze(1)
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
    return model


@torch.inference_mode()
def unet_predict(model, image: np.ndarray, size: int, device, thr: float = 0.5) -> np.ndarray:
    x = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    x = (x - 0.5) / 0.5
    t = torch.from_numpy(x).unsqueeze(0).repeat(3, 1, 1).unsqueeze(0).to(device)
    prob = torch.sigmoid(model(t))[0, 0].cpu().numpy()
    mask = (prob > thr)
    return cv2.resize(mask.astype(np.uint8), (IMG, IMG), interpolation=cv2.INTER_NEAREST).astype(bool)


# --------------------------------------------------------------------------- #
# 5. Mask R-CNN (instance)
# --------------------------------------------------------------------------- #
def build_mrcnn(size: int, n_train: int) -> "nn.Module":
    import torch
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
        """Whole-image geometric flips/rot90 (mask-safe) + photometric jitter on the
        image only.  No scaling or free rotation: they would require bbox/mask
        reconstruction and blur thin structures."""
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
        import torch
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
            boxes.append([x0, y0, w, h])
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
            "iscrowd": torch.zeros(len(boxes), dtype=torch.int64),
        }
        return image_t, target


def train_mrcnn(train: list[Record], val: list[Record], epochs: int, bs: int, size: int,
                device: "torch.device") -> "nn.Module":
    import torch
    from torch.utils.data import DataLoader

    ds = MRCNNDataset(train, size, augment=True)
    ld = DataLoader(ds, batch_size=bs, shuffle=True, num_workers=4, pin_memory=True,
                    collate_fn=lambda b: (list(zip(*b))[0], list(zip(*b))[1]))
    model = build_mrcnn(size, len(train)).to(device)
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
            targets = [{k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v for k, v in t.items()} for t in targets]
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
                print(f"[mrcnn] e{epoch} it{n_it}/{len(ld)} loss={tot / n_it:.4f} {lv} t={time.time() - t0:.0f}s", flush=True)
        # val proxy: instance IoU@0.5 F1 + fg dice
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
    return model


@torch.inference_mode()
def mrcnn_detect(model, image: np.ndarray, size: int, device, score_thr=0.5, mask_thr=0.5):
    import torch
    x = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
    t = torch.from_numpy(x.transpose(2, 0, 1).copy()).float() / 255.0
    t = t.unsqueeze(0).to(device)
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
        pred = mrcnn_instances(det_masks, det_scores, score_thr, mask_thr)
        gt = rec.masks
        inter = np.zeros((len(gt), len(pred)))
        for i, g in enumerate(gt):
            for j, p in enumerate(pred):
                inter[i, j] = (g & p).sum()
        if not (gt and pred):
            fp += len(pred)
            fn += len(gt)
            union_gt = np.logical_or.reduce(gt).astype(bool) if gt else np.zeros((IMG, IMG), bool)
            union_pred = np.logical_or.reduce(pred).astype(bool) if pred else np.zeros((IMG, IMG), bool)
        else:
            gt_a = np.array([g.sum() for g in gt])
            p_a = np.array([p.sum() for p in pred])
            union = gt_a[:, None] + p_a[None, :] - inter
            iou = np.where(union > 0, inter / np.maximum(union, 1), 0)
            # greedy highest-IoU-first matching
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
            union_gt = np.logical_or.reduce(gt).astype(bool)
            union_pred = np.logical_or.reduce(pred).astype(bool)
        dice_tp += (union_gt & union_pred).sum()
        dice_fp += (union_pred & ~union_gt).sum()
        dice_fn += (union_gt & ~union_pred).sum()
    f1 = 2 * tp / max(1, 2 * tp + fp + fn)
    dice = 2 * dice_tp / max(1, 2 * dice_tp + dice_fp + dice_fn)
    return {"f1": round(float(f1), 4), "fg_dice": round(float(dice), 4)}


# --------------------------------------------------------------------------- #
# 6. Full validation (official PQ) and test inference
# --------------------------------------------------------------------------- #
@torch.inference_mode()
def unet_full_val(model, val, size, device, thr):
    out = {}
    for rec in val:
        m = unet_predict(model, rec.image, size, device, thr)
        out[rec.stem] = instances_from_semantic(m)
    return out


@torch.inference_mode()
def mrcnn_full_val(model, val, size, device, score_thr, mask_thr):
    out = {}
    for rec in val:
        masks, scores = mrcnn_detect(model, rec.image, size, device, score_thr, mask_thr)
        out[rec.stem] = mrcnn_instances(masks, scores, score_thr, mask_thr)
    return out


def merged_candidates(cands: dict[str, list]) -> dict[str, list]:
    out = {}
    for stem in cands[0]:
        all_masks = []
        for c in cands.values():
            all_masks.extend(c.get(stem, []))
        out[stem] = merge_overlapping(all_masks)
    return out


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--unet-epochs", type=int, default=10)
    ap.add_argument("--mrcnn-epochs", type=int, default=10)
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--val-frac", type=float, default=0.2)
    args = ap.parse_args()
    # CPU kernel configuration (Kaggle script CLI args are not delivered via push)
    args.unet_epochs = 10
    args.mrcnn_epochs = 0
    t_start = time.time()
    seed_everything(42)
    torch.set_num_threads(2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[cfg] {vars(args)} device={device}", flush=True)

    root = find_competition_root()
    records, test_files = load_all(root)
    train, val = grouped_split(records, args.val_frac, 42)

    # ---------------- U-Net (train, or load from an attached kernel output) ---------------- #
    if args.unet_epochs > 0:
        unet = train_unet(train, val, args.unet_epochs, bs=8, size=args.size, device=device)
    else:
        import segmentation_models_pytorch as smp
        ck = next(Path("/kaggle/input").rglob("best_unet.pt"), None)
        if ck is None:
            raise RuntimeError("best_unet.pt not found in /kaggle/input (attach the CPU kernel output)")
        print(f"[unet] loading {ck}", flush=True)
        unet = smp.Unet("resnet34", in_channels=3, classes=1, encoder_weights=None).to(device)
        unet.load_state_dict(torch.load(ck, map_location=device, weights_only=False))
    unet_val = unet_full_val(unet, val, args.size, device, thr=0.5)
    unet_pq = pq_from_layers(val_rows(val, unet_val))
    print(f"[eval] unet val PQ={unet_pq['pq']:.4f} {unet_pq}", flush=True)

    # ---------------- MRCNN (skippable for CPU fallback runs) ---------------- #
    if args.mrcnn_epochs > 0:
        mrcnn = train_mrcnn(train, val, args.mrcnn_epochs, bs=1, size=args.size, device=device)
        mrcnn_val = mrcnn_full_val(mrcnn, val, args.size, device, 0.5, 0.5)
        mrcnn_pq = pq_from_layers(val_rows(val, mrcnn_val))
        print(f"[eval] mrcnn val PQ={mrcnn_pq['pq']:.4f} {mrcnn_pq}", flush=True)
    else:
        mrcnn_val = {r.stem: [] for r in val}
        mrcnn_pq = {"pq": 0.0}
    merge_val = merged_candidates([unet_val, mrcnn_val])
    merge_pq = pq_from_layers(val_rows(val, merge_val))
    print(f"[eval] merge val PQ={merge_pq['pq']:.4f} {merge_pq}", flush=True)

    # ---------------- Test inference ---------------- #
    test_dir = root / "test" / "test_images"
    unet_test, mrcnn_test = {}, {}
    for stem in test_files:
        img = cv2.imread(str(test_dir / f"{stem}.jpeg"), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise RuntimeError(f"missing test image {stem}")
        unet_test[stem] = instances_from_semantic(unet_predict(unet, img, args.size, device, 0.5))
        if args.mrcnn_epochs > 0:
            masks, scores = mrcnn_detect(mrcnn, img, args.size, device, 0.5, 0.5)
            mrcnn_test[stem] = mrcnn_instances(masks, scores, 0.5, 0.5)
        else:
            mrcnn_test[stem] = []
    merge_test = merged_candidates([unet_test, mrcnn_test])

    cands = {"unet": (unet_pq, unet_test), "mrcnn": (mrcnn_pq, mrcnn_test), "merge": (merge_pq, merge_test)}
    for name, (pq, preds) in cands.items():
        df = masks_to_rows(preds)
        df.to_csv(WORK / f"submission_{name}.csv", index=False)
        print(f"[out] submission_{name}.csv rows={len(df)} valPQ={pq['pq']:.4f}", flush=True)

    best = max(cands, key=lambda k: cands[k][0]["pq"])
    shutil.copy(WORK / f"submission_{best}.csv", WORK / "submission.csv")
    print(f"[out] BEST candidate={best} valPQ={cands[best][0]['pq']:.4f}", flush=True)

    # ---------------- Persist val data for offline analysis ---------------- #
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
    # save a compact val GT dump (object array: one (n_masks, H, W) uint8 per observation)
    gt_stems = {}
    for rec in val:
        if rec.masks:
            gt_stems[rec.stem] = np.stack(rec.masks).astype(np.uint8)
    stems = sorted(gt_stems)
    np.savez_compressed(WORK / "val_gt.npz",
                        stems=np.array(stems),
                        gt=np.array([gt_stems[s] for s in stems], dtype=object))
    print("[done]", json.dumps(report["val_pq"]), flush=True)


if __name__ == "__main__":
    main()
