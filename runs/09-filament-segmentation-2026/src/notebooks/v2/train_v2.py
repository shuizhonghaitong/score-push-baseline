"""Solar Filament Segmentation Challenge 2026 - pipeline v2 (memory-safe).

Builds on the v1 detector (attached via kernel_sources, file best_mrcnn.pt):
a native-resolution 256px crop refiner (CropUNet) with 4-flip TTA and
topology-safe selection, plus detector-seed Voronoi splitting for crowded
groups and positive-overlap merging.

Memory model (same as v1): predictions are kept as base64 COCO RLE strings;
pixels are materialised transiently, one observation at a time.

Candidates (val-PQ ranked): refiner (TTA), refiner_notta, detector_070,
detector_050.  Best -> submission.csv.

Outputs (in /kaggle/working):
- submission.csv + submission_{refiner,refiner_notta,detector_070,detector_050}.csv
- run_report.json, val_stems.json
- val_predictions_{candidate}.npz (per-stem newline-joined b64 RLE strings)
- best_refiner.pt
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import random
import shutil
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

IMG = 2048
CROP = 256
WORK = Path(os.environ.get("FILAMENT_WORK_DIR", "/kaggle/working"))
try:
    WORK.mkdir(parents=True, exist_ok=True)
except OSError:
    WORK = Path.cwd() / "working_local"
    WORK.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(Path(__file__).parent))
from refiner_lib import (TtaConfig, select_topology_safe,  # noqa: E402
                         square_bounds, split_by_detector_seeds)


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


# --------------------------------------------------------------------------- #
# Data (lazy, same conventions as v1)
# --------------------------------------------------------------------------- #
def find_competition_root() -> Path:
    env_root = os.environ.get("FILAMENT_DATA_ROOT")
    if env_root:
        return Path(env_root)
    anns = list(Path("/kaggle/input").rglob("MAGFiLO_1.0_Annotations_kaggle2026_train.json"))
    if len(anns) != 1:
        raise RuntimeError(f"expected 1 annotation file, found {anns}")
    return anns[0].parent.parent


@dataclass
class Record:
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
    coco = COCO(str(root / "train" / "MAGFiLO_1.0_Annotations_kaggle2026_train.json"))
    records: list[Record] = []
    for rec_id, info in coco.imgs.items():
        stem = Path(info["file_name"]).stem
        records.append(Record(record_id=str(rec_id), stem=stem,
                              img_path=root / "train" / "train_images" / info["file_name"],
                              ann_items=list(coco.imgToAnns[rec_id]), _coco=coco))
    test_files = sorted(p.stem for p in (root / "test" / "test_images").glob("*.jpeg"))
    print(f"[data] records={len(records)} obs={len({r.stem for r in records})} "
          f"test={len(test_files)}", flush=True)
    return records, test_files


def grouped_split(records, val_frac: float = 0.2, seed: int = 42):
    stems = sorted({r.stem for r in records})
    rng = random.Random(seed)
    rng.shuffle(stems)
    n_val = max(1, int(round(len(stems) * val_frac)))
    val_stems = set(stems[:n_val])
    train = [r for r in records if r.stem not in val_stems]
    val = [r for r in records if r.stem in val_stems]
    print(f"[split] train_obs={len(stems) - n_val} val_obs={n_val} "
          f"train_rec={len(train)} val_rec={len(val)}", flush=True)
    return train, val


# --------------------------------------------------------------------------- #
# RLE + streaming official PQ
# --------------------------------------------------------------------------- #
def mask_to_rle_b64(mask: np.ndarray) -> str:
    # Official submission format: raw pycocotools compressed RLE counts (ASCII).
    from pycocotools import mask as mu
    rle = mu.encode(np.asfortranarray(mask.astype(np.uint8)))
    counts = rle["counts"]
    if isinstance(counts, bytes):
        return counts.decode("latin-1")
    return str(counts)


def masks_to_rles(masks) -> list[str]:
    return [mask_to_rle_b64(m) for m in masks if m.any()]


def rle_b64_to_mask(s: str) -> np.ndarray:
    # Accept raw RLE counts (new format) first; fall back to base64 (legacy npz).
    from pycocotools import mask as mu
    try:
        return (mu.decode({"size": [IMG, IMG], "counts": s.encode("latin-1")}) > 0)
    except Exception:
        counts = base64.b64decode(s)
        return (mu.decode({"size": [IMG, IMG], "counts": counts}) > 0)


def rles_to_masks(rles: list[str]) -> list[np.ndarray]:
    return [rle_b64_to_mask(r) for r in rles]


def pq_stream(records, pred_rles_by_stem: dict, iou_threshold: float = 0.5) -> dict:
    tp_sum = 0.0
    tp = fp = fn = 0
    for rec in records:
        gt = rec.masks
        pred = rles_to_masks(pred_rles_by_stem.get(rec.stem, []))
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


def merge_overlapping(masks: list[np.ndarray]) -> list[np.ndarray]:
    """Label-image based merge (fast)."""
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
        for lab in np.unique(L[m]):
            j = int(lab) - 1
            if j != i:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[ri] = rj
    groups: dict[int, list[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    return [masks[g[0]] if len(g) == 1 else np.logical_or.reduce([masks[k] for k in g])
            for g in groups.values()]


# pycocotools compressed RLE of an all-zero 2048x2048 mask (verified: encodes to b"PPPP4").
EMPTY_RLE = "PPPP4"


def rles_to_rows(pred_rles_by_stem: dict) -> pd.DataFrame:
    rows = []
    for stem in sorted(pred_rles_by_stem):
        # Stems with no predicted instance get one empty-mask row so every
        # test image is covered (matches the 1st-place notebook convention).
        rles = pred_rles_by_stem[stem] or [EMPTY_RLE]
        for k, r in enumerate(rles, start=1):
            rows.append({"filament_id": f"{stem}_{k}", "segmentation_rle": r})
    return pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"])


# --------------------------------------------------------------------------- #
# Detector (v1 Mask R-CNN)
# --------------------------------------------------------------------------- #
def build_mrcnn(size: int) -> nn.Module:
    from torchvision.models.detection import maskrcnn_resnet50_fpn_v2
    from torchvision.models.detection.anchor_utils import AnchorGenerator
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor
    from torchvision.ops import MultiScaleRoIAlign
    model = maskrcnn_resnet50_fpn_v2(weights=None, weights_backbone=None)
    box_in = model.roi_heads.box_predictor.cls_score.in_features
    mask_in = model.roi_heads.mask_predictor.conv5_mask.in_channels
    mask_hid = model.roi_heads.mask_predictor.conv5_mask.out_channels
    model.roi_heads.box_predictor = FastRCNNPredictor(box_in, 2)
    model.roi_heads.mask_predictor = MaskRCNNPredictor(mask_in, mask_hid, 2)
    model.rpn.anchor_generator = AnchorGenerator(sizes=((16,), (32,), (64,), (128,), (256,)),
                                                 aspect_ratios=((0.5, 1.0, 2.0),) * 5)
    model.rpn.stride = [8, 16, 32, 64, 128]
    model.roi_heads.mask_roi_pool = MultiScaleRoIAlign(["0", "1", "2", "3"], output_size=28, sampling_ratio=2)
    model.transform.min_size = (size,)
    model.transform.max_size = size
    return model


@torch.inference_mode()
def detect(model, image: np.ndarray, size: int, device):
    x = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
    if x.ndim == 2:
        x = np.stack([x] * 3, axis=-1)
    t = torch.from_numpy(x.transpose(2, 0, 1).copy()).float().div_(255.0).to(device)
    with torch.amp.autocast("cuda", dtype=torch.float16):
        out = model([t])[0]
    scores = out["scores"].cpu().numpy()
    masks = (out["masks"][:, 0].cpu().numpy() >= 0.5)
    keep = np.flatnonzero(scores >= 0.30)
    return masks[keep], scores[keep]


def remove_small(mask: np.ndarray, min_px: int) -> np.ndarray:
    n, lab = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    out = np.zeros_like(mask)
    for i in range(1, n):
        if (lab == i).sum() >= min_px:
            out[lab == i] = 1
    return out


@torch.inference_mode()
def detect_flips(model, image: np.ndarray, size: int, device):
    """4-flip detection; masks returned in original orientation at (size, size)."""
    base = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
    if base.ndim == 2:
        base = np.stack([base] * 3, axis=-1)
    all_masks, all_scores = [], []
    for name in ("identity", "hflip", "vflip", "hvflip"):
        x = _FLIPS[name](base)
        t = torch.from_numpy(x.transpose(2, 0, 1).copy()).float().div_(255.0).to(device)
        with torch.amp.autocast("cuda", dtype=torch.float16):
            out = model([t])[0]
        scores = out["scores"].cpu().numpy()
        masks = (out["masks"][:, 0].cpu().numpy() >= 0.5)
        keep = np.flatnonzero(scores >= 0.30)
        all_masks.append(_FLIPS[name](masks[keep]))
        all_scores.append(scores[keep])
    if all_masks:
        return np.concatenate(all_masks, axis=0), np.concatenate(all_scores, axis=0)
    return np.zeros((0, size, size), bool), np.zeros((0,), float)


def _instances_from_raw(raw_masks, raw_scores, score_thr: float, min_px: int = 32):
    insts, inst_scores = [], []
    for k in np.flatnonzero(raw_scores >= score_thr):
        m = (raw_masks[k] >= 0.5)
        if not m.any():
            continue
        mm = cv2.resize(m.astype(np.uint8), (IMG, IMG), interpolation=cv2.INTER_NEAREST).astype(bool)
        mm = remove_small(mm, min_px)
        if mm.any():
            insts.append(mm)
            inst_scores.append(float(raw_scores[k]))
    groups = merge_overlapping(insts) if len(insts) > 1 else insts
    # map each merged group back to member indices for seed splitting
    if len(insts) > 1:
        L = np.zeros((IMG, IMG), np.int32)
        for i, m in enumerate(insts):
            L[m] = i + 1
        member_groups = []
        for g in groups:
            member_groups.append(sorted(int(lab) - 1 for lab in np.unique(L[g])))
    else:
        member_groups = [[0]] if insts else []
    group_scores = [max(inst_scores[mk] for mk in members) for members in member_groups]
    return groups, group_scores, member_groups, insts, inst_scores


@torch.inference_mode()
def detector_groups(model, image: np.ndarray, size: int, device, score_thr: float, min_px: int = 32):
    """Detector instances at score_thr -> (groups, group scores, member idx, insts, scores)."""
    raw_masks, raw_scores = detect(model, image, size, device)
    return _instances_from_raw(raw_masks, raw_scores, score_thr, min_px)


# --------------------------------------------------------------------------- #
# Refiner model
# --------------------------------------------------------------------------- #
class ConvBlock(nn.Sequential):
    def __init__(self, c_in, c_out):
        super().__init__(nn.Conv2d(c_in, c_out, 3, padding=1, bias=False), nn.BatchNorm2d(c_out),
                         nn.ReLU(inplace=True), nn.Conv2d(c_out, c_out, 3, padding=1, bias=False),
                         nn.BatchNorm2d(c_out), nn.ReLU(inplace=True))


class CropUNet(nn.Module):
    def __init__(self):
        super().__init__()
        from torchvision.models import resnet18
        enc = resnet18(weights=None)
        self.stem = nn.Sequential(enc.conv1, enc.bn1, enc.relu)
        self.pool = enc.maxpool
        self.layer1, self.layer2, self.layer3, self.layer4 = enc.layer1, enc.layer2, enc.layer3, enc.layer4
        self.up4 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.dec4 = ConvBlock(512, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.dec3 = ConvBlock(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.dec2 = ConvBlock(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, 2, 2)
        self.dec1 = ConvBlock(96, 32)
        self.up0 = nn.ConvTranspose2d(32, 16, 2, 2)
        self.head = nn.Conv2d(16, 1, 1)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, x):
        x = (x - self.mean) / self.std
        x0 = self.stem(x)
        x1 = self.layer1(self.pool(x0))
        x2 = self.layer2(x1)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        v = self.dec4(torch.cat([self.up4(x4), x3], 1))
        v = self.dec3(torch.cat([self.up3(v), x2], 1))
        v = self.dec2(torch.cat([self.up2(v), x1], 1))
        v = self.dec1(torch.cat([self.up1(v), x0], 1))
        return self.head(self.up0(v))


def build_refiner_tasks(records: list[Record], detector, size: int, device,
                        score_thr: float, max_tasks: int = 15000) -> list[dict]:
    """One task per detector group: native square crop + GT-union target.

    Crops stored as uint8 (256x256) and targets as uint8 to bound RAM.
    """
    tasks = []
    for i, rec in enumerate(records):
        groups, _, _, _, _ = detector_groups(detector, rec.image, size, device, score_thr)
        if not groups:
            continue
        gt_union = np.zeros((IMG, IMG), bool)
        for m in rec.masks:
            gt_union |= m
        for g in groups:
            x0, y0, x1, y1 = square_bounds(g)
            target = cv2.resize(gt_union[y0:y1, x0:x1].astype(np.uint8), (CROP, CROP),
                                interpolation=cv2.INTER_NEAREST)
            if target.sum() < 64:
                continue
            crop = cv2.resize(rec.image[y0:y1, x0:x1], (CROP, CROP), interpolation=cv2.INTER_LINEAR)
            tasks.append({"crop": crop.astype(np.uint8), "target": target.astype(np.uint8),
                          "bounds": (x0, y0, x1, y1)})
        if (i + 1) % 100 == 0:
            print(f"[tasks] {i + 1}/{len(records)} obs, n={len(tasks)} t={time.time():.0f}", flush=True)
    if len(tasks) > max_tasks:
        keep = set(random.Random(42).sample(range(len(tasks)), max_tasks))
        tasks = [t for i, t in enumerate(tasks) if i in keep]
    return tasks


def train_refiner(tasks: list[dict], device, epochs: int, bs: int) -> CropUNet:
    try:
        import albumentations as A
    except ImportError:
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "albumentations"])
        import albumentations as A
    from torch.utils.data import DataLoader, Dataset

    idx = list(range(len(tasks)))
    random.Random(42).shuffle(idx)
    n_val = max(1, len(idx) // 10)
    val_idx, tr_idx = idx[:n_val], idx[n_val:]

    class TDS(Dataset):
        def __init__(self, ids, augment):
            self.ids = ids
            self.augment = augment
            self.tr = None
            if augment:
                self.tr = A.Compose([
                    A.RandomBrightnessContrast(p=0.3),
                    A.OneOf([A.HorizontalFlip(p=1.0), A.VerticalFlip(p=1.0), A.Transpose(p=1.0)]),
                ])

        def __len__(self):
            return len(self.ids)

        def __getitem__(self, i):
            t = tasks[self.ids[i]]
            crop, target = t["crop"], t["target"]
            if self.tr is not None:
                r = self.tr(image=crop, mask=target)
                crop, target = r["image"], r["mask"]
            x = torch.from_numpy(crop.astype(np.float32) / 255.0).unsqueeze(0).repeat(3, 1, 1)
            y = torch.from_numpy(target.astype(np.float32))
            return x, y

    model = CropUNet().to(device)
    ds_tr, ds_val = TDS(tr_idx, True), TDS(val_idx, False)
    ld_tr = DataLoader(ds_tr, batch_size=bs, shuffle=True, num_workers=0, drop_last=True)
    ld_val = DataLoader(ds_val, batch_size=bs, num_workers=0)

    def dice_bce(logits, y):
        p = torch.sigmoid(logits)
        bce = nn.functional.binary_cross_entropy_with_logits(logits, y)
        inter = (p * y).sum((2, 3))
        union = p.sum((2, 3)) + y.sum((2, 3))
        dice = 1.0 - (2 * inter + 1.0) / (union + 1.0)
        return bce + dice.mean()

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs * max(1, len(ld_tr)))
    best, best_state, t0 = -1.0, None, time.time()
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
        model.eval()
        tp = fp = fn = 0
        with torch.inference_mode():
            for x, y in ld_val:
                x = x.to(device, non_blocking=True)
                p = (torch.sigmoid(model(x)) > 0.5).float()
                tp += (p * y.to(device)).sum().item()
                fp += (p * (1 - y.to(device))).sum().item()
                fn += ((1 - p) * y.to(device)).sum().item()
        vd = 2 * tp / max(1, 2 * tp + fp + fn)
        print(f"[refiner] epoch {epoch}/{epochs} loss={tot / max(1, n_it):.4f} val_fg_dice={vd:.4f} "
              f"t={time.time() - t0:.0f}s", flush=True)
        if vd > best:
            best = vd
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
        torch.save(best_state, WORK / "best_refiner.pt")
    return model


_FLIPS = {"identity": lambda a: a,
          "hflip": lambda a: a[:, ::-1].copy(),
          "vflip": lambda a: a[::-1, :].copy(),
          "hvflip": lambda a: a[::-1, ::-1].copy()}


@torch.inference_mode()
def refiner_probs(model, crop: np.ndarray, device, transforms=("identity", "hflip", "vflip", "hvflip")):
    """2D grayscale crop in, 2D probability maps out (flips are self-inverse)."""
    probs = []
    for tf in transforms:
        c = _FLIPS[tf](crop)
        x = torch.from_numpy(c.astype(np.float32) / 255.0).unsqueeze(0).repeat(3, 1, 1).unsqueeze(0).to(device)
        p = torch.sigmoid(model(x))[0, 0].cpu().numpy()
        probs.append(_FLIPS[tf](p))
    return probs[0], np.mean(probs, axis=0)


def decode_task(prob: np.ndarray, bounds, fallback: np.ndarray, threshold: float) -> np.ndarray:
    x0, y0, x1, y1 = bounds
    side = x1 - x0
    p2 = cv2.resize(prob, (side, side), interpolation=cv2.INTER_LINEAR)
    m = (p2 >= threshold).astype(np.uint8)
    if not m.any():
        return fallback
    out = np.zeros((IMG, IMG), bool)
    out[y0:y1, x0:x1] = m.astype(bool)
    return out


def process_obs_masks(refiner, image: np.ndarray, device, tta_cfg: TtaConfig, use_tta: bool,
                      groups, member_groups, insts, insts_scores) -> list[np.ndarray]:
    """Refiner-cropped masks (TTA or not) -> seed splits -> merge, over detector groups."""
    out_insts = []
    for g, members in zip(groups, member_groups):
        x0, y0, x1, y1 = square_bounds(g)
        crop = cv2.resize(image[y0:y1, x0:x1], (CROP, CROP), interpolation=cv2.INTER_LINEAR)
        if use_tta:
            single_p, tta_p = refiner_probs(refiner, crop, device)
            prob, _, _ = select_topology_safe(single_p, tta_p, tta_cfg)
        else:
            prob, _ = refiner_probs(refiner, crop, device, transforms=("identity",))
        mask = decode_task(prob, (x0, y0, x1, y1), g, tta_cfg.threshold)
        if len(members) >= 2:
            parts = split_by_detector_seeds(mask, [insts[mk] for mk in members],
                                            np.array([insts_scores[mk] for mk in members]))
            out_insts.extend(parts)
        else:
            out_insts.append(mask)
    return merge_overlapping(out_insts) if len(out_insts) > 1 else out_insts


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--refiner-epochs", type=int, default=20)
    ap.add_argument("--thr", type=float, default=0.70)
    args = ap.parse_args()
    if not os.environ.get("FILAMENT_SMOKE"):
        args.size = 1024
        args.refiner_epochs = 20
    t0 = time.time()
    seed_everything(42)
    torch.set_num_threads(min(4, os.cpu_count() or 1))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[cfg] {vars(args)} device={device}", flush=True)

    root = find_competition_root()
    records, test_files = load_all(root)
    train, val = grouped_split(records, args.val_frac, 42)

    env_ck = os.environ.get("FILAMENT_DETECTOR_CKPT")
    if env_ck:
        ck = Path(env_ck)
    else:
        ck = None
        for cand in Path("/kaggle/input").rglob("best_mrcnn.pt"):
            ck = cand
            break
    if ck is None:
        raise RuntimeError("best_mrcnn.pt not found in /kaggle/input (attach the v1 kernel output)")
    print(f"[detector] loading {ck}", flush=True)
    detector = build_mrcnn(args.size)
    detector.load_state_dict(torch.load(ck, map_location="cpu", weights_only=False))
    detector = detector.to(device).eval()

    print("[tasks] building refiner tasks...", flush=True)
    tasks = build_refiner_tasks(train, detector, args.size, device, 0.70)
    print(f"[tasks] n={len(tasks)}", flush=True)
    refiner_ok = len(tasks) >= 64
    if refiner_ok:
        refiner = train_refiner(tasks, device, args.refiner_epochs, bs=32)
    else:
        refiner = None
        print("[warn] too few refiner tasks, detector-only candidates", flush=True)
    del tasks

    tta_cfg = TtaConfig(threshold=args.thr)

    def infer_candidates(recs: list[Record] | list[tuple[str, np.ndarray]], with_refiner: bool):
        out = {"refiner": {}, "refiner_notta": {}, "detector_070": {}, "detector_050": {},
               "detector_tta": {}}
        for i, item in enumerate(recs):
            stem, image = item
            raw_masks, raw_scores = detect(detector, image, args.size, device)
            g070, _, m070, i070, is070 = _instances_from_raw(raw_masks, raw_scores, 0.70)
            if with_refiner and refiner is not None:
                out["refiner"][stem] = masks_to_rles(
                    process_obs_masks(refiner, image, device, tta_cfg, True, g070, m070, i070, is070))
                out["refiner_notta"][stem] = masks_to_rles(
                    process_obs_masks(refiner, image, device, tta_cfg, False, g070, m070, i070, is070))
            out["detector_070"][stem] = masks_to_rles(g070)
            g050, _, _, _, _ = _instances_from_raw(raw_masks, raw_scores, 0.50)
            out["detector_050"][stem] = masks_to_rles(g050)
            del raw_masks
            raw4, scores4 = detect_flips(detector, image, args.size, device)
            gtta, _, _, _, _ = _instances_from_raw(raw4, scores4, 0.50)
            out["detector_tta"][stem] = masks_to_rles(gtta)
            del raw4
            if (i + 1) % 25 == 0:
                print(f"[infer] {i + 1}/{len(recs)} t={time.time() - t0:.0f}s", flush=True)
        return out

    val_pairs = [(r.stem, r.image) for r in val]
    val_cand = infer_candidates(val_pairs, with_refiner=True)
    val_pq = {k: pq_stream(val, v) for k, v in val_cand.items()}
    del val_pairs
    for k, v in val_pq.items():
        print(f"[eval] {k} val PQ={v['pq']:.4f} {v}", flush=True)
    best = max(val_pq, key=lambda k: val_pq[k]["pq"])
    print(f"[eval] BEST val candidate={best} PQ={val_pq[best]['pq']:.4f}", flush=True)

    # test
    test_dir = root / "test" / "test_images"
    test_pairs = []
    for stem in test_files:
        img = cv2.imread(str(test_dir / f"{stem}.jpeg"), cv2.IMREAD_GRAYSCALE)
        if img is None or img.shape[:2] != (IMG, IMG):
            raise RuntimeError(f"bad test image {stem}")
        test_pairs.append((stem, img))
    test_cand = infer_candidates(test_pairs, with_refiner=True)
    for name in ("refiner", "refiner_notta", "detector_070", "detector_050", "detector_tta"):
        preds = test_cand[name]
        df = rles_to_rows(preds)
        df.to_csv(WORK / f"submission_{name}.csv", index=False)
        print(f"[out] submission_{name}.csv rows={len(df)}", flush=True)
        del test_cand[name]
    shutil.copy(WORK / f"submission_{best}.csv", WORK / "submission.csv")

    (WORK / "run_report.json").write_text(json.dumps({
        "args": vars(args), "best": best, "val_pq": val_pq,
        "seconds": round(time.time() - t0), "torch": torch.__version__,
    }, indent=1, default=str))
    val_stems = sorted({r.stem for r in val})
    (WORK / "val_stems.json").write_text(json.dumps(val_stems))
    for name in ("refiner", "refiner_notta", "detector_070", "detector_050", "detector_tta"):
        try:
            parts = {}
            for i, s in enumerate(val_stems):
                rles = val_cand[name].get(s, [])
                parts[f"pred_{i:03d}"] = np.array("\n".join(rles))
            np.savez_compressed(WORK / f"val_predictions_{name}.npz", stems=np.array(val_stems), **parts)
            print(f"[out] val_predictions_{name}.npz", flush=True)
        except Exception as e:
            print(f"[warn] val_predictions_{name}.npz failed: {e}", flush=True)
        del val_cand[name]
    print("[done]", json.dumps({k: v["pq"] for k, v in val_pq.items()}), flush=True)


if __name__ == "__main__":
    main()
