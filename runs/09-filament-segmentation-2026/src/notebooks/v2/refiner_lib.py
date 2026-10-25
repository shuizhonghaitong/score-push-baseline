"""Crop refiner + topology-safe TTA + instance post-processing (v2 building blocks).

Design follows the publicly shared LB-0.70 handoff pipeline (inference-only
reconstruction of another team's assets); here every model is trained by us
on the official competition training split.

Components
----------
- ``CropUNet``: ResNet18-encoder U-Net that refines a 256x256 native
  resolution square crop around a detector mask into a probability map.
- 4-flip TTA with topology guards (area ratio, component count, Dice).
- Detector-seed splitting of crowded tasks (Voronoi by seed points).
- Positive-overlap merging and small-instance filtering.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np


# --------------------------------------------------------------------------- #
# Geometry helpers
# --------------------------------------------------------------------------- #
def square_bounds(mask: np.ndarray, context: float = 1.8, minimum: int = 96,
                  max_side: int | None = None) -> tuple[int, int, int, int]:
    """Native-resolution square (side x side) window around a 2D mask."""
    value = np.asarray(mask, dtype=bool)
    h, w = value.shape
    ys, xs = np.where(value)
    cx = (xs.min() + xs.max() + 1) / 2
    cy = (ys.min() + ys.max() + 1) / 2
    side = int(math.ceil(max(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1) * context))
    side = min(max(side, minimum), min(h, w, max_side or min(h, w)))
    x0 = int(round(cx - side / 2))
    y0 = int(round(cy - side / 2))
    x1, y1 = x0 + side, y0 + side
    if x0 < 0:
        x1 -= x0
        x0 = 0
    if y0 < 0:
        y1 -= y0
        y0 = 0
    if x1 > w:
        x0 -= x1 - w
        x1 = w
    if y1 > h:
        y0 -= y1 - h
        y1 = h
    return int(x0), int(y0), int(x1), int(y1)


def component_count8(mask: np.ndarray) -> int:
    return int(cv2.connectedComponents(np.asarray(mask, np.uint8), connectivity=8)[0] - 1)


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    inter = int(np.logical_and(a, b).sum())
    union = int(np.logical_or(a, b).sum())
    return inter / union if union else 0.0


def mask_dice(a: np.ndarray, b: np.ndarray) -> float:
    den = int(a.sum()) + int(b.sum())
    return 1.0 if den == 0 else 2.0 * int(np.logical_and(a, b).sum()) / den


# --------------------------------------------------------------------------- #
# Positive-overlap grouping / merging
# --------------------------------------------------------------------------- #
def _overlap_groups(indices: list[int], overlap_test) -> list[list[int]]:
    parent = list(range(len(indices)))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for i in range(len(indices)):
        for j in range(i + 1, len(indices)):
            if overlap_test(i, j):
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[ri] = rj
    groups: dict[int, list[int]] = {}
    for i in range(len(indices)):
        groups.setdefault(find(i), []).append(indices[i])
    return list(groups.values())


def positive_overlap_groups(masks: list[np.ndarray]) -> list[list[int]]:
    """Group instance indices sharing at least one pixel (bbox prefilter)."""
    if not masks:
        return []
    boxes = []
    for m in masks:
        ys, xs = np.where(m)
        if len(ys) == 0:
            boxes.append(None)
        else:
            boxes.append((ys.min(), ys.max(), xs.min(), xs.max()))

    def overlap_test(i: int, j: int) -> bool:
        a, b = boxes[i], boxes[j]
        if a is None or b is None:
            return False
        if a[0] > b[1] or b[0] > a[1] or a[2] > b[3] or b[2] > a[3]:
            return False
        return bool(np.logical_and(masks[i], masks[j]).any())

    return _overlap_groups(list(range(len(masks))), overlap_test)


def merge_mask_groups(masks: list[np.ndarray], groups: list[list[int]]) -> list[np.ndarray]:
    return [np.logical_or.reduce([masks[k] for k in g]).astype(bool) for g in groups]


# --------------------------------------------------------------------------- #
# Detector-seed splitting of crowded tasks
# --------------------------------------------------------------------------- #
def split_by_detector_seeds(
    mask: np.ndarray,
    member_masks: list[np.ndarray],
    member_scores: np.ndarray,
    max_seed_iou: float = 0.10,
    min_unique_px: int = 16,
    min_seed_distance: float = 8.0,
) -> list[np.ndarray]:
    """Voronoi split of a merged task mask using disjoint detector seeds."""
    mask = np.asarray(mask, dtype=bool)
    if not mask.any() or len(member_masks) < 2:
        return [mask]
    scores = np.asarray(member_scores, dtype=np.float32)
    accepted: list[int] = []
    for k in sorted(range(len(member_masks)), key=lambda i: (-scores[i], i)):
        if all(mask_iou(member_masks[k], member_masks[o]) <= max_seed_iou for o in accepted):
            accepted.append(k)
    if len(accepted) < 2:
        return [mask]
    seeds: list[tuple[int, int]] = []
    for k in accepted:
        others = np.zeros_like(mask)
        for o in accepted:
            if o != k:
                others |= member_masks[o]
        unique = member_masks[k] & mask & ~others
        ys, xs = np.nonzero(unique)
        if len(ys) < min_unique_px:
            continue
        cy, cx = float(ys.mean()), float(xs.mean())
        n = int(np.argmin((ys - cy) ** 2 + (xs - cx) ** 2))
        seeds.append((int(ys[n]), int(xs[n])))
    if len(seeds) < 2:
        return [mask]
    for a in range(len(seeds)):
        for b in range(a + 1, len(seeds)):
            if math.hypot(seeds[a][0] - seeds[b][0], seeds[a][1] - seeds[b][1]) < min_seed_distance:
                return [mask]
    fy, fx = np.nonzero(mask)
    d2 = np.stack([(fy - sy) ** 2 + (fx - sx) ** 2 for sy, sx in seeds], axis=1)
    assign = d2.argmin(axis=1)
    out = []
    for s in range(len(seeds)):
        part = np.zeros_like(mask)
        sel = assign == s
        if not sel.any():
            return [mask]
        part[fy[sel], fx[sel]] = True
        out.append(part)
    if not np.array_equal(np.logical_or.reduce(out), mask):
        return [mask]
    return out


# --------------------------------------------------------------------------- #
# TTA probability selection with topology guards
# --------------------------------------------------------------------------- #
@dataclass
class TtaConfig:
    threshold: float = 0.70
    area_min: float = 0.60
    area_max: float = 1.70
    dice_min: float = 0.62
    min_control_area: int = 32
    max_extra_components: int = 1


def apply_crop_transform(arr: np.ndarray, transform: str) -> np.ndarray:
    if transform == "identity":
        return arr
    if transform == "hflip":
        return arr[:, :, ::-1].copy()
    if transform == "vflip":
        return arr[:, ::-1, :].copy()
    if transform == "hvflip":
        return arr[:, ::-1, ::-1].copy()
    raise ValueError(transform)


def invert_probability_transform(prob: np.ndarray, transform: str) -> np.ndarray:
    return apply_crop_transform(prob, transform)  # flips are self-inverse


def select_topology_safe(single_prob: np.ndarray, tta_prob: np.ndarray, cfg: TtaConfig):
    """Return (probability_to_use, accepted: bool, reason: str)."""
    single_mask = single_prob >= cfg.threshold
    tta_mask = tta_prob >= cfg.threshold
    if not tta_mask.any():
        return single_prob, False, "empty_tta"
    if not single_mask.any():
        return single_prob, False, "empty_single"
    area_ratio = tta_mask.sum() / max(int(single_mask.sum()), 1)
    if area_ratio < cfg.area_min:
        return single_prob, False, "area_shrink"
    if area_ratio > cfg.area_max:
        return single_prob, False, "area_growth"
    n_single = component_count8(single_mask)
    n_tta = component_count8(tta_mask)
    if n_tta > max(n_single + cfg.max_extra_components, 2):
        return single_prob, False, "fragmentation"
    if single_mask.sum() >= cfg.min_control_area and mask_dice(single_mask, tta_mask) < cfg.dice_min:
        return single_prob, False, "overlap"
    return tta_prob, True, "accepted"
