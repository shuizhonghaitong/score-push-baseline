"""Cross-run ensemble tuner (val-PQ ranked, applied to test CSVs).

Combines per-candidate instances from multiple completed runs that share the
same grouped 80/20 val split (same seed -> same val stems), e.g. the Kaggle
GPU v1 run (1024px unet/mrcnn/merge) and the local 768px U-Net run.

Inputs (each run dir):
  - val_stems.json
  - val_predictions_<model>.npz      (per-stem newline-joined b64 RLE)
  - submission_<model>.csv           (TEST)
  - val_gt.npz                       (only needed in the first/run dir)

Rules swept on val: every single model, union of all models, union of the
two best singles, union_nosmall of all models.  The best rule is applied to
the per-model TEST CSVs to produce the ensemble submission.

Usage:
  python src/tune_ensemble.py <ref_run_dir> <run2_dir> [run3_dir ...] [out.csv]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tune_postprocess import _decode, _encode, load_npz_rles, load_gt, merge_masks, pq_obs  # noqa: E402


def load_models(run_dir: Path) -> dict[str, dict[str, list[str]]]:
    out = {}
    for p in sorted(run_dir.glob("val_predictions_*.npz")):
        model = p.name[len("val_predictions_"):-len(".npz")]
        _, rles = load_npz_rles(p)
        out[model] = rles
    return out


def rule_masks(pixel: dict[str, list[np.ndarray]], rule: str,
               singles_order: list[str]) -> list[np.ndarray]:
    if rule.startswith("single:"):
        return pixel[rule.split(":", 1)[1]]
    if rule == "union_all":
        return merge_masks([m for ms in pixel.values() for m in ms])
    if rule == "union_all_nosmall":
        return [m for m in merge_masks([m for ms in pixel.values() for m in ms]) if m.sum() >= 64]
    if rule == "union_top2":
        a, b = singles_order[0], singles_order[1]
        return merge_masks(pixel[a] + pixel[b])
    raise ValueError(rule)


def main() -> None:
    argv = [a for a in sys.argv[1:] if not a.startswith("--")]
    assert len(argv) >= 2, "need at least two run dirs"
    out_csv = Path(argv[-1]) if len(argv) >= 3 else Path(argv[0]) / "tuned_ensemble.csv"
    run_dirs = [Path(a) for a in argv[: len(argv) - 1] if len(argv) >= 3] or [Path(a) for a in argv]

    models: dict[str, dict[str, list[str]]] = {}
    test_csvs: dict[str, pd.DataFrame] = {}
    stems = None
    for rd in run_dirs:
        st = json.loads((rd / "val_stems.json").read_text())
        if stems is None:
            stems = st
        else:
            assert st == stems, f"val stems differ between {rd} and first run"
        for model, rles in load_models(rd).items():
            key = f"{rd.name}/{model}"
            models[key] = rles
            csv = rd / f"submission_{model}.csv"
            if csv.exists():
                test_csvs[key] = pd.read_csv(csv)
    print(f"runs={len(run_dirs)} val stems={len(stems)} models={list(models)}")

    _, gt = load_gt(run_dirs[0] / "val_gt.npz")

    # single-model val PQ (for ordering)
    single_pq = {}
    pixel_cache: dict[str, dict[str, list[np.ndarray]]] = {}
    for key, rles in models.items():
        pixel = {}
        tot = {"tp_sum": 0.0, "tp": 0, "fp": 0, "fn": 0}
        for s in stems:
            ms = [_decode(r) for r in rles.get(s, [])]
            pixel[s] = ms
            r = pq_obs(gt[s], ms)
            for k in tot:
                tot[k] += r[k]
        den = tot["tp"] + 0.5 * tot["fp"] + 0.5 * tot["fn"]
        single_pq[key] = (tot["tp_sum"] / den) if den else 0.0
        pixel_cache[key] = pixel
    order = sorted(single_pq, key=lambda k: -single_pq[k])
    for k in order:
        print(f"  single {k:35s} PQ={single_pq[k]:.4f}")

    rules = [f"single:{k}" for k in order] + ["union_all", "union_all_nosmall"]
    if len(order) >= 2:
        rules.append("union_top2")
    results = {}
    for rule in rules:
        tot = {"tp_sum": 0.0, "tp": 0, "fp": 0, "fn": 0}
        for s in stems:
            pixel = {k: pixel_cache[k].get(s, []) for k in models}
            out = rule_masks(pixel, rule, order)
            r = pq_obs(gt[s], out)
            for k in tot:
                tot[k] += r[k]
        den = tot["tp"] + 0.5 * tot["fp"] + 0.5 * tot["fn"]
        results[rule] = (tot["tp_sum"] / den) if den else 0.0
    for rule in sorted(results, key=lambda k: -results[k]):
        print(f"  rule {rule:35s} PQ={results[rule]:.4f}")
    best = max(results, key=lambda k: results[k])
    print(f"best rule: {best} PQ={results[best]:.4f}")

    # apply to test (cover every official test stem; empty ones get one
    # empty-mask PPPP4 row so the CSV passes the missing=0 validation gate)
    full_stems = set(json.load(open(Path(__file__).resolve().parents[1] / "cache/test_stems_full.json")))
    test_stems = sorted(full_stems | {str(r.filament_id).rsplit("_", 1)[0] for k, df in test_csvs.items()
                                      for r in df.itertuples()})
    rows = []
    for s in test_stems:
        pixel = {}
        for k, df in test_csvs.items():
            sub = df[df["filament_id"].astype(str).str.startswith(s + "_")]
            pixel[k] = [_decode(str(r.segmentation_rle)) for r in sub.itertuples()]
        out = rule_masks(pixel, best, order)
        if not out:
            rows.append({"filament_id": f"{s}_1", "segmentation_rle": "PPPP4"})
        for j, m in enumerate(out, start=1):
            rows.append({"filament_id": f"{s}_{j}", "segmentation_rle": _encode(m)})
    pd.DataFrame(rows, columns=["filament_id", "segmentation_rle"]).to_csv(out_csv, index=False)
    print(f"wrote {out_csv} rows={len(rows)} test_stems={len(test_stems)}")


if __name__ == "__main__":
    main()
