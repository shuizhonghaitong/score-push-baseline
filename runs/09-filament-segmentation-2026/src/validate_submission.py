"""Validate a competition submission CSV before uploading.

Checks: header, 180 unique test stems covered, RLE base64 decode,
mask dimensions, mask count per image, total rows.
Usage: python src/validate_submission.py submission.csv [test_stems.json]
"""
from __future__ import annotations
import base64
import json
import sys
from pathlib import Path

import pandas as pd

RUN = Path(__file__).resolve().parents[1]
IMG = 2048


def decode_rle_b64(rle: str) -> tuple[int, int]:
    from pycocotools import mask as mu
    try:
        m = mu.decode({"size": [IMG, IMG], "counts": rle.encode("latin-1")})
    except Exception:
        raw = base64.b64decode(rle)  # legacy base64 RLE
        m = mu.decode({"size": [IMG, IMG], "counts": raw})
    return m.shape


def main() -> None:
    csv_path = Path(sys.argv[1])
    stems = json.load(open(sys.argv[2])) if len(sys.argv) > 2 else json.load(open(RUN / "cache/test_stems_full.json"))
    df = pd.read_csv(csv_path)
    print("columns:", list(df.columns), "rows:", len(df))
    assert list(df.columns) == ["filament_id", "segmentation_rle"], f"bad columns {list(df.columns)}"
    stems_in = {str(r.filament_id).rsplit("_", 1)[0] for r in df.itertuples()}
    missing = set(stems) - stems_in
    extra = stems_in - set(stems)
    print(f"stems covered: {len(stems_in & set(stems))}/{len(stems)}; missing={len(missing)} extra={len(extra)}")
    if missing:
        print("missing:", sorted(missing)[:10])
    if extra:
        print("extra:", sorted(extra)[:10])
    per = df["filament_id"].map(lambda s: str(s).rsplit("_", 1)[0]).value_counts()
    print("masks per image: min", per.min(), "median", int(per.median()), "max", per.max(),
          "images_with_0:", len(stems) - len(per))
    # quality gate: refuse clearly-broken submissions (val PQ floor)
    try:
        import json as _json
        rep = _json.loads((csv_path.parent / "run_report.json").read_text())
        best_pq = float(rep["val_pq"][rep["best"]]["pq"])
        print(f"run_report best={rep['best']} val_pq={best_pq:.4f}")
        if best_pq < 0.02:
            raise SystemExit("quality gate: val PQ below 0.02 floor; refusing submission")
    except FileNotFoundError:
        print("no run_report.json next to CSV; skipping quality gate")
    # decode a sample of RLEs
    import random
    sample = random.Random(0).sample(range(len(df)), min(20, len(df)))
    ok = 0
    for i in sample:
        shape = decode_rle_b64(str(df.iloc[i].segmentation_rle))
        assert shape == (IMG, IMG), f"row {i}: shape {shape}"
        ok += 1
    print(f"RLE decode OK on {ok} sampled rows")


if __name__ == "__main__":
    main()
