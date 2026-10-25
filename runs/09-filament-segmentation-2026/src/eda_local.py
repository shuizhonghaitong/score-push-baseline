"""Local EDA on the downloaded esairlab MAGFiLO 1.0 dataset.

Checks:
- annotation JSON structure (COCO), image/annotator coverage
- LEAK CHECK: which of the 180 competition test stems appear in this dataset
- per-image filament count, area and bbox-size distributions (2048 scale)
- date/site distribution from image-name conventions
Writes report/eda_stats.json and prints a summary.
"""
from __future__ import annotations
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

RUN = Path(__file__).resolve().parents[1]
ANN = RUN / "data/extracted/MAGFiLO_1.0_Annotations.json"
TEST_STEMS = json.load(open(RUN / "cache/test_stems_full.json"))


def main() -> None:
    data = json.load(open(ANN))
    images = data.get("images", [])
    anns = data.get("annotations", [])
    cats = data.get("categories", [])
    print("images:", len(images), "annotations:", len(anns), "categories:", cats[:3])
    if images:
        print("image keys:", sorted(images[0].keys()))
        print("image sample:", images[0])
    if anns:
        print("ann keys:", sorted(anns[0].keys()))
        print("ann sample:", anns[0])

    stems = [img.get("file_name", "") for img in images]
    stems = [s.split("/")[-1] for s in stems]
    stem_set = set(stems)
    test_stem_set = set(TEST_STEMS)
    leak = sorted(test_stem_set & stem_set)
    print(f"LEAK CHECK: {len(leak)}/{len(TEST_STEMS)} test stems present in dataset")
    if leak:
        print("leaked:", leak[:10])

    # annotator rows per physical observation
    per_img = Counter(a.get("image_id") for a in anns)
    id2stem = {img["id"]: (img.get("file_name", "") or "").split("/")[-1] for img in images}
    stems_of_entries = [id2stem[i] for i in per_img]
    rows_per_stem = Counter()
    for i in {img["id"]: img["id"] for img in images}:
        rows_per_stem[id2stem[i]] += 1
    print("annotator-rows-per-observation histogram:", dict(sorted(Counter(rows_per_stem.values()).items())))
    n_blank_entries = len(images) - len(per_img)
    print(f"entries with 0 annotations: {n_blank_entries}/{len(images)}")
    n_fil = Counter(per_img.values())
    print("filaments-per-annotator-record histogram (n_fil: records):", dict(sorted(n_fil.items())))

    areas, bw, bh = [], [], []
    for a in anns:
        if "area" in a and a["area"]:
            areas.append(a["area"])
        if a.get("bbox"):
            bw.append(a["bbox"][2]); bh.append(a["bbox"][3])
    if areas:
        areas = np.array(areas)
        print("area px percentiles p10/p50/p90:", np.percentile(areas, [10, 50, 90]).round(0),
              "max:", areas.max(), "zero-area:", int((areas == 0).sum()))
    if bw:
        bw = np.array(bw); bh = np.array(bh)
        print("bbox w p10/50/90:", np.percentile(bw, [10, 50, 90]).round(0))
        print("bbox h p10/50/90:", np.percentile(bh, [10, 50, 90]).round(0))

    # date distribution from names: YYYYMMDDHHMMSSxx
    years = Counter(s[:4] for s in stem_set if len(s) >= 8 and s[:4].isdigit())
    print("year histogram:", dict(sorted(years.items())))

    stats = {
        "n_images": len(images), "n_anns": len(anns),
        "n_test_leaked": len(leak), "leaked_test_stems": leak,
        "filaments_per_record_hist": dict(n_fil), "blank_records": n_blank_entries, "rows_per_obs_hist": dict(sorted(Counter(rows_per_stem.values()).items())),
        "area_p10_50_90": np.percentile(areas, [10, 50, 90]).tolist() if len(areas) else None,
        "bbox_w_p10_50_90": np.percentile(bw, [10, 50, 90]).tolist() if len(bw) else None,
        "bbox_h_p10_50_90": np.percentile(bh, [10, 50, 90]).tolist() if len(bh) else None,
        "year_hist": dict(years),
    }
    out = RUN / "report/eda_stats.json"
    out.write_text(json.dumps(stats, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
