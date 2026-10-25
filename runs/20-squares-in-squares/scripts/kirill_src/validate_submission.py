"""Evaluate the serialized CSV with the unmodified official v17 metric."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import argparse
import csv
import hashlib
import json
import math
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent / "reference_sources"))
import official_metric as metric


def validate(path, output, workers=3):
    start = time.perf_counter()
    puzzles = defaultdict(list)
    with Path(path).open(newline="") as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames == ["id", "n", "square_id", "x", "y", "angle"]
        for i, row in enumerate(reader):
            assert int(row["id"]) == i
            puzzles[int(row["n"])].append(row)
    results = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for result in pool.map(metric._evaluate_puzzle_worker, sorted(puzzles.items())):
            results.append(result)
            if len(results) % 100 == 0:
                print(f"Validated {len(results)}/{len(puzzles)} puzzles; {time.perf_counter()-start:.1f}s", flush=True)
    missing = [n for n in range(1, 1001) if n not in puzzles]
    rejected = [result for result in results if result[3] != "ok"]
    score = sum(result[1] for result in results) + sum(metric.infeasible_score(n) for n in missing)
    report = dict(file=str(path), sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                  metric_sha256=hashlib.sha256(Path(metric.__file__).read_bytes()).hexdigest(),
                  score=score, rows=sum(map(len, puzzles.values())), submitted_puzzles=len(puzzles),
                  missing_puzzles=len(missing), rejected=rejected,
                  elapsed_seconds=time.perf_counter()-start, results=results)
    Path(output).write_text(json.dumps(report, indent=2))
    print(json.dumps({k: v for k, v in report.items() if k != "results"}, indent=2), flush=True)
    assert not rejected, rejected[:3]
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    parser.add_argument("output")
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    validate(args.path, args.output, args.workers)
