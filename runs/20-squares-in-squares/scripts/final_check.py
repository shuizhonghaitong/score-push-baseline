"""End-to-end official metric check on the exact submission CSV."""
import io
import json
import math
import sys
import time

sys.path.insert(0, "scripts")
import importlib
m = importlib.import_module("metric_official")

import sys
path = sys.argv[1] if len(sys.argv) > 1 else "outputs/submission_run20a.csv"
text = open(path).read()
t0 = time.time()
puzzles = m._parse_submission_csv(text)
print(f"parsed {len(puzzles)} puzzles in {time.time()-t0:.1f}s")

total = 0.0
fails = []
t0 = time.time()
for n in range(1, 1001):
    if n not in puzzles:
        total += m.infeasible_score(n)
        fails.append((n, "absent"))
        continue
    err = m._validate_puzzle(n, puzzles[n])
    if err:
        total += m.infeasible_score(n)
        fails.append((n, err))
        continue
    squares, Lmp = m._build_squares(n, puzzles[n])
    ok, msg = m.verify_puzzle(squares)
    if not ok:
        total += m.infeasible_score(n)
        fails.append((n, msg))
        continue
    total += m.score_puzzle(n, float(Lmp))
print(f"verified all in {time.time()-t0:.1f}s")
print(f"TOTAL SCORE: {total:.10f}")
print(f"fails: {len(fails)}")
for f in fails[:10]:
    print("  ", f)
json.dump({"total": total, "fails": fails}, open("logs/" + "final_check_" + path.split("submission_")[-1].replace(".csv","") + ".json", "w"))
