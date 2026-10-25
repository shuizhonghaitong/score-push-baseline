"""Parse skarin's certified submission, measure per-puzzle L with official
metric, and adopt rows where his L beats our store."""
import csv
import json
import math
import sys

sys.path.insert(0, "scripts")
import importlib
m = importlib.import_module("metric_official")

rows = {}
with open("logs/skarin/extracted/submission_v3_22.479535.csv") as f:
    for r in csv.DictReader(f):
        n = int(r["n"])
        rows.setdefault(n, []).append((int(r["square_id"]), r["x"], r["y"], r["angle"]))

his_L = {}
for n, rs in rows.items():
    rs.sort(key=lambda t: t[0])
    sqs = [m.Square(float(x), float(y), float(a), x_str=x, y_str=y, angle_str=a) for _, x, y, a in rs]
    his_L[n] = float(m.measure_side_mp(sqs))

store = json.load(open("models/packings.json"))
adopted, kept, worse = 0, 0, []
for n, L in his_L.items():
    mine = store[str(n)]["L"]
    if L < mine - 1e-9:
        store[str(n)] = {"pts": [[float(x), float(y), float(a)] for _, x, y, a in
                                 sorted(rows[n], key=lambda t: t[0])],
                         "strs": [list(t[1:]) for t in sorted(rows[n], key=lambda t: t[0])],
                         "s": None, "src": "skarin", "feas": True, "L": L, "margin": None}
        adopted += 1
    elif L > mine + 1e-9:
        worse.append((n, mine, L))
    else:
        kept += 1
json.dump(store, open("models/packings.json", "w"))
tot = sum(store[str(n)]["L"] / math.sqrt(n) - 1 for n in range(1, 1001))
print(f"skarin puzzles: {len(his_L)}; adopted {adopted}, tie {kept}, ours better: {len(worse)}")
print("sample where ours better:", worse[:10])
print(f"total: {tot:.10f}")
