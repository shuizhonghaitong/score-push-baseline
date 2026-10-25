"""Gate a campaign candidate through the official metric, optionally update the store.

Usage: python verify_candidate.py logs/seam_results/2_799.json [--commit] [--tag SEAM2]
"""
import json, sys, importlib
sys.path.insert(0, "scripts")
m = importlib.import_module("metric_official")

src = sys.argv[1]
commit = "--commit" in sys.argv
tag = "SEAM"
if "--tag" in sys.argv:
    tag = sys.argv[sys.argv.index("--tag") + 1]

r = json.load(open(src))
n = r["n"]
pts = r["pts"]
assert len(pts) == n

text_lines = ["id,n,square_id,x,y,angle"]
for i, (x, y, a) in enumerate(pts):
    text_lines.append(f"{i},{n},{i},{x:.17g},{y:.17g},{a:.17g}")
text = "\n".join(text_lines) + "\n"

puzzles = m._parse_submission_csv(text)
rows = puzzles[n]
err = m._validate_puzzle(n, rows)
if err:
    print(f"REJECT {n}: validate: {err}")
    sys.exit(1)
squares, Lmp = m._build_squares(n, rows)
ok, msg = m.verify_puzzle(squares)
if not ok:
    print(f"REJECT {n}: verify: {msg}")
    sys.exit(1)
new_score = m.score_puzzle(n, float(Lmp))

store = json.load(open("models/packings.json"))
old = store[str(n)]
old_score = old["L"] / n ** 0.5 - 1
new_L = float(Lmp)
gain = old_score - new_score
print(f"n={n}: oldL={old['L']!r} newL={new_L!r} old_score={old_score:.10f} new_score={new_score:.10f} gain={gain:.3e} src={old.get('src')}")

if gain > 1e-12 and commit:
    old_pts = old["pts"]
    old_entry = dict(old)
    old_entry["L"] = new_L
    old_entry["pts"] = pts
    old_entry["strs"] = [[format(x, '.17g'), format(y, '.17g'), format(a, '.17g')] for x, y, a in pts]
    old_entry["src"] = f"{old.get('src')}+{tag}"
    old_entry["note"] = "candidate from " + src + " passed official verify_puzzle"
    store[str(n)] = old_entry
    json.dump(store, open("models/packings.json", "w"))
    total = sum(e["L"] / int(k) ** 0.5 - 1 for k, e in store.items())
    print(f"COMMITTED n={n}. store total now: {total:.12f}")
elif gain <= 1e-12:
    print("no gain; not committed")
