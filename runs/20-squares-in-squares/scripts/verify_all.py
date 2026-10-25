import json, sys, time
sys.path.insert(0, 'scripts')
import importlib
m = importlib.import_module('metric_official')

def sqs(pts):
    return [m.Square(x, y, a, x_str=f"{x:.17g}", y_str=f"{y:.17g}", angle_str=f"{a:.17g}") for x, y, a in pts]

store = json.load(open('models/packings.json'))
which = sys.argv[1] if len(sys.argv) > 1 else 'nongrid'
fails = []
t0 = time.time()
for n in range(1, 1001):
    v = store[str(n)]
    is_ng = v['src'] != 'grid'
    if which == 'nongrid' and not is_ng:
        continue
    if which == 'grid' and is_ng:
        continue
    t = time.time()
    ok, msg = m.verify_puzzle(sqs(v['pts']))
    if not ok:
        fails.append((n, v['src'], msg))
        print(f"FAIL n={n} {v['src']}: {msg}", flush=True)
print(f"done in {time.time()-t0:.1f}s; fails: {len(fails)}")
json.dump(fails, open(f"logs/verify_fails_{which}.json", "w"))
