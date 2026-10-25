#!/usr/bin/env python3
"""Slot-3 picker: best dev-set variant among v1/v2/t04.
Requires each run complete (16 dev tasks); waits up to 45 min for laggards.
Fallback (incomplete): t04 (temperature evidence; v2 already used slot 2)."""
import json, os, sys, time
RUN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COMPLETE = 16
VARIANTS = [('v1', 'ab_v1'), ('v2', 'ab_v2'), ('t04', 'ab_t04')]

def n(d):
    p = os.path.join(RUN, 'results', d, 'progress.jsonl')
    if not os.path.exists(p):
        return -1
    with open(p) as f:
        return sum(1 for l in f if l.strip())

def score(d):
    p = os.path.join(RUN, 'results', d, 'progress.jsonl')
    with open(p) as f:
        return sum(1 for l in f if l.strip() and json.loads(l).get('resolved'))

counts = {d: n(d) for _, d in VARIANTS}
have_any = any(c >= 0 for c in counts.values())
t0 = time.time()
while have_any and min(counts.values()) < COMPLETE and time.time() - t0 < 45 * 60:
    time.sleep(300)
    counts = {d: n(d) for _, d in VARIANTS}
if have_any and min(counts.values()) >= COMPLETE:
    scores = {d: score(d) for _, d in VARIANTS}
    best = max(VARIANTS, key=lambda vd: scores[vd[1]])
    print(f"devset { {vd[0]: scores[vd[1]] for vd in VARIANTS} }", file=sys.stderr)
    print(best[0])
else:
    print(f"ab incomplete {counts}; default t04", file=sys.stderr)
    print('t04')
