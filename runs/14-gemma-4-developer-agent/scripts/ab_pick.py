#!/usr/bin/env python3
"""Pick the local A/B winner for submit_slot2. Waits up to 45 min for both runs
to complete (16 dev tasks each); falls back to v2 (mechanism-backed default)."""
import json, os, sys, time
RUN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
p1 = os.path.join(RUN, 'results', 'ab_v1', 'progress.jsonl')
p2 = os.path.join(RUN, 'results', 'ab_v2', 'progress.jsonl')
COMPLETE = 16

def n(p):
    if not os.path.exists(p):
        return -1
    with open(p) as f:
        return sum(1 for l in f if l.strip())

def score(p):
    with open(p) as f:
        return sum(1 for l in f if l.strip() and json.loads(l).get('resolved'))

a1, a2 = n(p1), n(p2)
flip = a1 < COMPLETE and a2 >= COMPLETE
if flip:
    p1, p2 = p2, p1
    a1, a2 = a2, a1
t0 = time.time()
while a1 < COMPLETE and a2 >= 0 and time.time() - t0 < 45 * 60:
    time.sleep(300)
    a1, a2 = n(p1), n(p2)
if a1 >= COMPLETE and a2 >= COMPLETE:
    s1, s2 = score(p1), score(p2)
    winner = 'v2' if s2 > s1 else ('v1' if s1 > s2 else 'v2')
    if flip:
        winner = {'v1': 'v2', 'v2': 'v1'}[winner]
    print(f'ab {s1} vs {s2}' , file=sys.stderr)
    print(winner)
else:
    print(f'ab incomplete n1={a1} n2={a2}; default v2', file=sys.stderr)
    print('v2')
