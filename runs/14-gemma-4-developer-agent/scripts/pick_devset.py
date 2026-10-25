#!/usr/bin/env python3
"""Deterministic stratified dev subset of the public tasks (by instance_id hash)."""
import argparse
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=44)
    ap.add_argument('--gold', default=None, help='optional gold-check progress dir; keep only resolved gold tasks')
    ap.add_argument('--out', default=None)
    args = ap.parse_args()
    tasks = [l.strip() for l in open('data/tasks.jsonl') if l.strip()]
    import json
    rows = [json.loads(l) for l in open('data/tasks.jsonl')]
    gold = None
    if args.gold:
        gold = {json.loads(l)['id']: json.loads(l).get('resolved', False)
                for l in open(Path(args.gold) / 'progress.jsonl')}
        rows = [r for r in rows if gold.get(r['instance_id'])]
    by_repo = {}
    for r in rows:
        by_repo.setdefault(r['repo'], []).append(r['instance_id'])
    for repo, ids in by_repo.items():
        ids.sort(key=lambda i: hashlib.sha256(i.encode()).hexdigest())
    total = sum(len(v) for v in by_repo.values())
    print(f"candidates: {total}")
    picks = []
    for repo in sorted(by_repo):
        share = max(1, round(len(by_repo[repo]) / total * args.n))
        picks += by_repo[repo][:share]
    # fill or trim deterministically
    allids = [i for ids in (by_repo[r] for r in sorted(by_repo)) for i in ids]
    for i in allids:
        if len(picks) >= args.n:
            break
        if i not in picks:
            picks.append(i)
    picks = picks[:args.n]
    picks.sort()
    if args.out:
        Path(args.out).write_text(','.join(picks))
    print(f"selected {len(picks)} tasks:")
    print(','.join(picks))


if __name__ == '__main__':
    main()
