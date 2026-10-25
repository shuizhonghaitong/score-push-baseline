#!/usr/bin/env python3
"""Compare two eval result dirs (progress.jsonl) per task."""
import argparse
import json
from pathlib import Path


def load(d):
    rows = {}
    p = Path(d) / 'progress.jsonl'
    if p.exists():
        for line in p.read_text().splitlines():
            try:
                r = json.loads(line)
                rows[r['id']] = r
            except (ValueError, KeyError):
                pass
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('a')
    ap.add_argument('b')
    ap.add_argument('--label-a', default='A')
    ap.add_argument('--label-b', default='B')
    args = ap.parse_args()
    ra, rb = load(args.a), load(args.b)
    common = sorted(set(ra) & set(rb))
    wa = sum(1 for t in common if ra[t]['resolved'] and not rb[t]['resolved'])
    wb = sum(1 for t in common if rb[t]['resolved'] and not ra[t]['resolved'])
    both = sum(1 for t in common if ra[t]['resolved'] and rb[t]['resolved'])
    sa = sum(1 for t in common if ra[t]['resolved'])
    sb = sum(1 for t in common if rb[t]['resolved'])
    ta = sum(ra[t].get('seconds', 0) for t in common)
    tb = sum(rb[t].get('seconds', 0) for t in common)
    print(f"{args.label_a}: {sa}/{len(common)} resolved ({sa/len(common):.1%})  total {ta/60:.0f} min")
    print(f"{args.label_b}: {sb}/{len(common)} resolved ({sb/len(common):.1%})  total {tb/60:.0f} min")
    print(f"only-{args.label_a}: {wa}  only-{args.label_b}: {wb}  both: {both}")
    print()
    for t in common:
        if ra[t]['resolved'] != rb[t]['resolved']:
            w = args.label_a if ra[t]['resolved'] else args.label_b
            print(f"  {w} wins: {t}")
    only_a = sorted(set(ra) - set(rb))
    only_b = sorted(set(rb) - set(ra))
    if only_a:
        print(f"missing in B: {only_a}")
    if only_b:
        print(f"missing in A: {only_b}")


if __name__ == '__main__':
    main()
