#!/usr/bin/env python3
"""Fetch the newest compatible py3.12 x86_64 wheel per package from tsinghua."""
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
try:
    from packaging.utils import parse_version
except ImportError:
    from packaging.version import Version as parse_version
from packaging.requirements import Requirement

RUN = Path(__file__).resolve().parents[1]
CACHE = RUN / 'models' / 'pypi_cache'
CACHE.mkdir(parents=True, exist_ok=True)
BASE = 'https://pypi.tuna.tsinghua.edu.cn'
W = int(sys.argv[2]) if len(sys.argv) > 2 else 4


def parse_name_ver(fn: str):
    m = re.match(r'^(.+?)-(\d[^-]*)-(.+)\.whl$', fn)
    return m.group(1), parse_version(m.group(2)) if m else None


def pick(pkg: str):
    html = subprocess.run(['curl', '-s', '--max-time', '30', f'{BASE}/simple/{pkg}/'],
                          capture_output=True, text=True, check=True).stdout

    def tag_rank(fn):
        if re.search(r'cp312-cp312-manylinux.*x86_64', fn):
            return 0
        if re.search(r'abi3-manylinux.*x86_64', fn):
            return 1
        if re.search(r'(?:py3|py2\.py3)-none-(?:any|manylinux[0-9_.]*x86_64)', fn):
            return 2
        return None

    cands = []
    for href in re.findall(r'href="([^"]+)"', html):
        href = href.split('#')[0]
        fn = href.rsplit('/', 1)[-1]
        rank = tag_rank(fn)
        if rank is None:
            continue
        ver = parse_name_ver(fn)[1]
        if ver is None or ver.is_prerelease:
            continue
        full = href if href.startswith('http') else (BASE + href if href.startswith('/')
                                                     else BASE + '/packages/' + href)
        cands.append((fn, full, rank))
    if not cands:
        return None, 'NOURL'
    cands.sort(key=lambda c: (c[2], parse_name_ver(c[0])[1]))
    # lowest rank (binary first); within same rank, newest version
    best_rank = cands[0][2]
    same = [c for c in cands if c[2] == best_rank]
    same.sort(key=lambda c: parse_name_ver(c[0])[1])
    fn, full, _ = same[-1]
    head = subprocess.run(['curl', '-sI', full], capture_output=True, text=True).stdout
    m = re.search(r'content-length:\s*(\d+)', head, re.I)
    return (fn, full, int(m.group(1)) if m else 0), None


def dl_one(item):
    fn, full, size = item
    dest = CACHE / fn
    if dest.exists() and dest.stat().st_size == size:
        return f'skip {fn}'
    if size > 20_000_000:
        n = W
        chunk = -(-size // n)
        parts = []
        with ThreadPoolExecutor(max_workers=n) as ex:
            futs = []
            for i in range(n):
                s = i * chunk
                e = min(s + chunk, size) - 1
                if s >= size:
                    break
                p = dest.with_suffix(f'.part{i}')
                parts.append(p)
                futs.append(ex.submit(subprocess.run,
                                      ['curl', '-s', '--max-time', '1200', '-r', f'{s}-{e}', '-o', str(p), full],
                                      check=True))
            for f in futs:
                f.result()
        with open(dest, 'wb') as out:
            for p in parts:
                out.write(p.read_bytes())
                p.unlink()
    else:
        subprocess.run(['curl', '-s', '--max-time', '600', '-o', str(dest), full], check=True)
    return f'done {fn} {size}'


def main():
    pkgs = sys.argv[1].split()
    items, errs = [], []
    for pkg in pkgs:
        item, err = pick(pkg)
        if err:
            errs.append(f'{err} {pkg}')
        else:
            items.append(item)
    with ThreadPoolExecutor(max_workers=4) as ex:
        for line in ex.map(dl_one, items):
            print(line, flush=True)
    for e in errs:
        print(e, flush=True)
    print('ALL_FETCH_DONE', flush=True)


if __name__ == '__main__':
    main()
