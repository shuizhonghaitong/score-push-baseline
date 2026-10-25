"""Prepare a local (no-torch, no-t1) variant of the v4d pipeline for fast onset/queue iteration.

Usage (after competition data + anubisyy assets are downloaded):
  TFB_DATA_ROOT=<data dir with submission_key.csv> \
  TFB_ASSET_ROOT=<anubisyy assets dir with manifest.json> \
  TFB_WORK_ROOT=<scratch dir> \
  python local/prepare_local_run.py   # creates local/nb_local/ ready to run
Then:
  TFB_DATA_ROOT=... TFB_ASSET_ROOT=... TFB_WORK_ROOT=... TFB_THREADS=12 \
  .venv/bin/python local/nb_local/run_local.py
"""
import ast
import os
import shutil
from pathlib import Path

RUN = Path(__file__).resolve().parents[1]
SRC = RUN / 'nb_v4d2' / 'run_v4d2.py'
DST_DIR = RUN / 'local' / 'nb_local'
DST = DST_DIR / 'run_local.py'

src = SRC.read_text()

P = [
    ("WORK = Path('/kaggle/working')",
     "WORK = Path(os.environ.get('TFB_WORK_ROOT', '/kaggle/working'))"),
    ("THREADS = min(4, os.cpu_count() or 1)",
     "THREADS = int(os.environ.get('TFB_THREADS', 4))"),
    ("os.environ.update(TFB_DATA_ROOT=str(DATA), TFB_OUTPUT_ROOT=str(OUTPUT), TFB_THREADS=str(min(4, os.cpu_count() or 1)))",
     "os.environ.update(TFB_DATA_ROOT=str(DATA), TFB_OUTPUT_ROOT=str(OUTPUT), TFB_THREADS=str(THREADS))"),
    ("USE_T1_EXTRA = True", "USE_T1_EXTRA = False"),
]
for old, new in P:
    n = src.count(old)
    assert n == 1, (old[:60], n)
    src = src.replace(old, new)

# local run: keep task1 base state + queue + task4, skip t1extra (needs torch)
ast.parse(src)
DST_DIR.mkdir(parents=True, exist_ok=True)
DST.write_text(src)
shutil.copy(SRC.parent / 'kernel-metadata.json', DST_DIR / 'kernel-metadata.json')
print('wrote', DST)
print('run with TFB_DATA_ROOT/TFB_ASSET_ROOT/TFB_WORK_ROOT/TFB_THREADS set')
