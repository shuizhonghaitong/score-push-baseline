#!/bin/bash
# Local submit-mode run: MODE=submit over resumed features + local labels.
# Run from a dir containing ./data -> DATA_DIR. Usage: code/submit_local.sh DATA_DIR FEATS OUT_DIR [KEY=VAL ...]
set -u
cd "$(dirname "$0")/.."
PY=./venv/bin/python
DATA_DIR=$1; FEATS=$2; OUT=$3; shift 3
mkdir -p "$OUT"
env MODE=submit OUT_DIR=$OUT DATA_DIR=$DATA_DIR RESUME_FEATS=$FEATS NJOBS=${NJOBS:-8} "$@" $PY code/pipeline.py
ls -la "$OUT"/submission.csv 2>/dev/null && head -3 "$OUT"/submission.csv
