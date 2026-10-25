#!/bin/bash
# Local experiment driver. Usage: ./code/run_local.sh [tag] [KEY=VAL ...]
# Builds features once per (FEAT_V2, data) into local_cache/, then runs wf on top.
set -e
cd "$(dirname "$0")/.."
PY=./venv/bin/python
TAG=${1:-run}; shift || true
mkdir -p local_cache
FEAT_V2=${FEAT_V2:-off}
CACHE=local_cache/feats_${FEAT_V2}_$(date +%Y%m%d).parquet
EXTRA=("$@")
if [ ! -f "$CACHE" ]; then
  $PY code/pipeline.py < /dev/null >/dev/null 2>&1 || true
  MODE=diag OUT_DIR=local_cache FEAT_V2=$FEAT_V2 FEAT_SAVE="$CACHE" NJOBS=${NJOBS:-6} $PY code/pipeline.py
fi
ENVSTR=""
for kv in "${EXTRA[@]}"; do ENVSTR="$ENVSTR $kv"; done
env MODE=wf OUT_DIR=local_cache/wf_$TAG FEAT_V2=$FEAT_V2 RESUME_FEATS="$CACHE" NJOBS=${NJOBS:-6} $ENVSTR $PY code/pipeline.py
tail -2 local_cache/wf_$TAG/wf_results.csv 2>/dev/null || true
