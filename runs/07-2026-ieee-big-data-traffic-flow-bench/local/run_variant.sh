#!/bin/bash
# Run a local variant. Usage: bash local/run_variant.sh <va|vb|vc|vd|ve> [THREADS]
set -e
RUN=/mnt/disk3/liuyutong_home/workspace/skill/score-push/runs/07-2026-ieee-big-data-traffic-flow-bench
cd $RUN
VAR=$1; THREADS=${2:-16}
export TFB_DATA_ROOT=$RUN/data_local/kaggle_public
export TFB_ASSET_ROOT=$RUN/assets
export TFB_WORK_ROOT=$RUN/local/work_$VAR
export TFB_THREADS=$THREADS
export CUDA_VISIBLE_DEVICES=
export OMP_NUM_THREADS=$THREADS
mkdir -p $TFB_WORK_ROOT
test -f $TFB_DATA_ROOT/submission_key.csv || { echo "data missing"; exit 3; }
echo "=== variant $VAR threads=$THREADS start $(date '+%F %T') ==="
.venv/bin/python local/nb_$VAR/run_$VAR.py > logs/local_$VAR.log 2>&1
rc=$?
echo "=== variant $VAR rc=$rc end $(date '+%F %T') ==="
tail -5 logs/local_$VAR.log
exit $rc
