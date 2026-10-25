#!/bin/bash
# Run the v4d pipeline locally (no torch -> USE_T1_EXTRA=False) once data is present.
# Usage: bash local/run_local.sh
set -e
RUN=/mnt/disk3/liuyutong_home/workspace/skill/score-push/runs/07-2026-ieee-big-data-traffic-flow-bench
cd $RUN
export TFB_DATA_ROOT=$RUN/data_local/kaggle_public
export TFB_ASSET_ROOT=$RUN/assets
export TFB_WORK_ROOT=$RUN/local/work
export TFB_THREADS=${TFB_THREADS:-12}
export CUDA_VISIBLE_DEVICES=
mkdir -p $TFB_WORK_ROOT
# sanity: data root must contain submission_key.csv
test -f $TFB_DATA_ROOT/submission_key.csv || { echo "data_local/submission_key.csv missing"; exit 3; }
test -f $TFB_ASSET_ROOT/manifest.json || { echo "assets/manifest.json missing"; exit 3; }
echo "data: $TFB_DATA_ROOT"
echo "assets: $TFB_ASSET_ROOT"
.venv/bin/python local/nb_local/run_local.py 2>&1 | tail -40
