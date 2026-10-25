#!/bin/bash
# Manual winter compare (3 policies), tagged -now to avoid clobbering the auto one.
RUN=/mnt/disk3/liuyutong_home/workspace/skill/score-push/runs/12-neurips-2026-smart-buildings
cd "$RUN"
WEATHERS="fairbanks_1 fairbanks_3 fairbanks_8 minneapolis_1 minneapolis_2 denver_1 atlanta_4 singapore_8 phoenix_5"
for spec in "v4_policy V4Policy v4" "v5_policy V5Policy v5" "v4_variants V4fPolicy v4f"; do
  set -- $spec
  mod=policies/$1.py; cls=$2; tag=$3
  OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp .venv10/bin/python tools/eval_local.py \
    --policy-module "$mod" --policy-class "$cls" --steps 8640 --workers 8 \
    --weathers $WEATHERS --diagnose --out results/eval_${tag}_winter_now.json \
    > results/eval_${tag}_winter_now.log 2>&1
  echo "[$(date '+%F %T')] $tag done rc=$?" >> downloads/winter_now.log
done
echo "[$(date '+%F %T')] ALL DONE" >> downloads/winter_now.log
