#!/bin/bash
# Queue: v7 winter-56 (diagnose) then v7 209-scenario full eval.
# Starts once the v5 full eval JSON exists (session 66243 chain).
RUN=/mnt/disk3/liuyutong_home/workspace/skill/score-push/runs/12-neurips-2026-smart-buildings
cd "$RUN"
LOG=downloads/eval_queue.log
echo "[$(date '+%F %T')] v7 queue started" >> "$LOG"
while [ ! -f results/eval_v5_full.json ]; do sleep 120; done
sleep 90
WEATHERS="fairbanks_1 fairbanks_3 fairbanks_8 minneapolis_1 minneapolis_2 denver_1 atlanta_4 singapore_8 phoenix_9"
WEATHERS="fairbanks_1 fairbanks_3 fairbanks_8 minneapolis_1 minneapolis_2 denver_1 atlanta_4 singapore_8 phoenix_5"
OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp .venv10/bin/python tools/eval_local.py \
  --policy-module policies/v7_policy.py --policy-class V7Policy --steps 8640 --workers 8 \
  --weathers $WEATHERS --diagnose --out results/eval_v7_winter.json \
  > results/eval_v7_winter.log 2>&1
echo "[$(date '+%F %T')] v7 winter done rc=$?" >> "$LOG"
OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp .venv10/bin/python tools/eval_local.py \
  --policy-module policies/v7_policy.py --policy-class V7Policy --steps 8640 --workers 8 \
  --out results/eval_v7.json > results/eval_v7.log 2>&1
echo "[$(date '+%F %T')] v7 full done rc=$?" >> "$LOG"
echo "[$(date '+%F %T')] V7 QUEUE DONE" >> "$LOG"
