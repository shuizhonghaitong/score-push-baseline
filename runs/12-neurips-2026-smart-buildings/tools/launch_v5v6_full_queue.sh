#!/bin/bash
# Queue: v5 209-scenario full eval, then v6 209-scenario full eval.
# Starts once the V4g winter run finished (or after a 10:00 fallback deadline).
RUN=/mnt/disk3/liuyutong_home/workspace/skill/score-push/runs/12-neurips-2026-smart-buildings
cd "$RUN"
LOG=downloads/eval_queue.log
echo "[$(date '+%F %T')] queue started" >> "$LOG"
# Wait for v4g winter to finish (its JSON appears at the end of that run),
# or give up waiting at 10:00 to protect the schedule.
deadline=$(( $(date -d 'today 10:00' +%s) ))
while [ ! -f results/eval_v4g_winter.json ]; do
  if [ $(date +%s) -ge $deadline ]; then
    echo "[$(date '+%F %T')] fallback deadline reached; skipping wait" >> "$LOG"
    break
  fi
  sleep 120
done
sleep 90
run_eval() {
  local mod=$1 cls=$2 tag=$3
  OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp .venv10/bin/python tools/eval_local.py \
    --policy-module policies/$mod --policy-class $cls --steps 8640 --workers 8 \
    --out results/eval_${tag}.json > results/eval_${tag}.log 2>&1
  echo "[$(date '+%F %T')] ${tag} done rc=$?" >> "$LOG"
}
run_eval v5_policy.py V5Policy v5_full
echo "[$(date '+%F %T')] v6 full skipped by design (see PLAN)" >> "$LOG"
echo "[$(date '+%F %T')] FULL QUEUE DONE" >> "$LOG"
