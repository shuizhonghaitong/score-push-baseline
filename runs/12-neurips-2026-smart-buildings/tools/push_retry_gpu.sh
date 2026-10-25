#!/bin/bash
# Retry loop: push the GPU (T4) variant of the current best notebook.
RUN=/mnt/disk3/liuyutong_home/workspace/skill/score-push/runs/12-neurips-2026-smart-buildings
SKILL=/mnt/disk3/liuyutong_home/workspace/skill/score-push/skills/kaggle/scripts/kaggle_skill.py
PY=/mnt/disk3/liuyutong_home/miniconda3/envs/kaggle/bin/python
DIR="$RUN/notebooks/submission-v2-gpu"
LOG="$RUN/results/push_gpu.log"
STATUS="$RUN/push_gpu_status.txt"
i=0
while true; do
  i=$((i+1))
  ts=$(date '+%F %T')
  out=$("$PY" "$SKILL" notebook-push "$DIR" --yes 2>&1)
  code=$?
  { echo "[$ts] attempt $i rc=$code"; echo "$out" | tail -6; } >> "$LOG"
  if [ $code -eq 0 ]; then
    { echo "[$ts] PUSH rc=0 attempt $i"; echo "$out" | tail -3; } > "$STATUS"
    break
  fi
  sleep 360
done
