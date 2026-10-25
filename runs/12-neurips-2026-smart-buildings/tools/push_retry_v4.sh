#!/bin/bash
# Retry loop: push v4 (then v2) notebook variants until one lands.
RUN=/mnt/disk3/liuyutong_home/workspace/skill/score-push/runs/12-neurips-2026-smart-buildings
SKILL=/mnt/disk3/liuyutong_home/workspace/skill/score-push/skills/kaggle/scripts/kaggle_skill.py
PY=/mnt/disk3/liuyutong_home/miniconda3/envs/kaggle/bin/python
LOG="$RUN/results/push_v4.log"
i=0
while true; do
  i=$((i+1))
  ts=$(date '+%F %T')
  for dir in submission-v4 submission-v4-gpu submission-v2 submission-v2-gpu; do
    out=$("$PY" "$SKILL" notebook-push "$RUN/notebooks/$dir" --yes 2>&1)
    code=$?
    { echo "[$ts] attempt $i $dir rc=$code"; echo "$out" | tail -3; } >> "$LOG"
    if [ $code -eq 0 ]; then
      { echo "[$ts] PUSHED $dir attempt $i"; } > "$RUN/push_v4_status.txt"
      exit 0
    fi
    sleep 45
  done
  sleep 240
done
