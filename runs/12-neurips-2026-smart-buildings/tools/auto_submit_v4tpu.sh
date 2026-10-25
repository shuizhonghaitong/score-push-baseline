#!/bin/bash
# Poll v4tpu; on COMPLETE: submit (fresh daily quota). On ERROR/CANCEL: log + exit.
RUN=/mnt/disk3/liuyutong_home/workspace/skill/score-push/runs/12-neurips-2026-smart-buildings
SKILL=/mnt/disk3/liuyutong_home/workspace/skill/score-push/skills/kaggle/scripts/kaggle_skill.py
PY=/mnt/disk3/liuyutong_home/miniconda3/envs/kaggle/bin/python
LOG="$RUN/results/auto_submit_v4tpu.log"
export PATH=/mnt/disk3/liuyutong_home/miniconda3/envs/kaggle/bin:$PATH
export HTTP_PROXY=http://172.28.4.16:18999 HTTPS_PROXY=http://172.28.4.16:18999
export http_proxy=http://172.28.4.16:18999 https_proxy=http://172.28.4.16:18999
i=0
while true; do
  i=$((i+1))
  ts=$(date '+%F %T')
  status=$("$PY" -m kaggle kernels status lilac1996/smart-buildings-v4tpu 2>&1 | head -1)
  case "$status" in
    *COMPLETE*)
      echo "[$ts] v4tpu COMPLETE -> submitting (attempt $i)" >> "$LOG"
      out=$("$PY" "$SKILL" submit neurips-2026-smart-buildings --notebook lilac1996/smart-buildings-v4tpu --version 1 -m "v4 per-zone PI + SAT ramp + pre-latch" --yes 2>&1)
      code=$?
      { echo "[$ts] submit rc=$code"; echo "$out" | tail -5; } >> "$LOG"
      if [ $code -eq 0 ]; then
        echo "[$ts] SUBMITTED v4tpu v1" >> "$LOG"
        exit 0
      fi
      echo "[$ts] submit failed; will retry in 10 min" >> "$LOG"
      sleep 600
      ;;
    *ERROR*|*CANCEL*)
      echo "[$ts] v4tpu $status (attempt $i) -> exiting for manual review" >> "$LOG"
      exit 1
      ;;
    *)
      echo "[$ts] still: $status" >> "$LOG"
      sleep 120
      ;;
  esac
done
