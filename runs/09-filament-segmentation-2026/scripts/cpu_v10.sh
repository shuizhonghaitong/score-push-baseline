#!/bin/bash
source /mnt/disk3/liuyutong_home/workspace/skill/score-push/runs/09-filament-segmentation-2026/env.sh
RUN=/mnt/disk3/liuyutong_home/workspace/skill/score-push/runs/09-filament-segmentation-2026
CPU=lilac1996/filament-magfilo-v1-cpu-u-net
VDIR="$RUN/src/notebooks/v1-cpu-fast2"
echo "=== cpu_v10 start $(date -Is) ==="
while true; do
  st=$(kaggle kernels status "$CPU" 2>&1 | grep -oE 'KernelWorkerStatus\.[A-Z_]+' | head -1)
  echo "$(date -Is) cpu_status=$st"
  case "$st" in
    *COMPLETE*) echo "cpu kernel complete"; exit 0 ;;
    *RUNNING*)
      if [ -f "$VDIR/.kaggle-skill/notebook-push.json" ]; then echo "v10 already running; waiting"; sleep 540; continue; fi
      ;;
  esac
  if (cd "$VDIR" && python "$KAGGLE_SKILL/modules/notebooks/scripts/notebook_push.py" . --yes >> "$RUN/logs/vpush.log" 2>&1); then
    echo "=== v10 pushed $(date -Is) ==="
  else
    echo "v10 push failed (slot); retry next cycle"
  fi
  sleep 540
done
