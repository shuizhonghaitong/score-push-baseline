#!/bin/bash
source /mnt/disk3/liuyutong_home/workspace/skill/score-push/runs/09-filament-segmentation-2026/env.sh
RUN=/mnt/disk3/liuyutong_home/workspace/skill/score-push/runs/09-filament-segmentation-2026
SLUG=lilac1996/filament-magfilo-v1-cpu-u-net
echo "monitor start $(date -Is)" | tee "$RUN/logs/cpu_progress.log"
for i in $(seq 1 70); do
  st=$(kaggle kernels status "$SLUG" 2>&1 | grep -oE 'KernelWorkerStatus\.[A-Z_]+' | head -1)
  echo "---- $(date -Is) status=$st probe $i" >> "$RUN/logs/cpu_progress.log"
  case "$st" in
    *COMPLETE*|*ERROR*|*CANCEL*) echo "TERMINAL status=$st" >> "$RUN/logs/cpu_progress.log"; break ;;
  esac
  timeout 70 kaggle kernels logs "$SLUG" -f 2>/dev/null | tail -c 1500 >> "$RUN/logs/cpu_progress.log"
  echo "" >> "$RUN/logs/cpu_progress.log"
  sleep 540
done
echo "monitor end $(date -Is)" >> "$RUN/logs/cpu_progress.log"
