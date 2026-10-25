#!/bin/bash
# Restart the GPU v1 pipeline (use after budget reset 2026-10-24 00:00 UTC
# if session 42554 died or its notebook_wait timed out).
# Run as a persistent exec session:  bash scripts/restart_gpu_pipeline.sh
source /mnt/disk3/liuyutong_home/workspace/skill/score-push/runs/09-filament-segmentation-2026/env.sh
RUN=/mnt/disk3/liuyutong_home/workspace/skill/score-push/runs/09-filament-segmentation-2026
GPU=lilac1996/filament-magfilo-v1
st=$(kaggle kernels status "$GPU" 2>&1 | grep -oE 'KernelWorkerStatus\.[A-Z_]+' | head -1)
echo "restart: current status=$st $(date -Is)" >> "$RUN/logs/gpu_pipeline.log"
if [[ "$st" == *RUNNING* || "$st" == *COMPLETE* ]]; then
  echo "kernel already RUNNING/COMPLETE; skipping push" >> "$RUN/logs/gpu_pipeline.log"
  if [[ "$st" == *COMPLETE* ]]; then
    python "$KAGGLE_SKILL/modules/notebooks/scripts/notebook_wait.py" "$GPU" --out "$RUN/output/v1-gpu" --timeout 600 --interval 60
  fi
  exit 0
fi
echo "=== gpu_pipeline v7 (restart) start $(date -Is) ===" >> "$RUN/logs/gpu_pipeline.log"
while true; do
  echo "$(date -Is) pushing gpu kernel (self-contained)" >> "$RUN/logs/gpu_pipeline.log"
  if (cd "$RUN/src/notebooks/v1" && python "$KAGGLE_SKILL/modules/notebooks/scripts/notebook_push.py" . --yes >> "$RUN/logs/vpush.log" 2>&1); then
    echo "=== GPU PUSH OK $(date -Is) ===" >> "$RUN/logs/gpu_pipeline.log"
    break
  fi
  sleep 540
done
python "$KAGGLE_SKILL/modules/notebooks/scripts/notebook_wait.py" "$GPU" --out "$RUN/output/v1-gpu" --timeout 43000 --interval 120
rc=$?
echo "=== GPU PIPELINE EXIT rc=$rc $(date -Is) ===" >> "$RUN/logs/gpu_pipeline.log"
exit $rc
