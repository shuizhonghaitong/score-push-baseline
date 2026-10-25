#!/bin/bash
source /mnt/disk3/liuyutong_home/workspace/skill/score-push/runs/09-filament-segmentation-2026/env.sh
RUN=/mnt/disk3/liuyutong_home/workspace/skill/score-push/runs/09-filament-segmentation-2026
GPU=lilac1996/filament-magfilo-v1
echo "=== gpu_pipeline v6 start $(date -Is) ==="
while true; do
  echo "$(date -Is) pushing gpu kernel (self-contained)"
  if (cd "$RUN/src/notebooks/v1" && python "$KAGGLE_SKILL/modules/notebooks/scripts/notebook_push.py" . --yes >> "$RUN/logs/vpush.log" 2>&1); then
    echo "=== GPU PUSH OK $(date -Is) ==="
    break
  fi
  echo "push failed (likely GPU quota); sleep 9min"
  sleep 540
done
python "$KAGGLE_SKILL/modules/notebooks/scripts/notebook_wait.py" "$GPU" --out "$RUN/output/v1-gpu" --timeout 43000 --interval 120
rc=$?
echo "=== GPU PIPELINE EXIT rc=$rc $(date -Is) ==="
exit $rc
