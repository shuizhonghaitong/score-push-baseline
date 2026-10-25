#!/bin/bash
# TPU change detector (timeout-guarded).
RUN=/mnt/disk3/liuyutong_home/workspace/skill/score-push/runs/12-neurips-2026-smart-buildings
export PATH=/mnt/disk3/liuyutong_home/miniconda3/envs/kaggle/bin:$PATH
export HTTP_PROXY=http://172.28.4.16:18999 HTTPS_PROXY=http://172.28.4.16:18999
export http_proxy=http://172.28.4.16:18999 https_proxy=http://172.28.4.16:18999
prev=QUEUED
while true; do
  s=$(timeout 120 kaggle kernels status lilac1996/smart-buildings-v4tpu 2>/dev/null | grep -o 'KernelWorkerStatus\.[A-Z]*' | head -1)
  cur=${s##*.}
  if [ -n "$cur" ] && [ "$cur" != "$prev" ]; then
    echo "[$(date '+%F %T')] CHANGE: $cur" | tee -a "$RUN/results/tpu_change.log"
    prev=$cur
    if [ "$cur" != "QUEUED" ]; then break; fi
  fi
  sleep 240
done
