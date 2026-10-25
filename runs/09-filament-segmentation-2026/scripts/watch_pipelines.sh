#!/bin/bash
source /mnt/disk3/liuyutong_home/workspace/skill/score-push/runs/09-filament-segmentation-2026/env.sh
RUN=/mnt/disk3/liuyutong_home/workspace/skill/score-push/runs/09-filament-segmentation-2026
while true; do
  ts=$(date -Is)
  gpush=$(tail -n 6 "$RUN/logs/gpu_pipeline.log" | grep -E "push|PUSH" | tail -2 | tr '\n' '|')
  cpush=$(tail -n 4 "$RUN/logs/cpu_v10.log" | grep -E "cpu_status|pushed|complete" | tail -2 | tr '\n' '|')
  lasterr=$(tail -c 2000 "$RUN/logs/vpush.log" | grep -oE "Maximum batch [A-Z]+ session count of [0-9]+ reached|429 Client Error: Too Many Requests" | tail -1)
  echo "$ts gpu=[${gpush}] cpu=[${cpush}] err=[${lasterr}]" >> "$RUN/logs/watch.log"
  sleep 600
done
