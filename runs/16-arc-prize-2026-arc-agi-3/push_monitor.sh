#!/bin/bash
RUN=/mnt/disk3/liuyutong_home/workspace/skill/score-push/runs/16-arc-prize-2026-arc-agi-3
cd "$RUN"
export PATH=/mnt/disk3/liuyutong_home/miniconda3/envs/kaggle/bin:$PATH
LOG=$RUN/push_monitor.log

try_push() {
  local dir=$1 slug=$2 out tail
  out=$(timeout 240 kaggle kernels push -p "$dir" 2>&1)
  tail=$(echo "$out" | tail -4 | tr '\n' ' | ')
  echo "[$(date '+%m-%d %H:%M:%S')] try $slug :: $tail" >> "$LOG"
  if echo "$tail" | grep -qi "error"; then return 1; fi
  if kaggle kernels list --mine --page-size 200 --format json 2>/dev/null | grep -q "lilac1996/$slug"; then
    echo "[$(date '+%m-%d %H:%M:%S')] PUSH_SUCCESS $slug" >> "$LOG"
    touch "$RUN/.pushed_$slug"
    return 0
  fi
  echo "[$(date '+%m-%d %H:%M:%S')] $slug: no error text but kernel not listed yet" >> "$LOG"
  return 1
}

echo "[$(date '+%m-%d %H:%M:%S')] monitor v2 start pid=$$" >> "$LOG"
for i in $(seq 1 300); do
  v2=0; v1=0
  [ -f "$RUN/.pushed_arc-agi-3-duck-harness-run16-v2" ] && v2=1
  [ -f "$RUN/.pushed_arc-agi-3-duck-harness-run16-v1" ] && v1=1
  if [ $v2 -eq 0 ]; then try_push notebooks/duck_v2 arc-agi-3-duck-harness-run16-v2 && v2=1; fi
  if [ $v1 -eq 0 ]; then try_push notebooks/duck_v1 arc-agi-3-duck-harness-run16-v1 && v1=1; fi
  if [ $v2 -eq 1 ] && [ $v1 -eq 1 ]; then
    echo "[$(date '+%m-%d %H:%M:%S')] BOTH PUSHED" >> "$LOG"
    break
  fi
  sleep 300
done
echo "[$(date '+%m-%d %H:%M:%S')] monitor exit" >> "$LOG"
