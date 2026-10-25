#!/bin/bash
cd /mnt/disk3/liuyutong_home/workspace/skill/score-push/runs/06-name-that-llm || exit 1
source env.sh
LOG=work/push_c2_loop.log
S=/mnt/disk3/liuyutong_home/workspace/skill/skill/score-push/skills/kaggle/scripts/kaggle_skill.py
[ -f "$S" ] || S=/mnt/disk3/liuyutong_home/workspace/skill/score-push/skills/kaggle/scripts/kaggle_skill.py
log() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
log "push_c2_retry started"
for i in $(seq 1 12); do
  out=$(timeout 600 python3 $S notebook-push notebookC2 --yes 2>&1)
  if echo "$out" | grep -qE "429|Too Many|ProxyError|Max retries"; then
    log "attempt $i: rate-limited/proxy, sleep 600"
    sleep 600
  else
    ver=$(echo "$out" | grep -oE 'KernelVersion: [0-9]+' | grep -oE '[0-9]+' | tail -1)
    log "attempt $i: PUSHED ver=${ver:-?}"; touch work/.c2_pushed
    echo "${ver:-?}" > work/pushver_ntl-run06-v3-cpu
    log "=== push_c2_retry DONE ==="
    exit 0
  fi
done
log "12 attempts exhausted; leaving C2 push to manual handling"
log "=== push_c2_retry GAVE UP ==="
