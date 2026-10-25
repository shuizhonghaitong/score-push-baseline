#!/bin/bash
cd /mnt/disk3/liuyutong_home/workspace/skill/score-push/runs/06-name-that-llm || exit 1
source env.sh
LOG=work/push_c2_loop2.log
S=/mnt/disk3/liuyutong_home/workspace/skill/score-push/skills/kaggle/scripts/kaggle_skill.py
log() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
# only start if the first loop didn't get it
if [ -f work/.c2_pushed ]; then log "C2 already pushed by loop1"; exit 0; fi
log "push_c2_retry2 started (30min cadence)"
for i in $(seq 1 7); do
  out=$(timeout 600 python3 $S notebook-push notebookC2 --yes 2>&1)
  if echo "$out" | grep -qE "429|Too Many|ProxyError|Max retries"; then
    log "attempt $i: rate-limited/proxy, sleep 1800"
    sleep 1800
  else
    ver=$(echo "$out" | grep -oE 'KernelVersion: [0-9]+' | grep -oE '[0-9]+' | tail -1)
    log "attempt $i: PUSHED ver=${ver:-?}"
    [ -n "$ver" ] && echo "$ver" > work/pushver_ntl-run06-v3-cpu
    touch work/.c2_pushed
    log "=== push_c2_retry2 DONE ==="
    exit 0
  fi
done
log "=== push_c2_retry2 GAVE UP ==="
