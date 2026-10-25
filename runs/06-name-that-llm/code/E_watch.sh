#!/bin/bash
cd /mnt/disk3/liuyutong_home/workspace/skill/score-push/runs/06-name-that-llm
source env.sh
LOG=work/E_watch.log
log() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
log "E_watch v2 started (pid $$)"
while true; do
  st=$(timeout 90 kaggle kernels status lilac1996/ntl-run06-fork08b 2>&1 | tail -1)
  bu=$(timeout 90 kaggle quota 2>&1 | sed -n '3p' | awk '{print $2"/"$3}')
  log "E: $st | GPU $bu"
  if echo "$st" | grep -q "COMPLETE"; then log "E COMPLETE"; break; fi
  if echo "$st" | grep -q "CANCELLED\|FAILED\|error\|denied"; then log "E ENDED-NONCOMPLETE: $st"; break; fi
  sleep 60
done
log "E_watch ended"
