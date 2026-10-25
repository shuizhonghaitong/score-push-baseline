#!/bin/bash
# C2 push loop v3 (agent-2): push until the kernel actually registers.
# Verifies registration via kernels status (guards against false-positive "PUSHED").
cd /mnt/disk3/liuyutong_home/workspace/skill/score-push/runs/06-name-that-llm || exit 1
source env.sh
S=/mnt/disk3/liuyutong_home/workspace/skill/skill/score-push/skills/kaggle/scripts/kaggle_skill.py
[ -f "$S" ] || S=/mnt/disk3/liuyutong_home/workspace/skill/score-push/skills/kaggle/scripts/kaggle_skill.py
LOG=work/push_c2_loop3.log
log() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
i=0
while true; do
  i=$((i+1))
  out=$(timeout 900 python3 $S notebook-push notebookC2 --yes 2>&1)
  if echo "$out" | grep -qE "429|Too Many|ProxyError|Max retries"; then
    log "attempt $i: 429/proxy, sleep 900"
    sleep 900
    continue
  fi
  ver=$(echo "$out" | grep -oE 'KernelVersion: [0-9]+' | grep -oE '[0-9]+' | tail -1)
  sleep 60
  if timeout 90 kaggle kernels status lilac1996/ntl-run06-v3-cpu 2>/dev/null | grep -q "KernelWorkerStatus"; then
    [ -n "$ver" ] && echo "$ver" > work/pushver_ntl-run06-v3-cpu
    touch work/.c2_pushed
    log "attempt $i: C2 REGISTERED (ver=${ver:-?}) -> done"
    exit 0
  fi
  log "attempt $i: non-429 but NOT registered (false positive): $(echo "$out" | grep -iv 'untrusted\|^\s*$' | tail -2 | tr '\n' ' ') - retry in 600"
  sleep 600
done
