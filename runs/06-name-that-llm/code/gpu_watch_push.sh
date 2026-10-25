#!/bin/bash
cd /mnt/disk3/liuyutong_home/workspace/skill/score-push/runs/06-name-that-llm
source env.sh
S=/mnt/disk3/liuyutong_home/workspace/skill/score-push/skills/kaggle/scripts/kaggle_skill.py
log() { echo "$(date '+%F %T') $*" | tee -a work/gpu_watch.log; }
log "watch started (pid $$)"
PUSHED_B=0
while true; do
  S1=$(kaggle kernels status lilac1996/tfb-run-v3b 2>/dev/null | tail -1)
  S2=$(kaggle kernels status lilac1996/tfb-run-v3 2>/dev/null | tail -1)
  log "poll: $S1 | $S2"
  if ! echo "$S1 $S2" | grep -q "KernelWorkerStatus.RUNNING"; then
    log "slots appear free, pushing notebookB"
    (cd notebookB && python $S notebook-push . --yes >> work/gpu_watch.log 2>&1)
    if tail -50 work/gpu_watch.log | grep -q "Maximum batch GPU session count"; then
      log "push hit cap again (race); keep polling"
      continue
    fi
    PUSHED_B=1
    log "notebookB pushed ok"
    (cd notebookE && python $S notebook-push . --yes >> work/gpu_watch.log 2>&1)
    tail -20 work/gpu_watch.log
    log "push phase done; watcher exiting"
    break
  fi
  sleep 180
done
