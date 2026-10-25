#!/bin/bash
# Restores hold_notebook{F,E2,B} -> notebook{F,E2,B} once v1 watcher is done
# (or when v2 starts trying to push F, or after a 24h safety timeout).
cd /mnt/disk3/liuyutong_home/workspace/skill/score-push/runs/06-name-that-llm || exit 1
LOG=work/restore.log
start=$(date +%s)
echo "$(date '+%F %T') restore_dirs started (pid $$)" >> "$LOG"
while true; do
  if grep -q "DONE" work/fe_b_early.log 2>/dev/null; then
    echo "$(date '+%F %T') v1 DONE seen -> restoring" >> "$LOG"; break
  fi
  if grep -q "PUSH notebookF" work/fe_b_early2.log 2>/dev/null; then
    echo "$(date '+%F %T') v2 pushing F -> restoring (retry will pick up dirs)" >> "$LOG"; break
  fi
  if [ $(( $(date +%s) - start )) -gt 86400 ]; then
    echo "$(date '+%F %T') 24h timeout -> restoring" >> "$LOG"; break
  fi
  sleep 120
done
for n in F E2 B; do
  if [ -d "work/hold_notebook$n" ] && [ ! -d "notebook$n" ]; then
    mv "work/hold_notebook$n" "notebook$n"
    echo "$(date '+%F %T') restored notebook$n" >> "$LOG"
  fi
done
echo "$(date '+%F %T') restore_dirs done" >> "$LOG"
