#!/bin/bash
# Slot-2 submitter. Slot opens 2026-10-23 00:00 UTC (08:00 UTC+8); daily reset
# verified as UTC-midnight (v1 @01:20 UTC 10-22 still counted 'today' at 10:54 UTC).
# Window 00:05-14:00 UTC (extended for reset-time uncertainty), 15-min retry cadence; submits only when slot is open.
set -u
cd /mnt/disk3/liuyutong_home/workspace/skill/score-push/runs/14-gemma-4-developer-agent
PY=/mnt/disk3/liuyutong_home/miniconda3/envs/kaggle/bin/python
SK=/mnt/disk3/liuyutong_home/workspace/skill/score-push/skills/kaggle/scripts/kaggle_skill.py
LOG=results/submit_window.log
echo "$(date '+%F %T') SUBMIT_WINDOW start" | tee -a $LOG

in_window() { # now_epoch
  local n=$1
  local w2a w2b
  w2a=$(date -d '2026-10-23 00:05:00 UTC' +%s)
  w2b=$(date -d '2026-10-23 14:00:00 UTC' +%s)
  [ $n -ge $w2a ] && [ $n -le $w2b ]
}

submit_once() {
  OUT=$($PY $SK status gemma-4-developer-agent 2>&1)
  LEFT=$(echo "$OUT" | grep -o '[0-9]* left of 1 a day' | grep -o '^[0-9]*' | head -1)
  TODAY=$(echo "$OUT" | grep -o '[0-9]* today' | grep -o '^[0-9]*' | head -1)
  if [ -z "${LEFT:-}" ] || [ -z "${TODAY:-}" ]; then
    echo "$(date '+%F %T') status-parse-fail; retrying" | tee -a $LOG
    return 1
  fi
  if [ "$LEFT" != "1" ]; then
    echo "$(date '+%F %T') slot closed (left=$LEFT today=$TODAY); waiting" | tee -a $LOG
    return 2
  fi
  echo "$(date '+%F %T') slot OPEN (left=$LEFT today=$TODAY) -> submitting (auto pick)" | tee -a $LOG
  bash scripts/submit_slot2.sh >> $LOG 2>&1
  RC=$?
  echo "$(date '+%F %T') submit_slot2 rc=$RC" | tee -a $LOG
  if [ $RC -ne 0 ]; then
    echo "$(date '+%F %T') submit FAILED rc=$RC" | tee -a $LOG
    return 1
  fi
  sleep 20
  VER=$($PY $SK status gemma-4-developer-agent 2>&1)
  NEWTODAY=$(echo "$VER" | grep -o '[0-9]* today' | grep -o '^[0-9]*' | head -1)
  if [ -n "${NEWTODAY:-}" ] && [ "$NEWTODAY" != "$TODAY" ]; then
    echo "$VER" | sed -n '/latest submissions/,/GPU/p' >> $LOG
    echo "$(date '+%F %T') SUBMIT_WINDOW DONE (today count $TODAY -> $NEWTODAY)" | tee -a $LOG
    return 0
  fi
  echo "$(date '+%F %T') submit rc=0 but today-count unchanged ($TODAY); inconclusive" | tee -a $LOG
  return 1
}

while :; do
  NOW=$(date +%s)
  if ! in_window $NOW; then
    W2A=$(date -d '2026-10-23 00:05:00 UTC' +%s)
    if [ $NOW -lt $W2A ]; then
      sleep $(( W2A - NOW - 30 )); continue
    fi
    echo "$(date '+%F %T') window passed; exiting" | tee -a $LOG
    break
  fi
  submit_once
  RC=$?
  if [ $RC -eq 0 ]; then break; fi
  sleep 900
done
