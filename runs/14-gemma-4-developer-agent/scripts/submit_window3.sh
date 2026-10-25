#!/bin/bash
# Slot-3 submitter (opens 2026-10-24 00:00 UTC = 08:00 UTC+8).
# Hardened: 15-min bounded polls, per-iteration logging, clamped long sleep.
# Submits only when status shows "1 left of 1 a day"; picks via submit_slot2.sh
# (A/B winner if local A/B complete, else v2... NOTE: for slot 3 prefer t04 as
# default if v2 was slot 2 -- override with arg: bash scripts/submit_window3.sh t04)
set -u
cd /mnt/disk3/liuyutong_home/workspace/skill/score-push/runs/14-gemma-4-developer-agent
PY=/mnt/disk3/liuyutong_home/miniconda3/envs/kaggle/bin/python
SK=/mnt/disk3/liuyutong_home/workspace/skill/score-push/skills/kaggle/scripts/kaggle_skill.py
LOG=results/submit_window3.log
PICK=${1:-auto}
W3A=$(date -d '2026-10-24 00:05:00 UTC' +%s)
W3B=$(date -d '2026-10-24 14:00:00 UTC' +%s)
echo "$(date '+%F %T') SUBMIT_WINDOW3 start (window $W3A-$W3B, pick=$PICK)" | tee -a $LOG

while :; do
  NOW=$(date +%s)
  if [ $NOW -ge $W3B ]; then
    echo "$(date '+%F %T') window3 passed; exiting" | tee -a $LOG
    break
  fi
  D=$(( W3A - NOW - 30 ))
  if [ $D -gt 0 ] && [ $D -lt 43200 ]; then
    echo "$(date '+%F %T') sleeping ${D}s until window" | tee -a $LOG
    sleep $D
  else
    sleep 900
  fi
  NOW=$(date +%s)
  if [ $NOW -lt $W3A ]; then continue; fi
  OUT=$($PY $SK status gemma-4-developer-agent 2>&1)
  LEFT=$(echo "$OUT" | grep -o '[0-9]* left of 1 a day' | grep -o '^[0-9]*' | head -1)
  TODAY=$(echo "$OUT" | grep -o '[0-9]* today' | grep -o '^[0-9]*' | head -1)
  echo "$(date '+%F %T') poll left=${LEFT:-?} today=${TODAY:-?}" | tee -a $LOG
  if [ -z "${LEFT:-}" ] || [ -z "${TODAY:-}" ]; then
    echo "$(date '+%F %T') status parse fail; retry" | tee -a $LOG
    continue
  fi
  if [ "$LEFT" != "1" ]; then
    continue
  fi
  echo "$(date '+%F %T') slot OPEN -> submitting (pick=$PICK)" | tee -a $LOG
  if [ "$PICK" = "auto" ]; then
    V3=$($PY scripts/ab_pick3.py 2>> $LOG)
    echo "$(date '+%F %T') ab_pick3 -> $V3" | tee -a $LOG
    bash scripts/submit_slot2.sh "$V3" >> $LOG 2>&1
  else
    bash scripts/submit_slot2.sh "$PICK" >> $LOG 2>&1
  fi
  RC=$?
  echo "$(date '+%F %T') submit rc=$RC" | tee -a $LOG
  if [ $RC -ne 0 ]; then continue; fi
  sleep 20
  VER=$($PY $SK status gemma-4-developer-agent 2>&1)
  NEWTODAY=$(echo "$VER" | grep -o '[0-9]* today' | grep -o '^[0-9]*' | head -1)
  if [ -n "${NEWTODAY:-}" ] && [ "$NEWTODAY" != "$TODAY" ]; then
    echo "$VER" | sed -n '/latest submissions/,/GPU/p' >> $LOG
    echo "$(date '+%F %T') SUBMIT_WINDOW3 DONE (today $TODAY -> $NEWTODAY)" | tee -a $LOG
    break
  fi
  echo "$(date '+%F %T') submit rc=0 but today-count unchanged; inconclusive" | tee -a $LOG
  break
done
