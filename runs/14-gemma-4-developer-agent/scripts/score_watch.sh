#!/bin/bash
# Watch scores of this run's submissions (v1 + v2), 30-min cadence, 48h.
set -u
cd /mnt/disk3/liuyutong_home/workspace/skill/score-push/runs/14-gemma-4-developer-agent
PY=/mnt/disk3/liuyutong_home/miniconda3/envs/kaggle/bin/python
SK=/mnt/disk3/liuyutong_home/workspace/skill/score-push/skills/kaggle/scripts/kaggle_skill.py
LOG=results/scores_watch.log
END=$(( $(date +%s) + 48*3600 ))
while [ $(date +%s) -lt $END ]; do
  OUT=$($PY $SK status gemma-4-developer-agent 2>&1)
  L1=$(echo "$OUT" | grep '57504328' | head -1 | sed 's/^ *//')
  L2=$(echo "$OUT" | grep '57542053' | head -1 | sed 's/^ *//')
  RANK=$(echo "$OUT" | grep 'your rank' | head -1 | sed 's/^ *//')
  echo "$(date '+%F %T') v1=${L1:-?}" | tee -a $LOG
  echo "$(date '+%F %T') v2=${L2:-?}" | tee -a $LOG
  echo "$(date '+%F %T') $RANK" | tee -a $LOG
  sleep 1800
done
echo "$(date '+%F %T') SCORE_WATCH done" | tee -a $LOG
