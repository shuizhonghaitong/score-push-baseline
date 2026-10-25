#!/bin/bash
source /mnt/disk3/liuyutong_home/workspace/skill/score-push/runs/06-name-that-llm/env.sh 2>/dev/null
export S=/mnt/disk3/liuyutong_home/workspace/skill/score-push/skills/kaggle/scripts/kaggle_skill.py
cd /mnt/disk3/liuyutong_home/workspace/skill/score-push/runs/06-name-that-llm/notebookB
for i in $(seq 1 72); do
  ts=$(date +%H:%M:%S)
  out=$(python $S notebook-push . --yes 2>&1)
  echo "[$ts] attempt $i: $(echo "$out" | tail -1)" >> ../work/push_retry.log
  if echo "$out" | grep -q "Maximum batch GPU session count"; then
    sleep 600
  else
    echo "[$ts] PUSH ACCEPTED (attempt $i)" >> ../work/push_retry.log
    break
  fi
done
