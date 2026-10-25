#!/bin/bash
# Monitor 2: A/B job progress + pool + v1 score (score check every 3rd round ~30min)
set -u
cd /mnt/disk3/liuyutong_home/workspace/skill/score-push/runs/14-gemma-4-developer-agent
PY=/mnt/disk3/liuyutong_home/miniconda3/envs/kaggle/bin/python
SK=/mnt/disk3/liuyutong_home/workspace/skill/score-push/skills/kaggle/scripts/kaggle_skill.py
LOG=results/monitor2.log
ROUND=0
while :; do
  ROUND=$((ROUND+1))
  line="$(date '+%F %T') "
  for r in ab_v1 ab_v2 ab_t04; do
    if [ -f results/$r/progress.jsonl ]; then
      n=$(grep -c . results/$r/progress.jsonl 2>/dev/null); n=${n:-0}
      ok=$(grep -c '"resolved": true' results/$r/progress.jsonl 2>/dev/null); ok=${ok:-0}
      line+="$r=$n/16(res=$ok) "
    else
      line+="$r=- "
    fi
  done
  st() { cat "gpu-jobs/$1/status.json" 2>/dev/null | /mnt/disk3/liuyutong_home/miniconda3/envs/kaggle/bin/python -c "import json,sys; print(json.load(sys.stdin).get('status','?'))" 2>/dev/null || echo '?'; }
  line+="smoke=$(st 1792643636275244690-843c218f) v1=$(st 1792657201934231114-f770f2dc) v2=$(st 1792657202000645198-f2e37667) t04=$(st 1792668846215854484-86f0de9b) "
  if [ $((ROUND % 3)) -eq 0 ]; then
    SCORE=$($PY $SK status gemma-4-developer-agent 2>/dev/null | grep '57504328' | head -1 | sed 's/^ *//' | cut -c1-80)
    line+="score=${SCORE:-none}"
  fi
  echo "$line" | tee -a $LOG
  sleep 600
done
