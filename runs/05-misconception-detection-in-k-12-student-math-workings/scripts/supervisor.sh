#!/bin/bash
# Long-horizon supervisor: logs a timeline of the n7 pipeline + scores. Read-only (no pushes).
export HTTP_PROXY=http://172.28.4.16:18999 HTTPS_PROXY=http://172.28.4.16:18999 http_proxy=http://172.28.4.16:18999 https_proxy=http://172.28.4.16:18999
export PATH=/mnt/disk3/liuyutong_home/miniconda3/envs/kaggle/bin:$PATH
RUN=/mnt/disk3/liuyutong_home/workspace/skill/score-push/runs/05-misconception-detection-in-k-12-student-math-workings
cd $RUN
LOG=$RUN/logs/supervisor.log
say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $LOG; }
SK=/mnt/disk3/liuyutong_home/workspace/skill/score-push/skills/kaggle/scripts/kaggle_skill.py
SLUG=lilac1996/misconception-detection-vlm-lora-v1
START=$(date +%s)
say "supervisor start"; LASTST=""; LASTBEST=""; RND=0
while true; do
  if [ $(( $(date +%s) - START )) -gt 300000 ]; then say "supervisor stop 83h"; break; fi
  # pusher progress (new lines)
  if [ -f logs/push_n7_v4.log ]; then
    last=$(tail -1 logs/push_n7_v4.log)
    echo "$last" | grep -q "PUSH rc=0" && say "N7 PUSH ACCEPTED :: $last"
    echo "$last" | grep -qE "COMPLETE \+ FETCHED|COMPLETE_TAGGED" && say "N7 COMPLETE+FETCHEED :: $last"
  fi
  # kernel status (cheap read, dedup)
  st=$(timeout 45 kaggle kernels status $SLUG 2>&1 | grep -o 'KernelWorkerStatus\.[A-Z]*')
  if [ -n "$st" ] && [ "$st" != "$LASTST" ]; then say "n7 kernel: $st"; LASTST=$st; fi
  # n8
  if [ -f logs/push_n8.log ]; then
    echo "$(tail -1 logs/push_n8.log)" | grep -q "N8 PUSH rc=0" && say "N8 PUSH ACCEPTED :: $(tail -1 logs/push_n8.log)"
    echo "$(tail -1 logs/push_n8.log)" | grep -q "N8 COMPLETE + FETCHED" && say "N8 COMPLETE+FETCHEED :: $(tail -1 logs/push_n8.log)"
  fi
  # followup actions
  if [ -f logs/n7_followup.log ]; then
    grep -c "v6a: pure" logs/n7_followup.log >/dev/null 2>&1
    lastf=$(tail -1 logs/n7_followup.log)
    echo "$lastf" | grep -q "N8 pusher" && say "N8 launched by followup"
    echo "$lastf" | grep -q "N7_FOLLOWUP done" && say "followup finished :: $lastf"
  fi
  # scores (best public, dedup, every 3rd round = 90 min)
  RND=$((RND+1))
  if [ $((RND % 3)) -eq 1 ]; then
    best=$(timeout 45 python3 $SK status misconception-detection-in-k-12-student-math-workings 2>&1 | grep -o "best public: [0-9.]*" | head -1)
    if [ -n "$best" ] && [ "$best" != "$LASTBEST" ]; then say "$best"; LASTBEST=$best; fi
  fi
  sleep 1800
done
