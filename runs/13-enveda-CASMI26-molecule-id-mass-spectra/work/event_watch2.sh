#!/bin/bash
# Continuous event watcher: append a line to work/event2.log on every state change.
cd /mnt/disk3/liuyutong_home/workspace/skill/score-push/runs/13-enveda-CASMI26-molecule-id-mass-spectra
declare -A seen
emit() { # $1=tag $2=data
  line="$(date '+%F %T') $1 $2"
  if [ "${seen[$1]}" != "$2" ]; then
    seen[$1]="$2"
    echo "$line" >> work/event2.log
  fi
}
while true; do
  for f in v465_pushed nb_sub_pushed nb_sub_ready nb_sub_submitted nb_sub_score v465_submitted v465_score labels_ready v465_smoke_ok; do
    if [ -f "work/$f.txt" ]; then emit "FLAG_$f" "$(head -c 80 work/$f.txt | tr '\n' ' ')"; else emit "FLAG_$f" "absent"; fi
  done
  emit "LP" "$(tail -1 work/late_push.log 2>/dev/null)"
  emit "TB2" "$(tail -1 work/track_b2.log 2>/dev/null)"
  emit "LBL" "$(tail -1 work/labels.log 2>/dev/null)"
  emit "W2" "$(tail -1 work/v465_watch2.log 2>/dev/null)"
  emit "GATE" "$(tail -1 work/submit_gate.log 2>/dev/null)"
  emit "AUTO" "$(tail -1 work/auto_v465.log 2>/dev/null)"
  emit "SEQ" "$(tail -1 work/sequencer.log 2>/dev/null)"
  emit "CANARY" "$(cat gpu-jobs/1792690910738588428-c2877ba7/status.json 2>/dev/null | tr -d '\n ')"
  emit "TOKEN_AGE_MIN" "$(( ( $(date +%s) - $(stat -c %Y /mnt/disk3/liuyutong_home/workspace/skill/score-push/.private/kaggle-access-token 2>/dev/null || echo 0) ) / 60 ))"
  sleep 300
done
