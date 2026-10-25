#!/bin/bash
DIR="$1"; LOG="$2"; FLAG="$3"; CAD="$4"; BKO="$5"; INIT="${6:-0}"
cd /mnt/disk3/liuyutong_home/workspace/skill/score-push/runs/13-enveda-CASMI26-molecule-id-mass-spectra
export HTTP_PROXY=http://172.28.4.16:18999 HTTPS_PROXY=http://172.28.4.16:18999 http_proxy=http://172.28.4.16:18999 https_proxy=http://172.28.4.16:18999
if [ -s "$FLAG" ]; then echo "already pushed; stopping" | tee -a $LOG; exit 0; fi
sleep "$INIT"
for i in $(seq 1 1200); do
  if [ -s "$FLAG" ]; then echo "already pushed; stopping" | tee -a $LOG; exit 0; fi
  echo "=== cycle $i $(date +%H:%M:%S) ===" >> $LOG
  kaggle kernels push -p "$DIR" > /tmp/push_out.txt 2>&1
  if grep -qiE "https://www.kaggle.com/code/|kernel pushed|pushed version" /tmp/push_out.txt; then
    echo "PUSH OK $(date)" | tee -a $LOG
    cat /tmp/push_out.txt | tail -4 | tee -a $LOG
    echo "PUSHED" > "$FLAG"
    exit 0
  fi
  tail -2 /tmp/push_out.txt >> $LOG
  if grep -qiE "429|too many requests|response ended prematurely|rate" /tmp/push_out.txt; then
    sleep "$BKO"
  else
    sleep "$CAD"
  fi
done
echo "loop ended $(date)" | tee -a $LOG
