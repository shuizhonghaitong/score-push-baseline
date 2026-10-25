#!/bin/bash
export HTTP_PROXY=http://172.28.4.16:18999 HTTPS_PROXY=http://172.28.4.16:18999 http_proxy=http://172.28.4.16:18999 https_proxy=http://172.28.4.16:18999
export PATH=/mnt/disk3/liuyutong_home/miniconda3/envs/kaggle/bin:$PATH
RUN=/mnt/disk3/liuyutong_home/workspace/skill/score-push/runs/05-misconception-detection-in-k-12-student-math-workings
cd $RUN
LOG=$RUN/logs/fetch_full_data.log
say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $LOG; }
START=$(date +%s)
N=0
while true; do
  N=$((N+1))
  if [ $(( $(date +%s) - START )) -gt 64800 ]; then say "GIVEUP after 18h"; exit 1; fi
  out=$(timeout 1800 kaggle competitions download -c misconception-detection-in-k-12-student-math-workings -o /tmp/ 2>&1)
  if [ -f /tmp/misconception-detection-in-k-12-student-math-workings.zip ] && [ $(stat -c%s /tmp/misconception-detection-in-k-12-student-math-workings.zip) -gt 1000000 ]; then
    say "DOWNLOADED $(stat -c%s /tmp/misconception-detection-in-k-12-student-math-workings.zip) bytes (attempt $N)"
    mkdir -p /tmp/miscc_full && (cd /tmp/miscc_full && unzip -o -q /tmp/misconception-detection-in-k-12-student-math-workings.zip)
    # copy images into run data dir
    SRC=$(find /tmp/miscc_full -maxdepth 2 -type d -name images | head -1)
    if [ -n "$SRC" ]; then
      cp -rn $SRC/. $RUN/data/images/ 2>/dev/null
      say "images copied; total local: $(find $RUN/data/images -name '*.jpeg' | wc -l)"
    fi
    say "FETCH_FULL_DATA OK"
    exit 0
  fi
  say "attempt $N failed: $(echo "$out" | tail -1)"
  sleep 1800
done
