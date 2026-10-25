#!/bin/bash
export PATH=/mnt/disk3/liuyutong_home/miniconda3/envs/kaggle/bin:$PATH
export HTTPS_PROXY=http://172.28.4.16:18999 HTTP_PROXY=http://172.28.4.16:18999 https_proxy=http://172.28.4.16:18999 http_proxy=http://172.28.4.16:18999
cd /mnt/disk3/liuyutong_home/workspace/skill/score-push/runs/05-misconception-detection-in-k-12-student-math-workings
LOG=logs/wait_download.log
echo "[$(date -Iseconds)] waiting for rate limit reset" > $LOG
for i in $(seq 1 60); do
  if kaggle competitions download -c misconception-detection-in-k-12-student-math-workings --file "competition_data/taxonomy_v1.csv" --path data -q >> $LOG 2>&1 && [ -s data/taxonomy_v1.csv ]; then
    echo "[$(date -Iseconds)] rate limit cleared; starting paced download" >> $LOG
    .venv/bin/python scripts/download_data.py --workers 3 >> $LOG 2>&1
    echo "[$(date -Iseconds)] paced download finished" >> $LOG
    exit 0
  fi
  sleep 600
done
echo "[$(date -Iseconds)] gave up after 60 attempts" >> $LOG
