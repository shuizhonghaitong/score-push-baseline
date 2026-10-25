#!/bin/bash
cd /mnt/disk3/liuyutong_home/workspace/skill/score-push/runs/13-enveda-CASMI26-molecule-id-mass-spectra
export HTTP_PROXY=http://172.28.4.16:18999 HTTPS_PROXY=http://172.28.4.16:18999
export http_proxy=http://172.28.4.16:18999 https_proxy=http://172.28.4.16:18999
LOG=work/push_v465.log
for i in $(seq 1 600); do
  echo "=== v465 cycle $i $(date +%H:%M:%S) ===" | tee -a $LOG
  kaggle kernels push -p work/nb_v465 > /tmp/push_v465.txt 2>&1
  if grep -qiE "https://www.kaggle.com/code/|kernel pushed|pushed version" /tmp/push_v465.txt; then
    echo "V465 PUSH OK at $(date)" | tee -a $LOG
    cat /tmp/push_v465.txt | tee -a $LOG
    echo "PUSHED" > work/v465_pushed.txt
    break
  fi
  tail -2 /tmp/push_v465.txt | tee -a $LOG
  sleep 150
done
echo "=== v465 loop ended $(date) ===" | tee -a $LOG
