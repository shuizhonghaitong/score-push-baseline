#!/bin/bash
cd /mnt/disk3/liuyutong_home/workspace/skill/score-push/runs/13-enveda-CASMI26-molecule-id-mass-spectra
export HTTP_PROXY=http://172.28.4.16:18999 HTTPS_PROXY=http://172.28.4.16:18999 http_proxy=http://172.28.4.16:18999 https_proxy=http://172.28.4.16:18999
echo "cooldown start $(date); sleeping 2400s" > work/cooldown_test.log
sleep 2400
echo "=== cooldown test push $(date) ===" >> work/cooldown_test.log
kaggle kernels push -p work/nb_v465 > /tmp/cooldown_test.txt 2>&1
tail -4 /tmp/cooldown_test.txt >> work/cooldown_test.log
if grep -qiE "https://www.kaggle.com/code/|kernel pushed|pushed version" /tmp/cooldown_test.txt; then
  echo "COOLDOWN_PUSH_OK $(date)" | tee -a work/cooldown_test.log
  echo "PUSHED" > work/v465_pushed.txt
elif grep -qi "session count" /tmp/cooldown_test.txt; then
  echo "COOLDOWN_SESSION_FULL (429 cleared!) $(date)" | tee -a work/cooldown_test.log
else
  echo "COOLDOWN_STILL_429_OR_ERR $(date)" | tee -a work/cooldown_test.log
fi
