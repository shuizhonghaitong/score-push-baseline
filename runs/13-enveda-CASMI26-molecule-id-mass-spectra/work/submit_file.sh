#!/bin/bash
# Plain-file submission: no SaveKernel involved -> no 429-window risk, any time.
# Usage: bash work/submit_file.sh <submission.csv> <message>
cd /mnt/disk3/liuyutong_home/workspace/skill/score-push/runs/13-enveda-CASMI26-molecule-id-mass-spectra
export PATH=/mnt/disk3/liuyutong_home/miniconda3/envs/kaggle/bin:$PATH
export HTTP_PROXY=http://172.28.4.16:18999 HTTPS_PROXY=http://172.28.4.16:18999
export http_proxy=http://172.28.4.16:18999 https_proxy=http://172.28.4.16:18999
CSV=$1; MSG=$2
[ -f "$CSV" ] || { echo "NO FILE $CSV"; exit 2; }
TOK=/mnt/disk3/liuyutong_home/workspace/skill/score-push/.private/kaggle-access-token
SK=/mnt/disk3/liuyutong_home/workspace/skill/score-push/skills/kaggle/scripts/kaggle_skill.py
for try in 1 2 3; do
  out=$(/usr/bin/timeout 600 kaggle competitions submit enveda-CASMI26-molecule-id-mass-spectra -f "$CSV" -m "$MSG" 2>&1)
  rc=$?
  echo "$out" | tail -5
  if [ $rc -eq 0 ]; then break; fi
  if echo "$out" | grep -qE "401|Unauthorized|invalid"; then
    echo "401 -> refreshing token (attempt $try)"; sleep 10
    /usr/bin/timeout 900 python3 $SK auth-session --output $TOK --yes >/dev/null 2>&1
    continue
  fi
  echo "submit rc=$rc; retrying in 300s"; sleep 300
done
echo "### polling for score"
for i in $(seq 1 60); do
  s=$(/usr/bin/timeout 120 kaggle competitions submissions enveda-CASMI26-molecule-id-mass-spectra 2>&1)
  if ! echo "$s" | grep -qE "IN PROGRESS|INPROGRESS"; then
    echo "$s" | head -8
    break
  fi
  sleep 300
done
