#!/bin/bash
export PATH=/mnt/disk3/liuyutong_home/miniconda3/envs/kaggle/bin:$PATH
cd /mnt/disk3/liuyutong_home/workspace/skill/score-push/runs/14-gemma-4-developer-agent
for i in $(seq 1 30); do
  kaggle competitions download gemma-4-developer-agent --path data > dl.log 2>&1
  rc=$?
  echo "attempt $i rc=$rc $(date)" >> dl_retry.log
  if [ $rc -eq 0 ]; then echo "SUCCESS" >> dl_retry.log; break; fi
  sleep 60
done
