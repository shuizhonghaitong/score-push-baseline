#!/bin/bash
cd /mnt/disk3/liuyutong_home/workspace/skill/score-push/runs/13-enveda-CASMI26-molecule-id-mass-spectra
export HTTP_PROXY=http://172.28.4.16:18999 HTTPS_PROXY=http://172.28.4.16:18999
export http_proxy=http://172.28.4.16:18999 https_proxy=http://172.28.4.16:18999
D=data/enveda-casmi26-molecule-id-mass-spectra
for i in $(seq 1 200); do
  if [ -f "$D/train.parquet" ] && [ "$(stat -c%s "$D/train.parquet" 2>/dev/null)" = "3033286496" ]; then
    echo "TRAIN_DONE $(date)" >> work/dl_train3.log
    break
  fi
  /mnt/disk3/liuyutong_home/miniconda3/envs/kaggle/bin/kaggle competitions download enveda-CASMI26-molecule-id-mass-spectra -f train.parquet -p "$D" -o -q >> work/dl_train3.log 2>&1
  echo "attempt $i done rc=$? size=$(stat -c%s "$D/train.parquet" 2>/dev/null || echo 0) $(date)" >> work/dl_train3.log
  sleep 60
done
