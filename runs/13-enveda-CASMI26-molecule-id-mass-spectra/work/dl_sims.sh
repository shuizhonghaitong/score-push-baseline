#!/bin/bash
cd /mnt/disk3/liuyutong_home/workspace/skill/score-push/runs/13-enveda-CASMI26-molecule-id-mass-spectra
export HTTP_PROXY=http://172.28.4.16:18999 HTTPS_PROXY=http://172.28.4.16:18999
export http_proxy=http://172.28.4.16:18999 https_proxy=http://172.28.4.16:18999
mkdir -p data/local
for i in 1 2 3 4 5; do
  /mnt/disk3/liuyutong_home/miniconda3/envs/kaggle/bin/kaggle datasets download ahmedberatozer/casmi26-iceberg -p data/local/iceberg -o -q >> work/dl_sims.log 2>&1 && break
  echo "iceberg attempt $i failed, sleep 120" >> work/dl_sims.log; sleep 120
done
unzip -oq data/local/iceberg/casmi26-iceberg.zip -d data/local/iceberg/ 2>> work/dl_sims.log
for i in 1 2 3 4 5; do
  /mnt/disk3/liuyutong_home/miniconda3/envs/kaggle/bin/kaggle datasets download ahmedberatozer/casmi26-glacier -p data/local/glacier -o -q >> work/dl_sims.log 2>&1 && break
  echo "glacier attempt $i failed, sleep 120" >> work/dl_sims.log; sleep 120
done
unzip -oq data/local/glacier/casmi26-glacier.zip -d data/local/glacier/ 2>> work/dl_sims.log
echo "SIMS_DONE $(date)" >> work/dl_sims.log
