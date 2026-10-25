#!/bin/bash
cd /mnt/disk3/liuyutong_home/workspace/skill/score-push/runs/01-predicting-housing-affordability-in-the-u-s
export PATH=/mnt/disk3/liuyutong_home/miniconda3/envs/kaggle/bin:$PATH
export HTTP_PROXY=http://172.28.4.16:18999
export HTTPS_PROXY=http://172.28.4.16:18999
kaggle datasets download -p external vincentvaseghi/us-cities-housing-market-data --quiet
echo REDFIN_DL_DONE
