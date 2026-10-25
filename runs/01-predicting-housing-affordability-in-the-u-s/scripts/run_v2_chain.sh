#!/bin/bash
cd /mnt/disk3/liuyutong_home/workspace/skill/score-push/runs/01-predicting-housing-affordability-in-the-u-s
export PATH=/mnt/disk3/liuyutong_home/miniconda3/envs/kaggle/bin:$PATH
for s in 0 1 2 3 4; do for f in 0 1 2 3 4; do
  python -u -m scripts.fold_worker $s $f || echo "FAILED $s $f"
done; done
python -u -m scripts.assemble_v2
echo ALL_DONE
