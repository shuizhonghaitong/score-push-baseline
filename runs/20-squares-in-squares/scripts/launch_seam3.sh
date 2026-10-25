#!/bin/bash
cd /mnt/disk3/liuyutong_home/workspace/skill/score-push/runs/20-squares-in-squares
QP=/mnt/disk3/liuyutong_home/miniconda3/envs/qwen/bin/python
i=0
for n in $(cat logs/seam_targets_dariush.txt); do
  w=$((i % 4))
  echo "$n" >> logs/seam_results/v3list_$w.txt
  i=$((i+1))
done
for w in 0 1 2 3; do
  ( while read -r n; do
      [ -z "$n" ] && continue
      [ -f logs/seam_results/3_$n.json ] && continue
      OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 timeout 1500 $QP -u scripts/seam_campaign3.py $n > logs/seam_results/c3_$n.log 2>&1
    done < logs/seam_results/v3list_$w.txt
    echo "worker $w done $(date -u +%H:%M:%S)" >> logs/seam_results/c3_progress.log ) &
done
wait
echo ALL-DONE-V3
