#!/bin/bash
cd /mnt/disk3/liuyutong_home/workspace/skill/score-push/runs/20-squares-in-squares
QP=/mnt/disk3/liuyutong_home/miniconda3/envs/qwen/bin/python
targets=(367 491 411 637 588 370 587 634 799 446 535 330 333 639 448 410 495 492 689 798 366 582 738 910 412 797)
for w in 0 1 2 3; do
  list=""
  for i in "${!targets[@]}"; do
    if [ $((i % 4)) -eq $w ]; then list="$list ${targets[$i]}"; fi
  done
  echo "worker $w: $list"
  ( for n in $list; do
      echo "start $n $(date -u +%H:%M:%S)" >> logs/seam_results/c2_worker$w.log
      OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 timeout 2400 $QP -u scripts/seam_campaign2.py $n > logs/seam_results/c2_$n.log 2>&1
      echo "end $n rc=$? $(date -u +%H:%M:%S)" >> logs/seam_results/c2_worker$w.log
    done ) &
done
wait
echo ALL-DONE
