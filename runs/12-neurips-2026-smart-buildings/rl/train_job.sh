#!/bin/bash
# GPU job: train dynamics model on collected transitions, export numpy npz.
set -e
RUN=/mnt/disk3/liuyutong_home/workspace/skill/score-push/runs/12-neurips-2026-smart-buildings
cd "$RUN/rl"
export PYTHONPATH="$RUN/rl"
PY=/mnt/disk3/liuyutong_home/miniconda3/envs/qwen/bin/python
$PY cuda_check.py
$PY dynamics_model.py \
  --data "$RUN/data/collect-v4/*.npz" \
  --out "$RUN/rl/models/dynamics-v4.pt" \
  --epochs 15 --batch 8192 --lr 3e-4
$PY export_model.py \
  --ckpt "$RUN/rl/models/dynamics-v4.pt" \
  --out "$RUN/rl/models/dynamics-v4.npz"
echo "JOB DONE"
