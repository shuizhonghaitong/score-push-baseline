#!/bin/bash
RUN=/mnt/disk3/liuyutong_home/workspace/skill/score-push/runs/12-neurips-2026-smart-buildings
export PATH=/mnt/disk3/liuyutong_home/miniconda3/envs/kaggle/bin:$PATH
export HTTP_PROXY=http://172.28.4.16:18999 HTTPS_PROXY=http://172.28.4.16:18999
export http_proxy=http://172.28.4.16:18999 https_proxy=http://172.28.4.16:18999
cd "$RUN"
echo "=== $(date '+%F %T') ==="
kaggle kernels status lilac1996/smart-buildings-v4tpu 2>/dev/null | grep -o "KernelWorkerStatus\.[A-Z]*" | head -1
echo "-- auto_submit:"; tail -n 2 results/auto_submit_v4tpu.log 2>/dev/null
echo "-- eval_queue:"; tail -n 2 downloads/eval_queue.log 2>/dev/null
echo "-- files:"; ls -la results/eval_v4g_winter.json results/eval_v5_full.json results/eval_v7_winter.json results/eval_v7.json 2>/dev/null | awk '{print $5, $6, $7, $8, $9}'
