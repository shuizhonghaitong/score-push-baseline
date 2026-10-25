#!/bin/bash
# Queue local A/B jobs (one per GPU) for v1 vs v2 on the devset.
set -e
cd /mnt/disk3/liuyutong_home/workspace/skill/score-push/runs/14-gemma-4-developer-agent
LT=/mnt/disk3/liuyutong_home/workspace/skill/score-push/tools/local_training.py
PY=/mnt/disk3/liuyutong_home/miniconda3/envs/kaggle/bin/python
DEVSET_FILE=${1:-results/devset_kaggle.txt}
DEVSET=$(tr -d ' \n' < "$DEVSET_FILE")
echo "devset: $(echo $DEVSET | tr ',' '\n' | wc -l) tasks"
J1=$($PY $LT submit --run-dir . --memory-gib 64 --python venv-eval/bin/python scripts/job_entry.py --tasks "$DEVSET" --results results/ab_v1 --agent-dir agent)
J2=$($PY $LT submit --run-dir . --memory-gib 64 --python venv-eval/bin/python scripts/job_entry.py --tasks "$DEVSET" --results results/ab_v2 --agent-dir agent-v2)
echo "queued: $J1 $J2"
