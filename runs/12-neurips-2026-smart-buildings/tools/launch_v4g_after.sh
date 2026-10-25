#!/bin/bash
# Run V4g winter eval once BOTH winter-compare instances have finished.
RUN=/mnt/disk3/liuyutong_home/workspace/skill/score-push/runs/12-neurips-2026-smart-buildings
cd "$RUN"
while [ "$(grep -c 'ALL DONE' downloads/winter_compare.log 2>/dev/null)" -lt 2 ]; do sleep 120; done
echo "[$(date '+%F %T')] both compares done; launching V4g winter" > downloads/v4g_winter.log
WEATHERS="fairbanks_1 fairbanks_3 fairbanks_8 minneapolis_1 minneapolis_2 denver_1 atlanta_4 singapore_8 phoenix_5"
OMP_NUM_THREADS=1 MPLCONFIGDIR=/tmp .venv10/bin/python tools/eval_local.py \
  --policy-module policies/v4_variants.py --policy-class V4gPolicy --steps 8640 --workers 8 \
  --weathers $WEATHERS --diagnose --out results/eval_v4g_winter.json \
  > results/eval_v4g_winter.log 2>&1
echo "[$(date '+%F %T')] v4g done rc=$?" >> downloads/v4g_winter.log
