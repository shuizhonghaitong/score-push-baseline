#!/bin/bash
# Keepalive wrapper: respawn orch2b loop if it exits (crash protection).
cd /mnt/disk3/liuyutong_home/workspace/skill/score-push/runs/16-arc-prize-2026-arc-agi-3
export PATH=/mnt/disk3/liuyutong_home/miniconda3/envs/kaggle/bin:$PATH
while true; do
  python3 tools/orch2b.py loop >> orch2b.log 2>&1
  echo "[$(date '+%m-%d %H:%M:%S')] keepalive: orch2b exited rc=$?; respawning in 60s" >> orch2b.log
  sleep 60
done
