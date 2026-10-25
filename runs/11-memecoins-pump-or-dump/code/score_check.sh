#!/bin/bash
# After a submission: show its score + timestamped leaderboard snapshot.
SKILL=/mnt/disk3/liuyutong_home/workspace/skill/score-push/skills/kaggle/scripts/kaggle_skill.py
PY=/mnt/disk3/liuyutong_home/miniconda3/envs/kaggle/bin/python
cd "$(dirname "$0")/.."
echo "=== recent submissions ==="
kaggle competitions submissions memecoins-pump-or-dump 2>/dev/null | head -8
echo "=== leaderboard snapshot (rank evidence) ==="
$PY $SKILL leaderboard memecoins-pump-or-dump --top 60 2>&1 | tail -65
