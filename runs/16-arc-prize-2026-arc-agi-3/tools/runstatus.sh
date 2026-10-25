#!/bin/bash
# Quick status for run 16: push monitor, kernel sessions, downloads, LB top.
RUN=/mnt/disk3/liuyutong_home/workspace/skill/score-push/runs/16-arc-prize-2026-arc-agi-3
export PATH=/mnt/disk3/liuyutong_home/miniconda3/envs/kaggle/bin:$PATH
cd "$RUN"
echo "== date: $(date '+%m-%d %H:%M:%S')"
echo "-- push_monitor.log (tail 6)"
tail -6 push_monitor.log 2>/dev/null
echo "-- pushed markers:"
ls .pushed_* 2>/dev/null || echo "  (none)"
echo "-- my kernel statuses"
for k in lilac1996/arc-agi-3-duck-harness-run16-v1 lilac1996/arc-agi-3-duck-harness-run16-v2 lilac1996/arc-agi-3-ms2-harness-run16-v1; do
  [ -f "$RUN/$(echo $k | cut -d/ -f2)" ] || kaggle kernels status "$k" 2>&1 | tail -1 | sed "s/^/  $k: /"
done
echo "-- model chunks"
du -sh assets/.chunks 2>/dev/null
ls assets/qwen36-27b-fp8/ 2>/dev/null | head -3
[ -f assets/qwen36-27b-fp8/.model_ok ] && echo "  MODEL OK"
echo "-- wheelhouse"
du -sh assets/wheelhouse-v3 2>/dev/null
tail -2 /tmp/wheelhouse_dl.out 2>/dev/null
echo "-- lb top"
kaggle competitions leaderboard arc-prize-2026-arc-agi-3 -s 2>&1 | head -8
