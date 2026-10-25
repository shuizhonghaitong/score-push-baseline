#!/bin/bash
# One-shot status of the whole run (read-only, fast).
RUN=/mnt/disk3/liuyutong_home/workspace/skill/score-push/runs/09-filament-segmentation-2026
cd $RUN
echo "== $(date '+%m-%d %H:%M:%S') (local) =="
echo "-- train2b (1024 s42): $(grep -a '\[unet\] epoch' logs/local_train2b.log 2>/dev/null | tail -1) | done:$(grep -ac '^\[done\]' logs/local_train2b.log 2>/dev/null) | crash:$(grep -ac Traceback logs/local_train2b.log 2>/dev/null)"
echo "-- train3  (768 s7) : $(grep -a '\[unet\] epoch' logs/local_train3.log 2>/dev/null | tail -1) | done:$(grep -ac '^\[done\]' logs/local_train3.log 2>/dev/null) | crash:$(grep -ac Traceback logs/local_train3.log 2>/dev/null)"
for t in 4 5; do
  f=logs/local_train$t.log
  [ -f "$f" ] && echo "-- train$t : $(grep -a '\[unet\] epoch' $f 2>/dev/null | tail -1) | done:$(grep -ac '^\[done\]' $f 2>/dev/null)"
done
echo "-- chain4: $(tail -n 2 logs/local_chain4.log 2>/dev/null | tr '\n' ' | ')"
echo "-- chain5: $(tail -n 2 logs/local_chain5.log 2>/dev/null | tr '\n' ' | ')"
echo "-- push_gpu_first2: $(tail -n 2 logs/push_gpu_first2.log 2>/dev/null | tr '\n' ' | ')"
echo "-- unblock: $(grep -a 'PROBE status' logs/unblock_watcher.log 2>/dev/null | tail -1)"
echo "-- gcs_probe: $(tail -n 3 logs/gcs_probe_submit.log 2>/dev/null | tr '\n' ' | ')"
echo "-- nb_submitter: $(tail -n 2 logs/notebook_submitter.log 2>/dev/null | tr '\n' ' | ')"
echo "-- autopilot: $(tail -n 2 logs/autopilot.log 2>/dev/null | tr '\n' ' | ')"
echo "-- snap: $([ -f output/local-mirror-1024-snap/run_report.json ] && echo READY || echo waiting)"
echo "-- v1-gpu: $([ -f output/v1-gpu/run_report.json ] && echo READY || echo waiting)"
echo "-- files submitted: $(wc -l < state/auto/file_submitted.txt 2>/dev/null) | nb submitted: $(wc -l < state/auto/nb_submitted.txt 2>/dev/null)"
