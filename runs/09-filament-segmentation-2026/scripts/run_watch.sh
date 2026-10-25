#!/bin/bash
# Long-running 30-min watcher: status snapshot + anomaly flags.
RUN=/mnt/disk3/liuyutong_home/workspace/skill/score-push/runs/09-filament-segmentation-2026
source $RUN/env.sh
cd $RUN
LOG=$RUN/logs/run_watch2.log
for i in $(seq 1 2880); do
  echo "===== $(date -Is) cycle $i =====" >> $LOG
  bash $RUN/scripts/status_brief.sh >> $LOG 2>&1
  # anomaly flags
  for f in local_train2b local_train3 local_train4 local_train5; do
    if grep -q Traceback $RUN/logs/$f.log 2>/dev/null; then
      echo "!!! CRASH in $f.log:" >> $LOG
      grep -a -A8 Traceback $RUN/logs/$f.log | tail -20 >> $LOG
    fi
  done
  if grep -q "window open" $RUN/logs/push_gpu_first2.log 2>/dev/null; then
    echo "!!! SaveKernel WINDOW OPENED (push in progress)" >> $LOG
  fi
  if grep -q "GCS route REACHED" $RUN/logs/gcs_probe_submit.log 2>/dev/null; then
    echo "!!! GCS ROUTE RECOVERED (file submits starting)" >> $LOG
    cp $RUN/state/auto/file_submitted.txt $RUN/state/auto/file_submitted.snapshot.$i 2>/dev/null
  fi
  if grep -q "NB SUBMITTED" $RUN/logs/notebook_submitter.log 2>/dev/null; then
    echo "!!! NOTEBOOK SUBMISSION LANDED" >> $LOG
  fi
  sleep 1800
done
