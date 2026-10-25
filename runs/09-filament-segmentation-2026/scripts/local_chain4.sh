#!/bin/bash
# chain4 (post strayed-dup rework):
#  1) wait for run2b (1024 s42, output/local-mirror-1024b) [done]
#  2) snapshot its ensemble inputs to output/local-mirror-1024-snap
#     (unblocks gpu_ensemble_chain, session 54775)
#  3) wait for run3 (768 s7, output/local-mirror-3) [done]
#  4) build the 3-way ensemble (768s42 + 1024s42 + 768s7)
#  5) after the next UTC quota rollover: submit ensemble3 (+ raw run3)
#     via kaggle_skill (works only while the GCS route is up; gcs_probe_submit
#     retries these CSVs later if this round misses).
RUN=/mnt/disk3/liuyutong_home/workspace/skill/score-push/runs/09-filament-segmentation-2026
source $RUN/env.sh
cd $RUN
LOG=$RUN/logs/local_chain4.log
SK=$KAGGLE_SKILL/scripts/kaggle_skill.py
PY=$RUN/.venv/bin/python
R1=$RUN/output/local-mirror
R2B=$RUN/output/local-mirror-1024b
R3=$RUN/output/local-mirror-3
SNAP=$RUN/output/local-mirror-1024-snap
log() { echo "[$(date -Is)] $*" >> $LOG; }

log "=== chain4 start ==="
status=timeout
for i in $(seq 1 2160); do   # 3.6 days, 2-min polls
  if grep -q "^\[done\]" $RUN/logs/local_train2b.log 2>/dev/null; then status=done; break; fi
  if grep -q "Traceback" $RUN/logs/local_train2b.log 2>/dev/null; then status=crash; break; fi
  sleep 120
done
log "run2b status=$status"
[ "$status" = done ] || { log "ABORT: run2b did not finish"; exit 1; }
mkdir -p $SNAP
cp -f $R2B/run_report.json $R2B/val_gt.npz $R2B/val_stems.json $SNAP/ 2>/dev/null
cp -f $R2B/val_predictions_*.npz $SNAP/ 2>/dev/null
cp -f $R2B/submission_unet.csv $R2B/submission_mrcnn.csv $R2B/submission_merge.csv $SNAP/ 2>/dev/null
cp -f $R2B/submission.csv $SNAP/submission.csv
log "snapshot done: $(ls $SNAP | wc -l) files"

status=timeout
for i in $(seq 1 2160); do
  if grep -q "^\[done\]" $RUN/logs/local_train3.log 2>/dev/null; then status=done; break; fi
  if grep -q "Traceback" $RUN/logs/local_train3.log 2>/dev/null; then status=crash; break; fi
  sleep 120
done
log "run3 status=$status"
[ "$status" = done ] || { log "ABORT: run3 did not finish"; exit 1; }

$PY $RUN/src/tune_ensemble.py $R1 $SNAP $R3 $R3/ensemble3.csv >> $LOG 2>&1
log "ensemble3 rc=$?"

now_utc=$(date -u +%s); next=$(date -u -d "today 23:59:00" +%s)
[ $now_utc -lt $next ] && sleep $((next - now_utc))
log "quota rollover passed $(date -Is)"

$PY $RUN/src/validate_submission.py $R3/ensemble3.csv >> $LOG 2>&1 && \
python3 $SK submit filament-segmentation-2026 $R3/ensemble3.csv -m "local 3-way ensemble 768s42+1024s42+768s7" --yes >> $LOG 2>&1
log "submit ensemble3 rc=$?"
$PY $RUN/src/validate_submission.py $R3/submission.csv >> $LOG 2>&1 && \
python3 $SK submit filament-segmentation-2026 $R3/submission.csv -m "local unet768 12ep (s7)" --yes >> $LOG 2>&1
log "submit run3 rc=$?"
python3 $SK status filament-segmentation-2026 >> $LOG 2>&1 || true
log "=== chain4 end ==="
exit 0
