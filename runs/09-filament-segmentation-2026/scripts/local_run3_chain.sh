#!/bin/bash
# chain3: wait for run1 (768 s42) + run2 (1024 s42) to finish, snapshot the 1024
# outputs (the stray duplicate 768 run will later overwrite that dir), train run3
# (768px, FILAMENT_SEED=7, 12ep, bs4) as a foreground child, then build the 3-way
# ensemble and submit it (plus raw run3) after the UTC daily quota rollover.
RUN=/mnt/disk3/liuyutong_home/workspace/skill/score-push/runs/09-filament-segmentation-2026
source $RUN/env.sh
LOG=$RUN/logs/local_chain3.log
SK=$KAGGLE_SKILL/scripts/kaggle_skill.py
PY=$RUN/.venv/bin/python
R1=$RUN/output/local-mirror
R2=$RUN/output/local-mirror-1024
R3=$RUN/output/local-mirror-3
SNAP=$RUN/output/local-mirror-1024-snap

log() { echo "[$(date -Is)] $*" >> $LOG; }

log "=== chain3 start ==="
# 1) wait for run1 done (report exists) and run2 done (report says size 1024)
status=timeout
for i in $(seq 1 2160); do   # up to 3.6 days
  r1done=0; r2done=0
  [ -f $R1/run_report.json ] && r1done=1
  if [ -f $R2/run_report.json ] && grep -q '"size": 1024' $R2/run_report.json 2>/dev/null && [ -f $R2/submission.csv ]; then r2done=1; fi
  if [ $r1done -eq 1 ] && [ $r2done -eq 1 ]; then status=done; break; fi
  sleep 120
done
log "wait status=$status (r1=$r1done r2=$r2done)"
[ "$status" = done ] || { log "ABORT: runs did not finish"; exit 1; }

# 2) snapshot the 1024 ensemble inputs BEFORE the stray 768 duplicate overwrites $R2
mkdir -p $SNAP
cp -f $R2/run_report.json $R2/val_gt.npz $R2/val_stems.json $SNAP/ 2>/dev/null
cp -f $R2/val_predictions_*.npz $SNAP/ 2>/dev/null
cp -f $R2/submission_unet.csv $R2/submission_mrcnn.csv $R2/submission_merge.csv $SNAP/ 2>/dev/null
cp -f $R2/submission.csv $SNAP/submission.csv
log "snapshot done: $(ls $SNAP | wc -l) files"

# 3) train run3 (768px, seed 7) in the foreground
log "launching run3 (768 s7)"
FILAMENT_DATA_ROOT=$RUN/data/official/MAGFiLO_1.0_Kaggle_2026 \
FILAMENT_WORK_DIR=$R3 FILAMENT_SEED=7 \
TORCH_HOME=/tmp/torchhome HF_HOME=/tmp/hfhome OMP_NUM_THREADS=8 \
$PY -u $RUN/src/notebooks/v1-local/train_v1.py >> $RUN/logs/local_train3.log 2>&1
rc=$?
log "run3 rc=$rc"
[ $rc -eq 0 ] || { log "ABORT: run3 failed"; exit 1; }
grep -q "^\[done\]" $RUN/logs/local_train3.log || { log "ABORT: run3 log missing [done]"; exit 1; }

# 4) 3-way ensemble (run1 768s42, snap 1024s42, run3 768s7)
$PY $RUN/src/validate_submission.py $R3/submission.csv >> $LOG 2>&1
log "validate run3 rc=$?"
$PY $RUN/src/tune_ensemble.py $R1 $SNAP $R3 $R3/ensemble3.csv >> $LOG 2>&1
log "ensemble3 rc=$?"

# 5) wait for the next UTC daily quota rollover (00:00 UTC = 08:00 CST)
now_utc=$(date -u +%s)
next=$(date -u -d "today 23:59:00" +%s)
if [ $now_utc -lt $next ]; then
  sleep $((next - now_utc))
fi
log "quota rollover passed $(date -Is)"

# 6) submit 3-way ensemble, then raw run3
$PY $RUN/src/validate_submission.py $R3/ensemble3.csv >> $LOG 2>&1 && \
python3 $SK submit filament-segmentation-2026 $R3/ensemble3.csv -m "local 3-way ensemble 768s42+1024s42+768s7" --yes >> $LOG 2>&1
log "submit ensemble3 rc=$?"
$PY $RUN/src/validate_submission.py $R3/submission.csv >> $LOG 2>&1 && \
python3 $SK submit filament-segmentation-2026 $R3/submission.csv -m "local unet768 seed7" --yes >> $LOG 2>&1
log "submit run3 rc=$?"
python3 $SK status filament-segmentation-2026 >> $LOG 2>&1 || true
log "=== chain3 end ==="
exit 0
