#!/bin/bash
# Wait until the 1024px run writes run_report.json (size==1024) into output/local-mirror-1024,
# i.e. the 1024 run completed (robust against the stray 768 duplicate overwriting the same dir).
RUN=/mnt/disk3/liuyutong_home/workspace/skill/score-push/runs/09-filament-segmentation-2026
source $RUN/env.sh
LOG=$RUN/logs/local_chain2b.log
WORK=$RUN/output/local-mirror-1024
SK=$KAGGLE_SKILL/scripts/kaggle_skill.py
echo "=== local_first_score2b start $(date -Is) ===" >> $LOG
status=timeout
for i in $(seq 1 1440); do
  rep=$WORK/run_report.json
  if [ -f "$rep" ] && grep -q '"size": 1024' "$rep" 2>/dev/null && [ -f $WORK/submission.csv ]; then
    status=done; break
  fi
  sleep 120
done
echo "1024 run status=$status at $(date -Is)" >> $LOG
[ "$status" = done ] || { echo "ABORT: 1024 run did not finish" >> $LOG; exit 1; }
# 768 run must be done too for the ensemble (its dir output/local-mirror)
for i in $(seq 1 1440); do
  if [ -f $RUN/output/local-mirror/run_report.json ]; then break; fi
  sleep 120
done
$RUN/.venv/bin/python $RUN/src/validate_submission.py $WORK/submission.csv >> $LOG 2>&1
$RUN/.venv/bin/python $RUN/src/tune_ensemble.py $RUN/output/local-mirror $WORK $WORK/ensemble_local.csv >> $LOG 2>&1
echo "ensemble rc=$?" >> $LOG
if [ -f $WORK/ensemble_local.csv ]; then
  $RUN/.venv/bin/python $RUN/src/validate_submission.py $WORK/ensemble_local.csv >> $LOG 2>&1 && \
  python3 $SK submit filament-segmentation-2026 $WORK/ensemble_local.csv -m "local unet768+1024 ensemble" --yes >> $LOG 2>&1
  echo "submit ensemble rc=$? at $(date -Is)" >> $LOG
fi
$RUN/.venv/bin/python $RUN/src/validate_submission.py $WORK/submission.csv >> $LOG 2>&1 && \
  python3 $SK submit filament-segmentation-2026 $WORK/submission.csv -m "local unet1024 12ep" --yes >> $LOG 2>&1
echo "submit 1024 rc=$? at $(date -Is)" >> $LOG
python3 $SK status filament-segmentation-2026 >> $LOG 2>&1 || true
echo "=== local_first_score2b end $(date -Is) ===" >> $LOG
