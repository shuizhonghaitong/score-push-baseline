#!/bin/bash
# Unattended: wait for local 1024px U-Net training, validate, ensemble with local-mirror, submit best.
RUN=/mnt/disk3/liuyutong_home/workspace/skill/score-push/runs/09-filament-segmentation-2026
source $RUN/env.sh
LOG=$RUN/logs/local_chain2.log
TRAIN_LOG=$RUN/logs/local_train2.log
WORK=$RUN/output/local-mirror-1024
SK=$KAGGLE_SKILL/scripts/kaggle_skill.py
echo "=== local_first_score2 start $(date -Is) ===" >> $LOG
status=timeout
for i in $(seq 1 1152); do
  if grep -q "^\[done\]" $TRAIN_LOG 2>/dev/null; then status=done; break; fi
  if grep -q "Traceback" $TRAIN_LOG 2>/dev/null; then status=failed; break; fi
  sleep 120
done
echo "train status=$status at $(date -Is)" >> $LOG
[ "$status" = done ] || { echo "ABORT: training did not finish cleanly" >> $LOG; exit 1; }
# also wait for the 768 run (needed for the ensemble)
for i in $(seq 1 1152); do
  if grep -q "^\[done\]" $RUN/logs/local_train.log 2>/dev/null; then break; fi
  if grep -q "Traceback" $RUN/logs/local_train.log 2>/dev/null; then break; fi
  sleep 120
done
echo "768 run state: $(tail -1 $RUN/logs/local_train.log | cut -c1-100)" >> $LOG
$RUN/.venv/bin/python $RUN/src/validate_submission.py $WORK/submission.csv >> $LOG 2>&1 || true
# ensemble the two local runs
$RUN/.venv/bin/python $RUN/src/tune_ensemble.py $RUN/output/local-mirror $WORK $WORK/ensemble_local.csv >> $LOG 2>&1
echo "ensemble rc=$?" >> $LOG
if [ -f $WORK/ensemble_local.csv ]; then
  $RUN/.venv/bin/python $RUN/src/validate_submission.py $WORK/ensemble_local.csv >> $LOG 2>&1 && \
  python3 $SK submit filament-segmentation-2026 $WORK/ensemble_local.csv -m "local unet768+1024 ensemble" --yes >> $LOG 2>&1
  echo "submit ensemble rc=$? at $(date -Is)" >> $LOG
fi
# always also submit the raw 1024 model if it validates
$RUN/.venv/bin/python $RUN/src/validate_submission.py $WORK/submission.csv >> $LOG 2>&1 && \
  python3 $SK submit filament-segmentation-2026 $WORK/submission.csv -m "local unet1024 12ep" --yes >> $LOG 2>&1
echo "submit 1024 rc=$? at $(date -Is)" >> $LOG
python3 $SK status filament-segmentation-2026 >> $LOG 2>&1 || true
echo "=== local_first_score2 end $(date -Is) ===" >> $LOG
