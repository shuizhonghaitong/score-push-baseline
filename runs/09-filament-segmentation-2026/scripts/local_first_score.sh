#!/bin/bash
# Unattended: wait for local U-Net training to finish, validate, submit (file upload).
RUN=/mnt/disk3/liuyutong_home/workspace/skill/score-push/runs/09-filament-segmentation-2026
source $RUN/env.sh
LOG=$RUN/logs/local_chain.log
TRAIN_LOG=$RUN/logs/local_train.log
WORK=$RUN/output/local-mirror
SK=$KAGGLE_SKILL/scripts/kaggle_skill.py
echo "=== local_first_score start $(date -Is) ===" >> $LOG

status=timeout
for i in $(seq 1 1152); do   # up to 8 days
  if grep -q "^\[done\]" $TRAIN_LOG 2>/dev/null; then status=done; break; fi
  if grep -q "Traceback" $TRAIN_LOG 2>/dev/null; then status=failed; break; fi
  sleep 120
done
echo "train status=$status at $(date -Is)" >> $LOG
[ "$status" = done ] || { echo "ABORT: training did not finish cleanly" >> $LOG; exit 1; }

$RUN/.venv/bin/python $RUN/src/validate_submission.py $WORK/submission.csv >> $LOG 2>&1
vc=$?
echo "validate rc=$vc" >> $LOG
[ $vc -ne 0 ] && { echo "ABORT: validation failed" >> $LOG; exit 1; }

cat $WORK/run_report.json >> $LOG
echo "--- dry run ---" >> $LOG
python3 $SK submit filament-segmentation-2026 $WORK/submission.csv -m "v1-local unet768 12ep" >> $LOG 2>&1
echo "--- real submit ---" >> $LOG
python3 $SK submit filament-segmentation-2026 $WORK/submission.csv -m "v1-local unet768 12ep" --yes >> $LOG 2>&1
sc=$?
echo "submit rc=$sc at $(date -Is)" >> $LOG
if [ $sc -eq 0 ]; then
  python3 $SK watch filament-segmentation-2026 --timeout 3600 >> $LOG 2>&1 || true
  python3 $SK status filament-segmentation-2026 >> $LOG 2>&1 || true
fi
echo "=== local_first_score end $(date -Is) ===" >> $LOG
exit $sc
