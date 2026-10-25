#!/bin/bash
# Persistent loop: push GPU DL seed1337 until success, wait, fetch output.
RUN=/mnt/disk3/liuyutong_home/workspace/skill/score-push/runs/04-low-cost-motor-imagery-decoding-for-rehab-cross-subject
SK=/mnt/disk3/liuyutong_home/workspace/skill/score-push/skills/kaggle/scripts/kaggle_skill.py
PY=/mnt/disk3/liuyutong_home/miniconda3/envs/kaggle/bin/python
SLUG=lilac1996/motor-rehab-cross-subj-dl-seed2
LOG=$RUN/loop_gpu2.log
pushed=0
while true; do
  if [ -f $RUN/downloads/dl2_out/test_probs.csv ]; then echo "DL2 OUTPUT ALREADY PRESENT" >> $LOG; break; fi
  if [ $pushed -eq 0 ]; then
    echo "=== gpu2-push attempt $(date '+%H:%M:%S')" >> $LOG
    (cd $RUN/kaggle_nb_dl2 && $PY $SK notebook-push --yes . >> $LOG 2>&1)
    if tail -20 $LOG | grep -q "It is running on Kaggle"; then pushed=1; echo "GPU2 PUSH OK" >> $LOG; continue; fi
    sleep 240; continue
  fi
  echo "=== gpu2-wait $(date '+%H:%M:%S')" >> $LOG
  $PY $SK notebook-wait $SLUG >> $LOG 2>&1
  st=$?
  if [ $st -eq 0 ]; then
    (cd $RUN && /mnt/disk3/liuyutong_home/miniconda3/envs/kaggle/bin/kaggle kernels output $SLUG -p downloads/dl2_out >> $LOG 2>&1)
    if [ -f $RUN/downloads/dl2_out/test_probs.csv ]; then echo "GPU2 DONE $(date '+%H:%M:%S')" >> $LOG; break; fi
    echo "GPU2 fetch failed, retrying" >> $LOG; sleep 120; continue
  fi
  echo "gpu2 wait rc=$st (timeout or error), re-waiting" >> $LOG
  sleep 60
done
