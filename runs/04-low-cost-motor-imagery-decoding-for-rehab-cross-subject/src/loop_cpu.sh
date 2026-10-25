#!/bin/bash
# Persistent loop: CPU DL per-arch (dlcpu2) then CPU seed1337 (dlcpu3), sequential.
RUN=/mnt/disk3/liuyutong_home/workspace/skill/skill-push/runs/04 2>/dev/null
RUN=/mnt/disk3/liuyutong_home/workspace/skill/score-push/runs/04-low-cost-motor-imagery-decoding-for-rehab-cross-subject
SK=/mnt/disk3/liuyutong_home/workspace/skill/score-push/skills/kaggle/scripts/kaggle_skill.py
PY=/mnt/disk3/liuyutong_home/miniconda3/envs/kaggle/bin/python
KCLI=/mnt/disk3/liuyutong_home/miniconda3/envs/kaggle/bin/kaggle
LOG=$RUN/loop_cpu.log

run_one() { # $1=dir $2=slug $3=outdir
  local dir=$1 slug=$2 outdir=$3 pushed=0
  while true; do
    if [ -f $RUN/downloads/$outdir/test_probs.csv ]; then echo "$outdir PRESENT" >> $LOG; return 0; fi
    if [ $pushed -eq 0 ]; then
      echo "=== $outdir-push attempt $(date '+%H:%M:%S')" >> $LOG
      (cd $RUN/$dir && $PY $SK notebook-push --yes . >> $LOG 2>&1)
      if tail -20 $LOG | grep -q "It is running on Kaggle"; then pushed=1; echo "$outdir PUSH OK" >> $LOG; continue; fi
      sleep 300; continue
    fi
    echo "=== $outdir-wait $(date '+%H:%M:%S')" >> $LOG
    $PY $SK notebook-wait $slug >> $LOG 2>&1
    local st=$?
    if [ $st -eq 0 ]; then
      (cd $RUN && $KCLI kernels output $slug -p downloads/$outdir >> $LOG 2>&1)
      if [ -f $RUN/downloads/$outdir/test_probs.csv ]; then echo "$outdir DONE $(date '+%H:%M:%S')" >> $LOG; return 0; fi
      echo "$outdir fetch failed, retrying" >> $LOG; sleep 120; continue
    fi
    echo "$outdir wait rc=$st, re-waiting" >> $LOG
    sleep 60
  done
}

run_one kaggle_nb_dlcpu2 lilac1996/motor-rehab-cross-subj-dl-cpu-arch dlcpu2_out
run_one kaggle_nb_dlcpu3 lilac1996/motor-rehab-cross-subj-dl-cpu-seed2 dlcpu3_out
echo "CPU LOOPS ALL DONE" >> $LOG
