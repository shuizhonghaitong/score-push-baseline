#!/bin/bash
export HTTP_PROXY=http://172.28.4.16:18999 HTTPS_PROXY=http://172.28.4.16:18999 http_proxy=http://172.28.4.16:18999 https_proxy=http://172.28.4.16:18999
export PATH=/mnt/disk3/liuyutong_home/miniconda3/envs/kaggle/bin:$PATH
RUN=/mnt/disk3/liuyutong_home/workspace/skill/score-push/runs/05-misconception-detection-in-k-12-student-math-workings
cd $RUN
LOG=$RUN/logs/auto_loop.log
N7PUSHED=0; N3B2=0; N5=0
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a $LOG; }
say "AUTO_LOOP start"
START=$(date +%s)
while true; do
  if [ $(( $(date +%s) - START )) -gt 28800 ]; then say "timeout 8h, exit"; break; fi
  # --- n7: push only if not already created (404) ---
  if [ $N7PUSHED -eq 0 ]; then
    s=$(timeout 60 kaggle kernels status lilac1996/misconception-detection-vlm-lora-v1 2>&1)
    if echo "$s" | grep -q "404\|Not Found"; then
      p=$(timeout 150 kaggle kernels push -p notebooks/n7-push 2>&1)
      if echo "$p" | grep -qi "Maximum batch GPU"; then
        say "n7: slots full, wait"
      elif echo "$p" | grep -qi "push error"; then
        say "n7: push err :: $(echo "$p"|grep -o 'push error.*'|head -1)"
      else
        N7PUSHED=1; say "N7 PUSHED"
      fi
    else
      N7PUSHED=1; say "n7 already exists: $(echo "$s"|grep -o 'KernelWorkerStatus\.[A-Z]*')"
    fi
  fi
  # --- n3b2: download if missing ---
  if [ $N3B2 -eq 0 ] && [ ! -f outputs/ds_vlm_hr/vlm_answer_test.csv ]; then
    timeout 200 kaggle datasets download lilac1996/miscc-vlm-ans-hr-v1 -p outputs/ds_vlm_hr >/tmp/dl3 2>&1
    if [ -f outputs/ds_vlm_hr/miscc-vlm-ans-hr-v1.zip ]; then
      (cd outputs/ds_vlm_hr && unzip -o -q *.zip)
      [ -f outputs/ds_vlm_hr/vlm_answer_test.csv ] && N3B2=1 && say "N3B2 FETCHED" || say "n3b2: zip no csv"
    else
      say "n3b2: $(grep -o '404.*\|error.*' /tmp/dl3 | head -1)"
    fi
  fi
  # --- n5: status + download ---
  if [ $N5 -eq 0 ]; then
    s5=$(timeout 60 kaggle kernels status lilac1996/misconception-detection-vlm-diagnosis-v1 2>&1 | grep -o 'KernelWorkerStatus\.[A-Z]*')
    say "n5=$s5"
    if [ "$s5" = "KernelWorkerStatus.COMPLETE" ] && [ ! -f outputs/ds_diag/vlm_diag_test.csv ]; then
      timeout 300 kaggle datasets download lilac1996/miscc-vlm-diag-v1 -p outputs/ds_diag >/tmp/dl5 2>&1
      if [ -f outputs/ds_diag/miscc-vlm-diag-v1.zip ]; then
        (cd outputs/ds_diag && unzip -o -q *.zip)
        [ -f outputs/ds_diag/vlm_diag_test.csv ] && N5=1 && say "N5 FETCHED" || say "n5: zip no csv"
      else
        say "n5: dl $(grep -o '404.*\|error.*' /tmp/dl5 | head -1)"
      fi
    fi
  fi
  if [ $N7PUSHED -eq 1 ] && [ $N5 -eq 1 ] && [ $N3B2 -eq 1 ]; then say "ALL DONE"; break; fi
  sleep 60
done
say "AUTO_LOOP end"
