#!/bin/bash
export HTTP_PROXY=http://172.28.4.16:18999 HTTPS_PROXY=http://172.28.4.16:18999 http_proxy=http://172.28.4.16:18999 https_proxy=http://172.28.4.16:18999
export PATH=/mnt/disk3/liuyutong_home/miniconda3/envs/kaggle/bin:$PATH
RUN=/mnt/disk3/liuyutong_home/workspace/skill/score-push/runs/05-misconception-detection-in-k-12-student-math-workings
cd $RUN
LOG=$RUN/logs/n7_lifecycle.log
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a $LOG; }
# ---- Phase 1: push n7 until RUNNING ----
START=$(date +%s)
while true; do
  if [ $(( $(date +%s) - START )) -gt 43200 ]; then say "push timeout 12h"; break; fi
  st=$(timeout 60 kaggle kernels status lilac1996/misconception-detection-vlm-lora-v1 2>&1 | grep -o 'KernelWorkerStatus\.[A-Z]*')
  case "$st" in *RUNNING*|*COMPLETE*) say "n7 $st, stop pushing"; break;; esac
  p=$(timeout 150 kaggle kernels push -p notebooks/n7-push 2>&1)
  if echo "$p" | grep -qi "Maximum batch GPU"; then say "slots full (external), retry"
  elif echo "$p" | grep -qi "push error"; then say "push err :: $(echo "$p"|grep -o 'push error.*'|head -1)"
  else say "N7 PUSHED"; fi
  sleep 60
done
# ---- Phase 2: watch n7 to terminal ----
GOT=0
while true; do
  if [ $(( $(date +%s) - START )) -gt 50400 ]; then say "watch timeout 14h"; break; fi
  st=$(timeout 60 kaggle kernels status lilac1996/misconception-detection-vlm-lora-v1 2>&1 | grep -o 'KernelWorkerStatus\.[A-Z]*')
  case "$st" in
    *COMPLETE*)
      if [ $GOT -eq 0 ]; then
        mkdir -p outputs/ds_lora
        timeout 300 kaggle datasets download lilac1996/miscc-vlm-lora-v1 -p outputs/ds_lora >/tmp/dll 2>&1
        if [ -f outputs/ds_lora/miscc-vlm-lora-v1.zip ]; then
          (cd outputs/ds_lora && unzip -o -q *.zip); GOT=1
          [ -f outputs/ds_lora/vlm_diag_test.csv ] && say "N7 COMPLETE + FETCHED" || say "n7 complete, no csv: $(grep -o '404.*\|created\|error.*' /tmp/dll|head -1)"
        else say "n7 complete, dataset: $(grep -o '404.*\|created\|error.*' /tmp/dll|head -1)"; fi
      fi
      [ $GOT -eq 1 ] && break;;
    *ERROR*|*CANCEL*) say "N7 $st (terminal)"; break;;
    *) say "n7 $st";;
  esac
  sleep 300
done
say "N7_LIFECYCLE end"
