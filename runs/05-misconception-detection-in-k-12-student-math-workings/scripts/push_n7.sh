#!/bin/bash
export HTTP_PROXY=http://172.28.4.16:18999 HTTPS_PROXY=http://172.28.4.16:18999 http_proxy=http://172.28.4.16:18999 https_proxy=http://172.28.4.16:18999
export PATH=/mnt/disk3/liuyutong_home/miniconda3/envs/kaggle/bin:$PATH
RUN=/mnt/disk3/liuyutong_home/workspace/skill/score-push/runs/05-misconception-detection-in-k-12-student-math-workings
cd $RUN
LOG=$RUN/logs/push_n7.log
say(){ echo "[$(date +%H:%M:%S)] $*" | tee -a $LOG; }
say "PUSH_N7 start"
START=$(date +%s)
while true; do
  if [ $(( $(date +%s) - START )) -gt 43200 ]; then say "timeout 12h"; break; fi
  s=$(timeout 60 kaggle kernels status lilac1996/misconception-detection-vlm-lora-v1 2>&1)
  st=$(echo "$s" | grep -o 'KernelWorkerStatus\.[A-Z]*')
  case "$st" in
    *RUNNING*|*COMPLETE*) say "n7 already $st, done"; break;;
  esac
  # n7 not running (404 or ERROR): try push
  p=$(timeout 150 kaggle kernels push -p notebooks/n7-push 2>&1)
  if echo "$p" | grep -qi "Maximum batch GPU"; then
    say "slots full (external), retry"
  elif echo "$p" | grep -qi "push error"; then
    say "push err :: $(echo "$p"|grep -o 'push error.*'|head -1)"
  else
    say "N7 PUSHED"; sleep 20; st2=$(timeout 60 kaggle kernels status lilac1996/misconception-detection-vlm-lora-v1 2>&1 | grep -o 'KernelWorkerStatus\.[A-Z]*'); say "n7 now $st2"; break
  fi
  sleep 60
done
say "PUSH_N7 end"
