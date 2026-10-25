#!/bin/bash
cd /mnt/disk3/liuyutong_home/workspace/skill/score-push/runs/13-enveda-CASMI26-molecule-id-mass-spectra
export HTTP_PROXY=http://172.28.4.16:18999 HTTPS_PROXY=http://172.28.4.16:18999
export http_proxy=http://172.28.4.16:18999 https_proxy=http://172.28.4.16:18999
CPU=work/nb_sub
GPU=work/nb_sub_gpu
LOG=work/push_retry.log
whichfile=work/push_which.txt
cpu_fail=0
try_push() {  # $1=dir  $2=label  $3=out
  kaggle kernels push -p "$1" > "$3" 2>&1
  if grep -qiE "https://www.kaggle.com/code/|kernel pushed|pushed version" "$3"; then
    return 0
  fi
  return 1
}
for i in $(seq 1 900); do
  echo "=== cycle $i $(date +%H:%M:%S) ===" | tee -a $LOG
  if try_push $CPU CPU /tmp/push_cpu.txt; then
    echo "CPU PUSH OK at $(date)" | tee -a $LOG
    echo "CPU" > $whichfile
    cat /tmp/push_cpu.txt | tee -a $LOG
    break
  fi
  tail -2 /tmp/push_cpu.txt | tee -a $LOG
  if grep -qi "Maximum batch" /tmp/push_cpu.txt; then
    cpu_fail=$((cpu_fail+1))
    if [ $cpu_fail -ge 6 ] && [ ! -s $whichfile ]; then
      echo "CPU full (fail=$cpu_fail); trying GPU" | tee -a $LOG
      if try_push $GPU GPU /tmp/push_gpu.txt; then
        echo "GPU PUSH OK at $(date)" | tee -a $LOG
        echo "GPU" > $whichfile
        cat /tmp/push_gpu.txt | tee -a $LOG
        break
      fi
      tail -2 /tmp/push_gpu.txt | tee -a $LOG
    fi
  else
    echo "CPU non-session error; backing off 300s" | tee -a $LOG
    sleep 300; continue
  fi
  sleep 150
done
echo "=== loop ended $(date) which=$(cat $whichfile 2>/dev/null) ===" | tee -a $LOG
