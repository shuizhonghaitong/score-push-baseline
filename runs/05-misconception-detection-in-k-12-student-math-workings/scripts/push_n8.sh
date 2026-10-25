#!/bin/bash
export HTTP_PROXY=http://172.28.4.16:18999 HTTPS_PROXY=http://172.28.4.16:18999 http_proxy=http://172.28.4.16:18999 https_proxy=http://172.28.4.16:18999
export PATH=/mnt/disk3/liuyutong_home/miniconda3/envs/kaggle/bin:$PATH
RUN=/mnt/disk3/liuyutong_home/workspace/skill/score-push/runs/05-misconception-detection-in-k-12-student-math-workings
cd $RUN
LOG=$RUN/logs/push_n8.log
say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $LOG; }
SLUG=lilac1996/misconception-detection-trans-knn-v1
START=$(date +%s)
# delay 6 min so n7 gets queued first
sleep 360
push_once(){
  local out rc
  out=$(timeout 240 kaggle kernels push -p notebooks/n8-transknn 2>&1); rc=$?
  if [ $rc -eq 0 ]; then say "N8 PUSH rc=0 :: $(echo "$out" | tr '\n' ' ' | head -c 200)"; return 0; fi
  say "n8 push rc=$rc :: $(echo "$out" | tr '\n' ' ' | tail -c 200)"
  return 1
}
while true; do
  if [ $(( $(date +%s) - START )) -gt 172800 ]; then say "N8 GIVEUP 48h"; exit 1; fi
  if push_once; then
    say "n8 accepted; watching"
    while true; do
      if [ $(( $(date +%s) - START )) -gt 172800 ]; then say "N8 GIVEUP 48h"; exit 1; fi
      st=$(timeout 60 kaggle kernels status $SLUG 2>&1 | grep -o 'KernelWorkerStatus\.[A-Z]*')
      say "n8 status: $st"
      case "$st" in
        *COMPLETE*)
          log=$(timeout 300 kaggle kernels logs $SLUG 2>&1); echo "$log" > $RUN/logs/n8_kernel.log
          echo "$log" | grep -E "N8_TAG|oof MAP|DATASET|Error|error" | tail -10 | sed 's/^/    | /' | tee -a $LOG
          mkdir -p outputs/ds_kntrans
          for v in 1 2 3 4 5 6; do
            timeout 300 kaggle datasets download lilac1996/miscc-knn-trans-v1 -p outputs/ds_kntrans >/tmp/dll8 2>&1
            [ -f outputs/ds_kntrans/miscc-knn-trans-v1.zip ] && (cd outputs/ds_kntrans && unzip -o -q *.zip) && break
            say "n8 dataset retry $v"; sleep 300
          done
          if [ -f outputs/ds_kntrans/knn_trans_probs.npz ]; then say "N8 COMPLETE + FETCHED"; exit 0; fi
          say "N8 complete but dataset missing"; break;;
        *ERROR*|*CANCEL*) say "N8 $st (terminal)"; break;;
      esac
      sleep 300
    done
  fi
  sleep 600
done
