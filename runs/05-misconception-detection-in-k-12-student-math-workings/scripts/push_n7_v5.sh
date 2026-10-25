#!/bin/bash
# v5: like v4 but OOM-aware: on ERROR-with-OOM-ABORT, switch to 384px once, then give up.
export HTTP_PROXY=http://172.28.4.16:18999 HTTPS_PROXY=http://172.28.4.16:18999 http_proxy=http://172.28.4.16:18999 https_proxy=http://172.28.4.16:18999
export PATH=/mnt/disk3/liuyutong_home/miniconda3/envs/kaggle/bin:$PATH
RUN=/mnt/disk3/liuyutong_home/workspace/skill/score-push/runs/05-misconception-detection-in-k-12-student-math-workings
cd $RUN
LOG=$RUN/logs/push_n7_v5.log
say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $LOG $RUN/logs/push_n7_v4.log; }
SLUG=lilac1996/misconception-detection-vlm-lora-v1
START=$(date +%s)
RES="448"
TAG="N7_TAG 448px-ep1-v4-short"

push_once(){
  local out rc
  out=$(timeout 240 kaggle kernels push -p notebooks/n7-push 2>&1); rc=$?
  if [ $rc -eq 0 ]; then say "PUSH rc=0 :: $(echo "$out" | tr '\n' ' ' | head -c 160)"; return 0; fi
  say "push rc=$rc :: $(echo "$out" | tr '\n' ' ' | tail -c 160)"
  return 1
}
switch_384(){
  if [ "$RES" = "384" ]; then return 1; fi
  cp notebooks/n7-push-384/main.py notebooks/n7-push/main.py
  RES="384"; TAG="N7_TAG 384px-ep1-v4-short"
  say "SWITCHED to 384px (OOM on 448px); new tag $TAG"
  return 0
}
stale_recover(){
  # status shows ERROR or a stale COMPLETE: re-push; if OOM, downgrade once
  local log
  log=$(timeout 300 kaggle kernels logs $SLUG 2>&1)
  if echo "$log" | grep -q "$TAG"; then
    if echo "$log" | grep -q "ABORT.*OOM\|consecutive OOM"; then
      if switch_384; then say "re-pushing at 384px"; push_once; return; fi
      say "GIVEUP: 384px also OOMed; no more downgrades. Manual action needed."; exit 2
    fi
  fi
  say "re-pushing (same res $RES)"
  push_once
}

STALE=0
while true; do
  if [ $(( $(date +%s) - START )) -gt 259200 ]; then say "GIVEUP 72h"; exit 1; fi
  if push_once; then
    STALE=0
    say "push accepted ($RES px); watching (tag=$TAG)"
    while true; do
      if [ $(( $(date +%s) - START )) -gt 259200 ]; then say "GIVEUP 72h"; exit 1; fi
      st=$(timeout 60 kaggle kernels status $SLUG 2>&1 | grep -o 'KernelWorkerStatus\.[A-Z]*')
      if [ -z "$st" ]; then sleep 120; continue; fi
      if echo "$st" | grep -q "COMPLETE"; then
        log=$(timeout 300 kaggle kernels logs $SLUG 2>&1)
        if echo "$log" | grep -q "$TAG"; then say "new version already COMPLETE with tag"; break; fi
        STALE=$((STALE+1)); say "stale COMPLETE ($STALE)"
        if [ $STALE -ge 6 ]; then STALE=0; stale_recover; continue 2; fi
        sleep 300; continue
      fi
      if echo "$st" | grep -q "RUNNING\|PENDING"; then say "new version $st"; break; fi
      if echo "$st" | grep -q "ERROR\|CANCEL"; then
        say "version $st; recovering"
        stale_recover
        continue 2
      fi
      sleep 120
    done
    while true; do
      if [ $(( $(date +%s) - START )) -gt 259200 ]; then say "GIVEUP 72h"; exit 1; fi
      st=$(timeout 60 kaggle kernels status $SLUG 2>&1 | grep -o 'KernelWorkerStatus\.[A-Z]*')
      say "status: $st"
      case "$st" in
        *COMPLETE*)
          log=$(timeout 300 kaggle kernels logs $SLUG 2>&1)
          echo "$log" > $RUN/logs/n7_v5_kernel.log
          if ! echo "$log" | grep -q "$TAG"; then say "COMPLETE but tag missing (stale); re-push"; break; fi
          echo "$log" | grep -E "N7_TAG|step |OOM|ABORT|final loss|LoRA train MAP|DATASET|BUDGET" | tail -25 | sed 's/^/    | /' | tee -a $LOG
          mkdir -p outputs/ds_lora
          ok=0
          for v in 1 2 3 4 5 6; do
            timeout 300 kaggle datasets download lilac1996/miscc-vlm-lora-v1 -p outputs/ds_lora >/tmp/dll 2>&1
            if [ -f outputs/ds_lora/miscc-vlm-lora-v1.zip ]; then
              (cd outputs/ds_lora && unzip -o -q *.zip); ok=1; break
            fi
            say "dataset fetch retry $v: $(grep -o '404.*\|error.*' /tmp/dll|head -1)"; sleep 300
          done
          if [ $ok -eq 1 ]; then
            echo "N7_V5 COMPLETE_TAGGED_FETCHED" >> $LOG
            say "N7_V5 COMPLETE + FETCHED"; break
          else
            say "N7_V5 complete but dataset unavailable"; break
          fi;;
        *ERROR*|*CANCEL*)
          say "terminal $st; recovering"
          stale_recover
          break;;
      esac
      sleep 600
    done
    if grep -q "N7_V5 COMPLETE_TAGGED_FETCHED" $LOG; then say "PUSH_N7_V5 end (success)"; exit 0; fi
    say "round over; new push round"
  fi
  sleep 1800
done
