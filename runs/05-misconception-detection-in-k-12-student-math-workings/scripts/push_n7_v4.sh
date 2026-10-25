#!/bin/bash
# v4: rc==0 => accepted (fail-closed); stale-COMPLETE for >30min => re-push.
export HTTP_PROXY=http://172.28.4.16:18999 HTTPS_PROXY=http://172.28.4.16:18999 http_proxy=http://172.28.4.16:18999 https_proxy=http://172.28.4.16:18999
export PATH=/mnt/disk3/liuyutong_home/miniconda3/envs/kaggle/bin:$PATH
RUN=/mnt/disk3/liuyutong_home/workspace/skill/score-push/runs/05-misconception-detection-in-k-12-student-math-workings
cd $RUN
LOG=$RUN/logs/push_n7_v4.log
say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $LOG; }
SLUG=lilac1996/misconception-detection-vlm-lora-v1
TAG="N7_TAG 448px-ep1-v4-short"
START=$(date +%s)

push_once(){
  local out rc
  out=$(timeout 240 kaggle kernels push -p notebooks/n7-push 2>&1); rc=$?
  if [ $rc -eq 0 ]; then
    say "PUSH rc=0 :: $(echo "$out" | tr '\n' ' ' | head -c 200)"
    return 0
  fi
  say "push rc=$rc :: $(echo "$out" | tr '\n' ' ' | tail -c 200)"
  return 1
}

STALE_COUNT=0
while true; do
  if [ $(( $(date +%s) - START )) -gt 172800 ]; then say "GIVEUP 48h"; exit 1; fi
  if push_once; then
    STALE_COUNT=0
    say "push accepted; watching for new version (tag=$TAG)"
    # phase 2: wait for non-stale status
    while true; do
      if [ $(( $(date +%s) - START )) -gt 172800 ]; then say "GIVEUP 48h"; exit 1; fi
      st=$(timeout 60 kaggle kernels status $SLUG 2>&1 | grep -o 'KernelWorkerStatus\.[A-Z]*')
      if [ -z "$st" ]; then sleep 120; continue; fi
      if echo "$st" | grep -q "COMPLETE"; then
        log=$(timeout 300 kaggle kernels logs $SLUG 2>&1)
        if echo "$log" | grep -q "$TAG"; then say "new version already COMPLETE with tag (fast?!)"; break; fi
        STALE_COUNT=$((STALE_COUNT+1))
        say "stale COMPLETE (count $STALE_COUNT)"
        if [ $STALE_COUNT -ge 6 ]; then
          say "still stale after 30min -> re-push (push may not have created a version)"
          STALE_COUNT=0
          if push_once; then say "re-push accepted; keep watching"; continue; fi
          sleep 300
          continue 2   # back to outer push loop
        fi
        sleep 300; continue
      fi
      if echo "$st" | grep -q "RUNNING\|PENDING"; then say "new version $st"; break; fi
      if echo "$st" | grep -q "ERROR\|CANCEL"; then
        say "new version $st; re-push in 5 min"
        sleep 300
        if push_once; then say "re-push accepted"; break; fi
        continue 2
      fi
      sleep 120
    done
    # phase 3: watch to terminal + fetch
    while true; do
      if [ $(( $(date +%s) - START )) -gt 172800 ]; then say "GIVEUP 48h"; exit 1; fi
      st=$(timeout 60 kaggle kernels status $SLUG 2>&1 | grep -o 'KernelWorkerStatus\.[A-Z]*')
      say "status: $st"
      case "$st" in
        *COMPLETE*)
          log=$(timeout 300 kaggle kernels logs $SLUG 2>&1)
          echo "$log" > $RUN/logs/n7_v4_kernel.log
          if ! echo "$log" | grep -q "$TAG"; then say "COMPLETE but tag missing (stale); re-push"; break; fi
          echo "$log" | grep -E "N7_TAG|step |OOM|ABORT|final loss|LoRA train MAP|DATASET" | tail -25 | sed 's/^/    | /' | tee -a $LOG
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
            echo "N7_V4 COMPLETE_TAGGED_FETCHED" >> $LOG
            say "N7_V4 COMPLETE + FETCHED"; break
          else
            say "N7_V4 complete but dataset unavailable"; break
          fi;;
        *ERROR*|*CANCEL*)
          say "N7_V4 $st (terminal); re-push"
          if push_once; then say "re-push accepted; keep watching"; break; fi
          break;;
      esac
      sleep 600
    done
    # after a terminal outcome: if fetched, done; else loop to push again
    if grep -q "N7_V4 COMPLETE + FETCHED" $LOG; then say "PUSH_N7_V4 end (success)"; exit 0; fi
    say "round over; starting a new push round"
    STALE_COUNT=0
  fi
  sleep 1800
done
