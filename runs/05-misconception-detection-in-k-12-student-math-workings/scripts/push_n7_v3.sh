#!/bin/bash
# Push n7 448px v3, verify the NEW version actually starts (not stale COMPLETE), watch to terminal, fetch.
export HTTP_PROXY=http://172.28.4.16:18999 HTTPS_PROXY=http://172.28.4.16:18999 http_proxy=http://172.28.4.16:18999 https_proxy=http://172.28.4.16:18999
export PATH=/mnt/disk3/liuyutong_home/miniconda3/envs/kaggle/bin:$PATH
RUN=/mnt/disk3/liuyutong_home/workspace/skill/score-push/runs/05-misconception-detection-in-k-12-student-math-workings
cd $RUN
LOG=$RUN/logs/push_n7_v3.log
say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $LOG; }
SLUG=lilac1996/misconception-detection-vlm-lora-v1
TAG="N7_TAG 448px-ep1-v3"
START=$(date +%s)

push_once(){
  out=$(timeout 240 kaggle kernels push -p notebooks/n7-push 2>&1)
  rc=$?
  echo "$out"
  if [ $rc -ne 0 ] && ! echo "$out" | grep -q "429\|Maximum batch GPU\|slots"; then
    say "PUSH rc=$rc (nonstandard): $(echo "$out"|tail -2|tr '\n' ' ')"
  fi
  if echo "$out" | grep -q "429"; then return 429; fi
  if echo "$out" | grep -qi "Maximum batch GPU"; then return 409; fi
  if echo "$out" | grep -qi "error\|denied\|failed"; then return 500; fi
  return 0
}

# ---- phase 1: push until accepted (max 30h) ----
N=0
while true; do
  N=$((N+1))
  if [ $(( $(date +%s) - START )) -gt 108000 ]; then say "GIVEUP push after 30h"; exit 1; fi
  push_once > /tmp/push_out_$N.txt
  rc=$?
  if [ $rc -eq 0 ]; then
    say "PUSH ACCEPTED (attempt $N): $(grep -o 'version [0-9]*\|Pushing.*' /tmp/push_out_$N.txt | head -2 | tr '\n' ' ')"
    break
  elif [ $rc -eq 429 ]; then
    say "attempt $N: 429 rate-limited"
  elif [ $rc -eq 409 ]; then
    say "attempt $N: GPU slots full"
  else
    say "attempt $N: PUSH ERROR :: $(tail -2 /tmp/push_out_$N.txt | tr '\n' ' ')"
  fi
  sleep $(( 300 + (N % 6) * 120 ))
done

# ---- phase 2: wait until status is RUNNING or a fresh COMPLETE with our tag ----
say "watching for new-version start..."
while true; do
  if [ $(( $(date +%s) - START )) -gt 129600 ]; then say "GIVEUP wait-for-start 36h"; exit 1; fi
  st=$(timeout 60 kaggle kernels status $SLUG 2>&1 | grep -o 'KernelWorkerStatus\.[A-Z]*')
  if echo "$st" | grep -q COMPLETE; then
    log=$(timeout 300 kaggle kernels logs $SLUG 2>&1)
    if echo "$log" | grep -q "$TAG"; then say "STALE-COMPLETE-WITH-TAG? (unexpected, checking...)"; fi
    # complete without tag = stale 640px read; keep waiting for new version
    if ! echo "$log" | grep -q "$TAG"; then say "status COMPLETE but no $TAG (stale); waiting for new version"; sleep 300; continue; fi
  fi
  if echo "$st" | grep -q "RUNNING\|PENDING\|COMPLETE"; then
    say "new version is $st"
    break
  fi
  if echo "$st" | grep -q "ERROR\|CANCEL"; then
    say "new version $st before start; will re-push in 10 min"
    sleep 600
    push_once > /tmp/push_re.txt; rc2=$?
    if [ $rc2 -eq 0 ]; then say "RE-PUSH accepted"; break; fi
  fi
  sleep 300
done

# ---- phase 3: watch to terminal + fetch (max 36h) ----
while true; do
  if [ $(( $(date +%s) - START )) -gt 388800 ]; then say "GIVEUP watch 108h"; exit 1; fi
  st=$(timeout 60 kaggle kernels status $SLUG 2>&1 | grep -o 'KernelWorkerStatus\.[A-Z]*')
  say "status: $st"
  case "$st" in
    *COMPLETE*)
      log=$(timeout 300 kaggle kernels logs $SLUG 2>&1)
      echo "$log" > $RUN/logs/n7_v3_kernel.log
      if ! echo "$log" | grep -q "$TAG"; then say "COMPLETE but tag missing; treat as stale, keep waiting"; sleep 300; continue; fi
      echo "$log" | grep -E "step |OOM|ABORT|final loss|LoRA train MAP|DATASET" | tail -25 | sed 's/^/    | /' | tee -a $LOG
      mkdir -p outputs/ds_lora
      ok=0
      for v in 1 2 3 4; do
        timeout 300 kaggle datasets download lilac1996/miscc-vlm-lora-v1 -p outputs/ds_lora >/tmp/dll 2>&1
        if [ -f outputs/ds_lora/miscc-vlm-lora-v1.zip ]; then
          (cd outputs/ds_lora && unzip -o -q *.zip); ok=1; break
        fi
        say "dataset fetch retry $v: $(grep -o '404.*\|error.*' /tmp/dll|head -1)"; sleep 300
      done
      if [ $ok -eq 1 ]; then
        echo "N7_V3 COMPLETE_TAGGED_FETCHED" >> $LOG
        say "N7_V3 COMPLETE + FETCHED"; break
      else
        say "N7_V3 complete but dataset unavailable"; break
      fi;;
    *ERROR*|*CANCEL*)
      say "N7_V3 $st (terminal)"; break;;
  esac
  sleep 600
done
say "PUSH_N7_V3 end"
