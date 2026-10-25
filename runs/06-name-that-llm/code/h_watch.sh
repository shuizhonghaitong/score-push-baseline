#!/bin/bash
# H watcher (agent-2): push notebookH (0.568 BERT-trio fork) after G starts,
# wait terminal, harvest + submit. 1xT4, ~1-2h run, queues behind G if concurrency cap.
cd /mnt/disk3/liuyutong_home/workspace/skill/score-push/runs/06-name-that-llm || exit 1
source env.sh
S=/mnt/disk3/liuyutong_home/workspace/skill/skill/score-push/skills/kaggle/scripts/kaggle_skill.py
PY=./venv/bin/python
LOG=work/h_watch.log
SLUG=lilac1996/ntl-run06-bert3-h
T1630=$(date -d "2026-10-23 16:30" +%s)
log() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
kstate() { timeout 90 kaggle kernels status "$1" 2>&1 | grep -o 'KernelWorkerStatus\.[A-Z_]*' | tail -1; }

log "=== h_watch started (pid $$) ==="
# gate: G push logged by v2 (or 16:30 fallback)
while true; do
  if grep -q "PUSH notebookG" work/fe_b_early2.log 2>/dev/null; then log "gate: G push seen"; break; fi
  if [ $(date +%s) -gt $T1630 ]; then log "gate: 16:30 fallback"; break; fi
  sleep 300
done
sleep 300
# push with 429 retry
pushed=""
for i in $(seq 1 40); do
  out=$(timeout 900 python3 $S notebook-push notebookH --yes 2>&1)
  if echo "$out" | grep -qE "429|Too Many|ProxyError|Max retries"; then
    log "H push attempt $i: 429/proxy, sleep 900"
    sleep 900
    continue
  fi
  ver=$(echo "$out" | grep -oE 'KernelVersion: [0-9]+' | grep -oE '[0-9]+' | tail -1)
  sleep 60
  st=$(kstate "$SLUG")
  if [ -n "$st" ]; then
    [ -n "$ver" ] && echo "$ver" > work/pushver_ntl-run06-bert3-h
    log "H push ok ver=${ver:-?} status=$st"; pushed=1; break
  fi
  log "H push attempt $i: non-429 but not registered, retry in 600"
  sleep 600
done
[ -n "$pushed" ] || { log "H push failed after 40 attempts"; exit 1; }
# wait terminal
start=$(date +%s)
while true; do
  sleep 300
  st=$(kstate "$SLUG")
  case "$st" in
    *COMPLETE*) log "H COMPLETE"; break ;;
    *CANCEL*|*ERROR*|*FAILED*) log "H terminal: $st"; break ;;
    "") log "H status unparseable (throttle window?), keep polling" ;;
  esac
  if [ $(( $(date +%s) - start )) -gt 64800 ]; then log "H watch >18h -> give up"; break; fi
done
# harvest + submit
OUT=work/H_out; mkdir -p "$OUT"
log "HARVEST H"
timeout 1800 kaggle kernels output "$SLUG" --file-pattern "submission.parquet" -p "$OUT" >> "$LOG" 2>&1
timeout 900 kaggle kernels output "$SLUG" --file-pattern "*.npz" -p "$OUT" >> "$LOG" 2>&1
timeout 900 kaggle kernels output "$SLUG" --file-pattern "*.npy" -p "$OUT" >> "$LOG" 2>&1
if [ -f "$OUT/submission.parquet" ]; then
  $PY work/validate_sub.py "$OUT/submission.parquet" >> "$LOG" 2>&1 && {
    timeout 900 python3 $S submit name-that-llm "$OUT/submission.parquet" -m "run06 H: BERT-trio 3x5fold equal blend (0.568 public fork) + probas" --expect 0.56 --yes >> "$LOG" 2>&1 \
      && log "H SUBMITTED" || log "H SUBMIT FAIL"
  } || log "H VALIDATE FAIL"
else
  log "H NO OUTPUT"
fi
log "=== h_watch DONE ==="
