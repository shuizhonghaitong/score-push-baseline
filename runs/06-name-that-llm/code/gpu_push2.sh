#!/bin/bash
# Robust GPU push watcher: retry notebookE then notebookB until accepted.
cd /mnt/disk3/liuyutong_home/workspace/skill/score-push/runs/06-name-that-llm
source env.sh
S=/mnt/disk3/liuyutong_home/workspace/skill/score-push/skills/kaggle/scripts/kaggle_skill.py
LOG=work/gpu_push2.log
mkdir -p work
EFLAG=work/pushed_E.flag
BFLAG=work/pushed_B.flag
: > "$LOG"
log() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
log "gpu_push2 started (pid $$)"

push_one() { # $1=notebook dir  $2=flag file  $3=slug
  if [ -f "$2" ]; then return 0; fi
  out=$(python "$S" notebook-push "$1" --yes 2>&1)
  if echo "$out" | grep -q "Maximum batch GPU session count"; then
    log "$1: CAP hit, retrying later"
    return 10
  fi
  if echo "$out" | grep -qi "exited with status [^0]"; then
    log "$1: PUSH ERROR -> $(echo "$out" | grep -iv '^<untrusted\|^</untrusted' | grep -v '^\s*$' | tail -2 | tr '\n' ' ')"
    return 20
  fi
  # verify a session now exists (not 404)
  st=$(kaggle kernels status "lilac1996/$3" 2>&1 | tail -1)
  if echo "$st" | grep -q "404"; then
    log "$1: push reported ok but status 404 (uncertain). $st"
    return 30
  fi
  log "$1: PUSH ACCEPTED. status: $st"
  touch "$2"
  return 0
}

while true; do
  if [ -f "$EFLAG" ] && [ -f "$BFLAG" ]; then log "both notebooks pushed; exiting"; break; fi
  if [ ! -f "$EFLAG" ]; then push_one notebookE "$EFLAG" ntl-run06-fork08b; fi
  if [ ! -f "$BFLAG" ]; then push_one notebookB "$BFLAG" ntl-run06-blend-v3; fi
  sleep 60
done
log "gpu_push2 loop ended"
