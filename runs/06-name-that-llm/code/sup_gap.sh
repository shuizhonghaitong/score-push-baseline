#!/bin/bash
# Gap supervisor (run 06): if v2's 8h gate for G stalls while partial GPU quota
# (>=3h) is available, run B -> E2 -> F in that order ourselves, hold their
# notebook dirs (sup_held_*) so v2 skips them, submit each; else idle.
cd /mnt/disk3/liuyutong_home/workspace/skill/score-push/runs/06-name-that-llm || exit 1
source env.sh
LOG=work/sup_gap.log
S=/mnt/disk3/liuyutong_home/workspace/skill/score-push/skills/kaggle/scripts/kaggle_skill.py
PY=./venv/bin/python
log() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }

gpu_rem() { timeout 90 kaggle quota 2>/dev/null | awk '/^GPU/ {print $3}'; }
g_running() { timeout 90 kaggle kernels status lilac1996/ntl-run06-forkv21 2>/dev/null | grep -qE "RUNNING|QUEUED"; }

push_nb() { # $1 dir, $2 slug ; sets PUSH_VER
  local dir=$1 slug=$2 out
  log "PUSH $dir -> $slug"
  out=$(timeout 900 python3 $S notebook-push "$dir" --yes 2>&1)
  echo "$out" >> "$LOG"
  sleep 45
  local st; st=$(timeout 90 kaggle kernels status "$slug" 2>&1 | tail -1)
  log "push->status: $st"
  if echo "$st" | grep -q "404\|Not found\|no such"; then log "PUSH not registered"; return 1; fi
  PUSH_VER=$(echo "$out" | grep -oE 'KernelVersion: [0-9]+' | grep -oE '[0-9]+' | tail -1)
  if [ -z "$PUSH_VER" ]; then
    local old_v; old_v=$(cat "work/pushver_${slug##*/}" 2>/dev/null || echo 0)
    PUSH_VER=$((old_v + 1)); echo "$PUSH_VER" > "work/pushver_${slug##*/}"
  fi
  return 0
}

wait_terminal() { # $1 slug -> 0 complete, 2 cancel, 3 error, 4 requeue, 5 timeout
  local slug=$1 start=$(date +%s) qs=
  while true; do
    sleep 120
    local st s
    st=$(timeout 90 kaggle kernels status "$slug" 2>&1 | tail -1)
    s=$(echo "$st" | grep -o 'KernelWorkerStatus\.[A-Z_]*')
    case "$s" in
      *COMPLETE*) log "TERMINAL COMPLETE: $st"; return 0 ;;
      *CANCEL*) log "TERMINAL CANCEL: $st"; return 2 ;;
      *ERROR*|*FAILED*) log "TERMINAL ERROR: $st"; return 3 ;;
      *QUEUED*)
        if [ -z "$qs" ]; then qs=$(date +%s); log "queued (batch slot)"; fi
        if [ $(( $(date +%s) - qs )) -gt 10800 ]; then
          log "QUEUED >3h -> delete + repush"; timeout 120 kaggle kernels delete "$slug" -y >> "$LOG" 2>&1
          return 4
        fi ;;
      *RUNNING*) qs="" ;;
    esac
    if [ $(( $(date +%s) - start )) -gt 50400 ]; then log "watch >14h -> give up"; return 5; fi
  done
}

harvest() { # $1 slug, $2 outdir, $3 msg, $4 expect
  local slug=$1 out=$2 msg=$3 expect=$4
  log "HARVEST $slug -> $out"
  mkdir -p "$out"
  timeout 1800 kaggle kernels output "$slug" --file-pattern "submission.parquet" -p "$out" >> "$LOG" 2>&1
  timeout 900 kaggle kernels output "$slug" --file-pattern "*.npz" -p "$out" >> "$LOG" 2>&1
  timeout 900 kaggle kernels output "$slug" --file-pattern "*.npy" -p "$out" >> "$LOG" 2>&1
  local sub="$out/submission.parquet"
  [ -f "$sub" ] || { log "NO OUTPUT for $slug"; return 1; }
  $PY work/validate_sub.py "$sub" >> "$LOG" 2>&1 || { log "VALIDATE FAIL $slug"; return 1; }
  local ver; ver=$(cat "work/pushver_${slug##*/}" 2>/dev/null)
  log "SUBMIT $slug v$ver ($msg) expect $expect"
  timeout 900 python3 $S submit name-that-llm --notebook "$slug" --version "$ver" -m "$msg" --expect "$expect" --yes >> "$LOG" 2>&1 \
    && log "SUBMITTED $slug v$ver" || { log "SUBMIT FAIL $slug"; return 1; }
  return 0
}

run_one() { # $1 dir, $2 slug, $3 gate, $4 msg, $5 expect, $6 outdir
  local dir=$1 slug=$2 gate=$3 msg=$4 expect=$5 out=$6 attempt=1 rem rc
  rem=$(gpu_rem)
  awk "BEGIN{exit !(${rem:-0} >= $gate)}" || { log "gate not met: rem=${rem} need=$gate"; return 1; }
  while [ $attempt -le 3 ]; do
    if ! push_nb "$dir" "$slug"; then sleep 900; attempt=$((attempt+1)); continue; fi
    wait_terminal "$slug"; rc=$?
    case $rc in
      0) harvest "$slug" "$out" "$msg" "$expect" && return 0 || return 1 ;;
      4) log "re-push attempt $((attempt+1))"; attempt=$((attempt+1)) ;;
      *) log "terminal rc=$rc -> skip"; return 1 ;;
    esac
  done
  return 1
}

log "=== sup_gap started (pid $$) $(date -u '+%FT%TZ') ==="
cand_done() { [ -f "work/sup_done_$1" ]; }
v2_pushed() { grep -q "PUSH notebook$1" work/fe_b_early2.log 2>/dev/null; }

while true; do
  rem=$(gpu_rem)
  log "poll: rem=${rem:-?}h G_running=$(g_running && echo y || echo n)"
  # ideal path: enough for G and v2 active -> we do nothing
  if awk "BEGIN{exit !(${rem:-0} >= 8.0)}"; then log "quota >=8h -> idle (v2 owns queue)"; sleep 1800; continue; fi
  # stall path: partial quota, run what fits (ascending gates), one at a time
  if awk "BEGIN{exit !(${rem:-0} >= 3.0)}"; then
    for spec in "B 3.0" "E2 4.5" "F 6.0"; do
      set -- $spec; c=$1; g=$2
      [ -d "notebook$c" ] || continue
      cand_done "$c" && continue
      v2_pushed "$c" && { log "$c already handled by v2"; continue; }
      # E2/F slugs
      case $c in
        B)  slug=lilac1996/ntl-run06-blend-v3;    msg="run06 B: SVC4 x Qwen3-4B blend v3";  exp=0.46 ;;
        E2) slug=lilac1996/ntl-run06-fork08b-e2;  msg="run06 E2: two-stage LoRA 8B robust"; exp=0.55 ;;
        F)  slug=lilac1996/ntl-run06-fork0612;    msg="run06 F: 0.612 public ensemble fork"; exp=0.61 ;;
      esac
      log "GAP: starting $c (rem=${rem}h)"
      if run_one "notebook$c" "$slug" "$g" "$msg" "$exp" "work/${c}_out"; then
        mv "notebook$c" "work/sup_held_notebook$c"
        touch "work/sup_done_$c"
        log "GAP: $c done and held"
      else
        log "GAP: $c failed; retrying later (dir left in place)"
      fi
      break  # one candidate per poll cycle
    done
  fi
  sleep 1200
done
