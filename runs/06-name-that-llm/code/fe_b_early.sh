#!/bin/bash
# Early-start watcher v2 (run 06): when the rolling GPU quota allows, push
# G (nlm-v21 fork, top public ensemble) -> F (0.612 ensemble fork) -> E2 (robust 0.600 fork) -> B (blend v3),
# harvesting + submitting each as it completes; finally offline blends.
cd /mnt/disk3/liuyutong_home/workspace/skill/score-push/runs/06-name-that-llm || exit 1
source env.sh
LOG=work/fe_b_early2.log
S=/mnt/disk3/liuyutong_home/workspace/skill/score-push/skills/kaggle/scripts/kaggle_skill.py
PY=./venv/bin/python
log() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }

gpu_line() { timeout 90 kaggle quota 2>/dev/null | awk '/^GPU/ {print $2, $3}'; }

wait_gpu() { # $1 min remaining hours
  local need=$1 cur rem
  log "WAIT-GPU need>=$need h"
  while true; do
    cur=$(gpu_line)
    rem=${cur##* }
    awk "BEGIN{exit !(${rem:-0} >= $need)}" && { log "GPU ready: $cur h"; return 0; }
    log "  quota [$cur h] (need $need)"
    sleep 900
  done
}

push_nb() { # $1 dir, $2 slug ; sets PUSH_VER
  local dir=$1 slug=$2 out
  log "PUSH $dir -> $slug"
  out=$(timeout 900 python3 $S notebook-push "$dir" --yes 2>&1)
  echo "$out" >> "$LOG"
  sleep 45
  local st
  st=$(timeout 90 kaggle kernels status "$slug" 2>&1 | tail -1)
  log "push->status: $st"
  if echo "$st" | grep -q "404\|Not found\|no such"; then log "PUSH not registered"; return 1; fi
  PUSH_VER=$(echo "$out" | grep -oE 'KernelVersion: [0-9]+' | grep -oE '[0-9]+' | tail -1)
  if [ -z "$PUSH_VER" ]; then
    local old_v; old_v=$(cat "work/pushver_${slug##*/}" 2>/dev/null || echo 0)
    PUSH_VER=$((old_v + 1))
  fi
  echo "$PUSH_VER" > "work/pushver_${slug##*/}"
  return 0
}

wait_terminal() { # $1 slug -> 0 complete, 2 cancel, 3 late error, 4 requeue, 5 timeout, 6 early error
  local slug=$1 start=$(date +%s) qs=
  while true; do
    sleep 90
    local st s
    st=$(timeout 90 kaggle kernels status "$slug" 2>&1 | tail -1)
    s=$(echo "$st" | grep -o 'KernelWorkerStatus\.[A-Z_]*')
    case "$s" in
      *COMPLETE*) log "TERMINAL COMPLETE: $st"; return 0 ;;
      *CANCEL*) log "TERMINAL CANCEL: $st"; return 2 ;;
      *ERROR*|*FAILED*)
        if [ $(( $(date +%s) - start )) -lt 2400 ]; then log "EARLY ERROR: $st"; return 6; fi
        log "TERMINAL ERROR: $st"; return 3 ;;
      *QUEUED*)
        if [ -z "$qs" ]; then qs=$(date +%s); log "queued (batch slot)"; fi
        if [ $(( $(date +%s) - qs )) -gt 10800 ]; then
          log "QUEUED >3h -> delete + repush"
          timeout 120 kaggle kernels delete "$slug" -y >> "$LOG" 2>&1
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

submit_file() { # $1 file, $2 msg, $3 expect
  local f=$1 msg=$2 expect=$3
  $PY work/validate_sub.py "$f" >> "$LOG" 2>&1 || { log "BLEND VALIDATE FAIL $f"; return 1; }
  log "SUBMIT-FILE $f ($msg) expect $expect"
  timeout 900 python3 $S submit name-that-llm "$f" -m "$msg" --expect "$expect" --yes >> "$LOG" 2>&1 \
    && log "SUBMITTED-FILE $f" || { log "SUBMIT-FILE FAIL $f"; return 1; }
}

run_candidate() { # $1 dir, $2 slug, $3 min_gpu, $4 msg, $5 expect, $6 outdir, $7 fdir(fallback), $8 fslug
  local dir=$1 slug=$2 need=$3 msg=$4 expect=$5 out=$6 fdir=$7 fslug=$8 attempt=1 rc
  while [ $attempt -le 3 ]; do
    wait_gpu "$need"
    if ! push_nb "$dir" "$slug"; then sleep 600; attempt=$((attempt+1)); continue; fi
    wait_terminal "$slug"; rc=$?
    case $rc in
      0) harvest "$slug" "$out" "$msg" "$expect" && return 0 || return 1 ;;
      4) log "re-push attempt $((attempt+1)) for $slug"; attempt=$((attempt+1)) ;;
      6)
        if [ -n "$fdir" ] && [ $attempt -le 1 ]; then
          log "early failure of $slug -> fallback $fdir"
          dir=$fdir; slug=$fslug; fdir=""; fslug=""
        else
          log "$slug early terminal rc=$rc -> skip candidate"; return 1
        fi ;;
      *) log "$slug terminal rc=$rc -> skip candidate"; return 1 ;;
    esac
  done
  log "$slug attempts exhausted -> skip"; return 1
}

log "=== fe_b_early v2 started (pid $$) $(date -u '+%FT%TZ') ==="
run_candidate notebookG lilac1996/ntl-run06-forkv21  8.0 "run06 G: nlm-v21 top-public ensemble fork, save_lp, 7.5h ATLAS" 0.62 work/G_out
run_candidate notebookF  lilac1996/ntl-run06-fork0612 6.0 "run06 F: 0.612 public ensemble fork + probas" 0.61 work/F_out
run_candidate notebookE2 lilac1996/ntl-run06-fork08b-e2 4.5 "run06 E2: two-stage LoRA 8B + SVC + Gaussian (robust)" 0.55 work/E2_out
run_candidate notebookB  lilac1996/ntl-run06-blend-v3 3.0 "run06 B: SVC4 x Qwen3-4B blend v3" 0.46 work/B_out

if [ -f work/G_out/final_lp.npy ] && [ -f work/E2_out/fork_probs.npz ]; then
  if $PY work/blend_ge.py >> "$LOG" 2>&1; then submit_file work/blend_GE.parquet "run06 blend G.70+E2.30 logit + sinkhorn" 0.63; fi
elif [ -f work/F_out/fork0612_probs.npz ] && [ -f work/E2_out/fork_probs.npz ]; then
  if $PY work/blend_fe.py >> "$LOG" 2>&1; then submit_file work/blend_FE.parquet "run06 blend F.70+E2.30 logit + sinkhorn" 0.62; fi
fi
if [ -f work/G_out/final_lp.npy ] && [ -f work/E2_out/fork_probs.npz ] && [ -f work/B_out/submission.parquet ]; then
  if $PY work/blend_geb.py >> "$LOG" 2>&1; then submit_file work/blend_GEB.parquet "run06 blend G.62+E2.28+B.10 + sinkhorn" 0.62; fi
fi
log "=== fe_b_early v2 DONE $(date -u '+%FT%TZ') ==="
