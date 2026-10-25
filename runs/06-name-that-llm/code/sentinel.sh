#!/bin/bash
# Push sentinel (run 06): if v2/sup_gap log a PUSH but the kernel never registers
# (429/proxy failure), re-push the same notebook dir ourselves, fix pushver, and
# (only when the watcher already gave up) wait+harvest+submit the rescued run.
cd /mnt/disk3/liuyutong_home/workspace/skill/score-push/runs/06-name-that-llm || exit 1
source env.sh
LOG=work/sentinel.log
S=/mnt/disk3/liuyutong_home/workspace/skill/score-push/skills/kaggle/scripts/kaggle_skill.py
PY=./venv/bin/python
log() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }

kernel_exists() { timeout 90 kaggle kernels status "$1" 2>/dev/null | grep -q "KernelWorkerStatus"; }

DIR_OF() { case $1 in
  *forkv21*) echo notebookG ;;
  *fork0612*) echo notebookF ;;
  *fork08b-e2*) echo notebookE2 ;;
  *blend-v3*) echo notebookB ;;
  *v3-cpu*) echo notebookC2 ;;
esac; }

rescue() { # $1 slug $2 dir
  local slug=$1 dir=$2
  [ -d "$dir" ] || { log "rescue $slug: dir $dir missing"; return; }
  log "RESCUE push $slug from $dir"
  local out ver
  out=$(timeout 900 python3 $S notebook-push "$dir" --yes 2>&1)
  echo "$out" >> "$LOG"
  ver=$(echo "$out" | grep -oE 'KernelVersion: [0-9]+' | grep -oE '[0-9]+' | tail -1)
  sleep 60
  if kernel_exists "$slug"; then
    [ -n "$ver" ] && echo "$ver" > "work/pushver_${slug##*/}"
    log "RESCUE ok: $slug exists (ver ${ver:-?})"
  else
    log "RESCUE failed for $slug; will not retry (watcher owns subsequent attempts)"
  fi
}

log "=== sentinel started $(date -u '+%FT%TZ') ==="
while true; do
  for src in work/fe_b_early2.log work/sup_gap.log; do
    [ -f "$src" ] || continue
    # last PUSH line in this log
    line=$(grep "PUSH " "$src" | tail -1)
    [ -n "$line" ] || continue
    ts=$(echo "$line" | awk '{print $1" "$2}')
    slug=$(echo "$line" | grep -oE 'lilac1996/[a-z0-9-]+' | head -1)
    [ -n "$slug" ] || continue
    marker="work/.sentinel_seen_${slug##*/}"
    if [ -f "$marker" ]; then continue; fi
    # wait 25 min after the logged push before judging (v2 retries at +10m, +20m)
    t_line=$(date -d "$ts" +%s 2>/dev/null) || continue
    age=$(( $(date +%s) - t_line ))
    if [ $age -lt 1500 ]; then sleep 120; break; fi
    echo "$age" > "$marker"
    if kernel_exists "$slug"; then
      log "$slug registered (age ${age}s) - no action"
    else
      dir=$(DIR_OF "$slug")
      rescue "$slug" "$dir"
      # if the original watcher gave up on this slug, do full wait+harvest+submit
      sleep 90
      if ! kernel_exists "$slug"; then continue; fi
      # check whether the watcher will still track it (it gave up if it logged terminal/give-up for this slug after the push)
      gave_up=$(awk -v s="$slug" '$0 ~ s && /give up|terminal rc=|attempts exhausted|skip candidate/' "$src" | tail -1 | awk '{print $1" "$2}')
      if [ -n "$gave_up" ] && [ "$(date -d "$gave_up" +%s 2>/dev/null)" -gt "$t_line" ] 2>/dev/null; then
        log "watcher gave up on $slug -> sentinel takes over wait+harvest+submit"
        start=$(date +%s); qs=
        while true; do
          sleep 120
          st=$(timeout 90 kaggle kernels status "$slug" 2>&1 | tail -1)
          s=$(echo "$st" | grep -o 'KernelWorkerStatus\.[A-Z_]*')
          case "$s" in
            *COMPLETE*) log "RESCUED $slug COMPLETE"; break ;;
            *CANCEL*|*ERROR*|*FAILED*) log "RESCUED $slug terminal: $st"; break ;;
            *QUEUED*)
              if [ -z "$qs" ]; then qs=$(date +%s); fi
              if [ $(( $(date +%s) - qs )) -gt 10800 ]; then log "rescue queued >3h"; break; fi ;;
          esac
          if [ $(( $(date +%s) - start )) -gt 50400 ]; then log "rescue watch >14h"; break; fi
        done
        outdir="work/rescue_${slug##*/}_out"; mkdir -p "$outdir"
        timeout 1800 kaggle kernels output "$slug" --file-pattern "submission.parquet" -p "$outdir" >> "$LOG" 2>&1
        timeout 900 kaggle kernels output "$slug" --file-pattern "*.npz" -p "$outdir" >> "$LOG" 2>&1
        timeout 900 kaggle kernels output "$slug" --file-pattern "*.npy" -p "$outdir" >> "$LOG" 2>&1
        if [ -f "$outdir/submission.parquet" ]; then
          $PY work/validate_sub.py "$outdir/submission.parquet" >> "$LOG" 2>&1 \
            && timeout 900 python3 $S submit name-that-llm "$outdir/submission.parquet" -m "run06 rescue: $slug (sentinel)" --expect 0.5 --yes >> "$LOG" 2>&1 \
            && log "RESCUE SUBMITTED $slug" || log "RESCUE SUBMIT FAIL"
        fi
      fi
    fi
  done
  sleep 300
done
