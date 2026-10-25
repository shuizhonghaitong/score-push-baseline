#!/bin/bash
# Autopilot for filament-segmentation-2026.
# Watches kernel states, downloads outputs, validates and submits per the runbook.
# Pushes: CPU v10 -> session 87901 (cpu_v10.sh); GPU v1 -> session 42554 + this
# script (backup, pre-gated on the weekly GPU budget reset); v2 -> this script
# (only after GPU v1 is COMPLETE).
source /mnt/disk3/liuyutong_home/workspace/skill/score-push/runs/09-filament-segmentation-2026/env.sh
RUN=/mnt/disk3/liuyutong_home/workspace/skill/score-push/runs/09-filament-segmentation-2026
COMP=filament-segmentation-2026
AUTO="$RUN/state/auto"; mkdir -p "$AUTO"
LOG="$RUN/logs/autopilot.log"
PY="$RUN/.venv/bin/python"
NB_WAIT="$KAGGLE_SKILL/modules/notebooks/scripts/notebook_wait.py"
NB_PUSH="$KAGGLE_SKILL/modules/notebooks/scripts/notebook_push.py"
SKILL="$KAGGLE_SKILL/scripts/kaggle_skill.py"
CPU_SLUG=lilac1996/filament-magfilo-v1-cpu-u-net
GPU_SLUG=lilac1996/filament-magfilo-v1
V2_SLUG=lilac1996/filament-magfilo-v2
GPU_GATE=$(date -u -d '2026-10-24 00:10:00' +%s)
DATA_GATE=$(date -u -d '2026-10-24 12:55:00' +%s)
STOP_AT=$(date -u -d '2026-11-14 18:00:00' +%s)
log() { echo "[$(date -Is)] $*" | tee -a "$LOG"; }
kstatus() { kaggle kernels status "$1" 2>&1 | grep -oE 'KernelWorkerStatus\.[A-Z_]+' | head -1; }
remaining() {
  "$PY" "$SKILL" status "$COMP" 2>&1 | grep -oE '[0-9]+ left of [0-9]+ a day' | head -1 | grep -oE '^[0-9]+'
}
daily_count() { cat "$AUTO/submits_$(date -u +%Y%m%d)" 2>/dev/null || echo 0; }
submit() { # $1 csv  $2 message
  local csv="$1" msg="$2"
  [ -f "$csv" ] || { log "submit: missing $csv"; return 1; }
  if [ "$(daily_count)" -ge 3 ]; then log "submit: daily cap (3) reached; defer"; return 1; fi
  local rem; rem=$(remaining); case "$rem" in ''|0) log "submit: no daily quota (rem=$rem); defer"; return 1;; esac
  "$PY" "$RUN/src/validate_submission.py" "$csv" > "$AUTO/validate_$(basename "$csv").log" 2>&1
  if ! grep -q "RLE decode OK" "$AUTO/validate_$(basename "$csv").log"; then
    log "submit: VALIDATION FAILED for $csv"; tail -5 "$AUTO/validate_$(basename "$csv").log" >> "$LOG"; return 1
  fi
  if ! grep -q "missing=0" "$AUTO/validate_$(basename "$csv").log"; then
    log "submit: stems not fully covered in $csv; defer"; return 1
  fi
  log "submitting $csv ($msg)"
  "$PY" "$SKILL" submit "$COMP" "$csv" -m "$msg" --yes >> "$LOG" 2>&1
  if [ $? -ne 0 ]; then log "submit: kaggle_skill submit failed (429?); retry next cycle"; return 1; fi
  "$PY" "$SKILL" watch "$COMP" --interval 60 --timeout 2400 >> "$LOG" 2>&1
  local n; n=$(( $(daily_count) + 1 )); echo "$n" > "$AUTO/submits_$(date -u +%Y%m%d)"
  echo "$(date -u +%Y-%m-%dT%H:%M:%SZ),$csv,$msg,$n" >> "$AUTO/submitted.csv"
  log "submitted $csv (day count=$n)"
  return 0
}

log "=== autopilot start (pid $$) ==="
for i in $(seq 1 260); do
  now=$(date -u +%s)
  [ "$now" -ge "$STOP_AT" ] && { log "stop: deadline reached"; break; }

  # ---------- Phase A: CPU v10 ----------
  if [ -f "$AUTO/cpu_submitted" ]; then
    :
  else
    st=$(kstatus "$CPU_SLUG")
    case "$st" in
      *COMPLETE*)
        if [ ! -f "$RUN/output/cpu-v10/run_report.json" ]; then
          "$PY" "$NB_WAIT" "$CPU_SLUG" --out "$RUN/output/cpu-v10" --timeout 3600 >> "$LOG" 2>&1
          log "cpu download rc=$?"
        fi
        if [ -f "$RUN/output/cpu-v10/run_report.json" ]; then
          best=$("$PY" -c "import json;print(json.load(open('$RUN/output/cpu-v10/run_report.json')).get('best',''))" 2>/dev/null)
          csv="$RUN/output/cpu-v10/submission_${best}.csv"; [ -f "$csv" ] || csv="$RUN/output/cpu-v10/submission.csv"
          if submit "$csv" "v10 cpu u-net 512 bs2 10ep (auto)"; then
            touch "$AUTO/cpu_submitted"
          fi
        fi
        ;;
      *RUNNING*) log "cpu kernel RUNNING (wait for pusher 87901 / completion)";;
      *) log "cpu kernel status=$st (pusher 87901 handles retries)";;
    esac
  fi

  # ---------- Phase B: GPU v1 (gated on weekly budget reset) ----------
  if [ -f "$AUTO/gpu1_submitted" ]; then
    :
  elif [ "$now" -ge "$GPU_GATE" ]; then
    st=$(kstatus "$GPU_SLUG")
    case "$st" in
      *RUNNING*) log "gpu kernel RUNNING";;
      *COMPLETE*)
        if [ ! -f "$RUN/output/v1-gpu/run_report.json" ]; then
          "$PY" "$NB_WAIT" "$GPU_SLUG" --out "$RUN/output/v1-gpu" --timeout 3600 >> "$LOG" 2>&1
          log "gpu download rc=$?"
        fi
        if [ -f "$RUN/output/v1-gpu/run_report.json" ]; then
          best=$("$PY" -c "import json;print(json.load(open('$RUN/output/v1-gpu/run_report.json')).get('best',''))" 2>/dev/null)
          bestpq=$("$PY" -c "import json;r=json.load(open('$RUN/output/v1-gpu/run_report.json'));print(r['val_pq'][r['best']]['pq'])" 2>/dev/null)
          csv="$RUN/output/v1-gpu/submission_${best}.csv"; [ -f "$csv" ] || csv="$RUN/output/v1-gpu/submission.csv"
          if [ -f "$csv" ]; then
            if "$PY" "$RUN/src/tune_postprocess.py" "$RUN/output/v1-gpu" "$RUN/output/v1-gpu/tuned_submission.csv" > "$RUN/output/v1-gpu/tune.log" 2>&1; then
              tunedpq=$(grep -oE 'best rule: [a-z_]+ PQ=[0-9.]+' "$RUN/output/v1-gpu/tune.log" | grep -oE 'PQ=[0-9.]+' | cut -d= -f2)
              use_tuned=$("$PY" -c "import sys;t=float('$tunedpq' or 0);b=float('$bestpq' or 0);print('yes' if t>b+0.002 else 'no')" 2>/dev/null)
              if [ "$use_tuned" = "yes" ] && [ -f "$RUN/output/v1-gpu/tuned_submission.csv" ]; then
                log "tuned PQ=$tunedpq > raw best PQ=$bestpq; submitting tuned"
                csv="$RUN/output/v1-gpu/tuned_submission.csv"
              fi
            fi
            if submit "$csv" "v1 gpu unet+mrcnn best-cand (auto)"; then
              touch "$AUTO/gpu1_submitted"
            fi
          else
            log "gpu1: no candidate csv found (best=$best)"
          fi
        fi
        ;;
      *)
        # push (backup role) then a short wait to see whether it starts
        log "gpu push (backup) status=$st"
        (cd "$RUN/src/notebooks/v1" && "$PY" "$NB_PUSH" . --yes >> "$LOG" 2>&1)
        log "gpu push rc=$?"
        "$PY" "$NB_WAIT" "$GPU_SLUG" --out "$RUN/output/v1-gpu" --timeout 600 >> "$LOG" 2>&1
        log "gpu short-wait rc=$? (124=still running: ok)"
        ;;
    esac
  fi

  # ---------- Phase C: v2 (only after GPU v1 submitted) ----------
  if [ -f "$AUTO/v2_submitted" ]; then
    :
  elif [ -f "$AUTO/gpu1_submitted" ]; then
    st=$(kstatus "$V2_SLUG")
    case "$st" in
      *RUNNING*) log "v2 kernel RUNNING";;
      *COMPLETE*)
        if [ ! -f "$RUN/output/v2/run_report.json" ]; then
          "$PY" "$NB_WAIT" "$V2_SLUG" --out "$RUN/output/v2" --timeout 3600 >> "$LOG" 2>&1
          log "v2 download rc=$?"
        fi
        if [ -f "$RUN/output/v2/run_report.json" ]; then
          best=$("$PY" -c "import json;print(json.load(open('$RUN/output/v2/run_report.json')).get('best',''))" 2>/dev/null)
          csv="$RUN/output/v2/submission_${best}.csv"; [ -f "$csv" ] || csv="$RUN/output/v2/submission.csv"
          if submit "$csv" "v2 refiner best-cand (auto)"; then
            touch "$AUTO/v2_submitted"
          fi
        fi
        ;;
      *)
        log "v2 push (gpu1 complete) status=$st"
        (cd "$RUN/src/notebooks/v2" && "$PY" "$NB_PUSH" . --yes >> "$LOG" 2>&1)
        log "v2 push rc=$?"
        "$PY" "$NB_WAIT" "$V2_SLUG" --out "$RUN/output/v2" --timeout 600 >> "$LOG" 2>&1
        ;;
    esac
  fi

  # ---------- Phase D: official data download window ----------
  if [ ! -f "$AUTO/data_downloaded" ] && [ "$now" -ge "$DATA_GATE" ]; then
    if kaggle competitions download "$COMP" --path "$RUN/data/official" --quiet >> "$LOG" 2>&1; then
      touch "$AUTO/data_downloaded"; log "official data downloaded"
    else
      log "official data download failed; retry next cycle"
    fi
  fi

  # all done?
  if [ -f "$AUTO/v2_submitted" ] && [ -f "$AUTO/data_downloaded" ]; then
    log "all phases done; sleeping long"; sleep 3600
  fi
  sleep 600
done
log "=== autopilot exit ==="
