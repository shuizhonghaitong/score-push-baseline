#!/bin/bash
# Harvest + submit the CPU v3 notebook (safe + hung variants).
cd /mnt/disk3/liuyutong_home/workspace/skill/score-push/runs/06-name-that-llm || exit 1
source env.sh
S=/mnt/disk3/liuyutong_home/workspace/skill/score-push/skills/kaggle/scripts/kaggle_skill.py
PY=./venv/bin/python
LOG=work/c2_harvest.log
log() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
SLUG=lilac1996/ntl-run06-v3-cpu
OUT=work/C2_out
mkdir -p "$OUT"
log "harvest start"
timeout 1800 kaggle kernels output "$SLUG" --file-pattern "submission.parquet" -p "$OUT" >> "$LOG" 2>&1
timeout 1800 kaggle kernels output "$SLUG" --file-pattern "*.npy" -p "$OUT" >> "$LOG" 2>&1
timeout 1800 kaggle kernels output "$SLUG" --file-pattern "*.npz" -p "$OUT" >> "$LOG" 2>&1
if [ -f work/.c2_submitted_locally ]; then
  log "local C2 already submitted -> harvest files only, no submit"
  [ -f "$OUT/submission.parquet" ] || true
  log "harvest done (no submit)"; exit 0
fi
ls -la "$OUT" >> "$LOG"
[ -f "$OUT/submission.parquet" ] || { log "NO submission.parquet"; exit 1; }
$PY work/validate_sub.py "$OUT/submission.parquet" >> "$LOG" 2>&1 && log "safe validated"
if [ -f "$OUT/submission_hung.parquet" ]; then
  $PY work/validate_sub.py "$OUT/submission_hung.parquet" >> "$LOG" 2>&1 && log "hung validated"
fi
# submit safe first
timeout 900 $PY -c "pass" 2>/dev/null
timeout 900 python3 $S submit name-that-llm "$OUT/submission.parquet"  -m "run06 C2: 5-member TF-IDF SVC v3 (equal-weight blend, sink+emdash) CPU" --expect 0.455 --yes >> "$LOG" 2>&1 && log "SUBMITTED safe" || log "SUBMIT FAIL safe"
if [ -f "$OUT/submission_hung.parquet" ]; then
  timeout 900 python3 $S submit name-that-llm "$OUT/submission_hung.parquet" -m "run06 C2h: 5-member TF-IDF SVC v3 + Hungarian dense clusters (fcluster 0.90)" --expect 0.52 --yes >> "$LOG" 2>&1 && log "SUBMITTED hung" || log "SUBMIT FAIL hung"
fi
if [ -f "$OUT/submission_hung85.parquet" ]; then
  timeout 900 python3 $S submit name-that-llm "$OUT/submission_hung85.parquet" -m "run06 C2h85: 5-member TF-IDF SVC v3 + Hungarian dense clusters (fcluster 0.85)" --expect 0.50 --yes >> "$LOG" 2>&1 && log "SUBMITTED hung85" || log "SUBMIT FAIL hung85"
fi
log "harvest done"
