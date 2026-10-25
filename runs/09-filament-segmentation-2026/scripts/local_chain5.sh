#!/bin/bash
# chain5: after run3 completes, use the freed CPU for two more local models
#  (run4: 768 s123; run5: 1024 s7) and build 4-way / 5-way local ensemble
#  CSVs.  No auto-submit here: submission is manual or via the gcs queue once
#  the upload route recovers.
RUN=/mnt/disk3/liuyutong_home/workspace/skill/score-push/runs/09-filament-segmentation-2026
source $RUN/env.sh
LOG=$RUN/logs/local_chain5.log
PY=$RUN/.venv/bin/python
R1=$RUN/output/local-mirror
SNAP=$RUN/output/local-mirror-1024-snap
R3=$RUN/output/local-mirror-3
R4=$RUN/output/local-mirror-4
R5=$RUN/output/local-mirror-5
DATA=$RUN/data/official/MAGFiLO_1.0_Kaggle_2026

log() { echo "[$(date -Is)] $*" >> $LOG; }

log "=== chain5 start ==="

status=timeout
for i in $(seq 1 2160); do
  if grep -q "^\[done\]" $RUN/logs/local_train3.log 2>/dev/null; then status=done; break; fi
  if grep -q "Traceback" $RUN/logs/local_train3.log 2>/dev/null; then status=crash; break; fi
  sleep 120
done
log "run3 status=$status"
[ "$status" = done ] || { log "ABORT: run3 did not finish"; exit 1; }

log "launching run4 (768 s123)"
FILAMENT_DATA_ROOT=$DATA FILAMENT_WORK_DIR=$R4 FILAMENT_SEED=123 \
TORCH_HOME=/tmp/torchhome HF_HOME=/tmp/hfhome OMP_NUM_THREADS=8 \
$PY -u $RUN/src/notebooks/v1-local/train_v1.py >> $RUN/logs/local_train4.log 2>&1
rc=$?
log "run4 rc=$rc"
[ $rc -eq 0 ] || { log "ABORT: run4 failed"; exit 1; }
grep -q "^\[done\]" $RUN/logs/local_train4.log || { log "ABORT: run4 log missing [done]"; exit 1; }

# ensure the 1024 snapshot exists (chain4 makes it right after run2b)
for i in $(seq 1 2160); do
  [ -f $SNAP/run_report.json ] && { log "snap ready"; break; }
  sleep 300
done
[ -f $SNAP/run_report.json ] || { log "ABORT: 1024 snapshot never appeared"; exit 1; }

$PY $RUN/src/tune_ensemble.py $R1 $SNAP $R3 $R4 $R4/ensemble4local.csv >> $LOG 2>&1
log "ensemble4local rc=$?"
$PY $RUN/src/validate_submission.py $R4/ensemble4local.csv >> $LOG 2>&1
log "validate4local rc=$?"

# run5: 1024 s7 via a copy of the v1-local variant with size flipped
mkdir -p $RUN/src/notebooks/v1-local1024
cp -f $RUN/src/notebooks/v1-local/train_v1.py $RUN/src/notebooks/v1-local1024/train_v1.py
$PY - <<'PYEOF'
from pathlib import Path
p = Path("/mnt/disk3/liuyutong_home/workspace/skill/score-push/runs/09-filament-segmentation-2026/src/notebooks/v1-local1024/train_v1.py")
s = p.read_text()
assert s.count("        args.size = 768\n") == 1
p.write_text(s.replace("        args.size = 768\n", "        args.size = 1024\n"))
print("v1-local1024 ready")
PYEOF
log "launching run5 (1024 s7)"
FILAMENT_DATA_ROOT=$DATA FILAMENT_WORK_DIR=$R5 FILAMENT_SEED=7 \
TORCH_HOME=/tmp/torchhome HF_HOME=/tmp/hfhome OMP_NUM_THREADS=8 \
$PY -u $RUN/src/notebooks/v1-local1024/train_v1.py >> $RUN/logs/local_train5.log 2>&1
rc=$?
log "run5 rc=$rc"
[ $rc -eq 0 ] || { log "ABORT: run5 failed"; exit 1; }
grep -q "^\[done\]" $RUN/logs/local_train5.log || { log "ABORT: run5 log missing [done]"; exit 1; }

$PY $RUN/src/tune_ensemble.py $R1 $SNAP $R3 $R4 $R5 $R5/ensemble5local.csv >> $LOG 2>&1
log "ensemble5local rc=$?"
$PY $RUN/src/validate_submission.py $R5/ensemble5local.csv >> $LOG 2>&1
log "validate5local rc=$?"
log "=== chain5 end ==="
exit 0
