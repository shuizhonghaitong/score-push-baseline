#!/bin/bash
# Wait for: (a) GPU v1 kernel output downloaded by autopilot (output/v1-gpu),
# (b) run3 completion (output/local-mirror-3), (c) the 1024 snapshot
# (output/local-mirror-1024-snap, created by chain3 after run2).  Then build a
# 4-run ensemble and submit it (plus a v1-gpu best-cand if autopilot has not
# already submitted it) after the next UTC daily quota rollover.
RUN=/mnt/disk3/liuyutong_home/workspace/skill/score-push/runs/09-filament-segmentation-2026
source $RUN/env.sh
LOG=$RUN/logs/gpu_ensemble_chain.log
SK=$KAGGLE_SKILL/scripts/kaggle_skill.py
PY=$RUN/.venv/bin/python
V1GPU=$RUN/output/v1-gpu
R1=$RUN/output/local-mirror
R2S=$RUN/output/local-mirror-1024-snap
R3=$RUN/output/local-mirror-3
AUTO=$RUN/state/auto
log() { echo "[$(date -Is)] $*" >> $LOG; }

log "=== gpu_ensemble_chain start ==="
status=timeout
for i in $(seq 1 2880); do   # up to 4 days
  if [ -f $V1GPU/run_report.json ] && [ -f $R3/run_report.json ] && \
     [ -f $R2S/run_report.json ] && ls $V1GPU/val_predictions_*.npz >/dev/null 2>&1; then
    status=done; break
  fi
  sleep 300
done
log "wait status=$status (v1gpu=$([ -f $V1GPU/run_report.json ] && echo y || echo n) r3=$([ -f $R3/run_report.json ] && echo y || echo n) r2snap=$([ -f $R2S/run_report.json ] && echo y || echo n))"
[ "$status" = done ] || { log "ABORT: inputs never ready"; exit 1; }

# 4-run ensemble: local 768s42 + 1024s42(snap) + 768s7 + gpu 1024(unet/mrcnn/merge)
$PY $RUN/src/tune_ensemble.py $R1 $R2S $R3 $V1GPU $V1GPU/ensemble4.csv >> $LOG 2>&1
log "ensemble4 rc=$?"

# wait for next UTC daily rollover
now_utc=$(date -u +%s); next=$(date -u -d "today 23:59:00" +%s)
[ $now_utc -lt $next ] && sleep $((next - now_utc))
log "quota rollover passed $(date -Is)"

# submit the ensemble4 if validation passes
$PY $RUN/src/validate_submission.py $V1GPU/ensemble4.csv >> $LOG 2>&1 && \
python3 $SK submit filament-segmentation-2026 $V1GPU/ensemble4.csv -m "4-way ensemble local768+local1024+local768s7+gpu1024" --yes >> $LOG 2>&1
log "submit ensemble4 rc=$?"
python3 $SK status filament-segmentation-2026 >> $LOG 2>&1 || true
log "=== gpu_ensemble_chain end ==="
exit 0
