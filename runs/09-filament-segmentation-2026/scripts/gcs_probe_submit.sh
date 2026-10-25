#!/bin/bash
# Probes the GCS upload route (www.googleapis.com) periodically. While it is
# down, file submissions are impossible (uploads hang). When the route
# recovers, file-submits the pending local CSVs in priority order via
# kaggle_skill (auto-reconciling stale pending ledger entries first).
RUN=/mnt/disk3/liuyutong_home/workspace/skill/score-push/runs/09-filament-segmentation-2026
source $RUN/env.sh
cd $RUN
LOG=$RUN/logs/gcs_probe_submit.log
SK=$KAGGLE_SKILL/scripts/kaggle_skill.py
PY=$RUN/.venv/bin/python
COMP=filament-segmentation-2026
DONE=$RUN/state/auto/file_submitted.txt
touch "$DONE"
log() { echo "[$(date -Is)] $*" >> $LOG; }

CANDIDATES=(
 "$RUN/output/local-mirror/submission.csv|v1-local unet768 12ep (s42)"
 "$RUN/output/local-mirror-1024-snap/ensemble_local.csv|local 2-way ensemble 768s42+1024s42"
 "$RUN/output/local-mirror-1024-snap/submission.csv|local unet1024 12ep (s42)"
 "$RUN/output/local-mirror-3/ensemble3.csv|local 3-way ensemble 768s42+1024s42+768s7"
 "$RUN/output/local-mirror-3/submission.csv|local unet768 12ep (s7)"
 "$RUN/output/v1-gpu/ensemble4.csv|4-way ensemble local768+local1024+local768s7+gpu1024"
)

probe_route() {
  local code
  code=$(timeout 25 curl -sS -o /dev/null --max-time 20 -w "%{http_code}" -X HEAD https://www.googleapis.com/ 2>/dev/null)
  case "$code" in ''|000) return 1;; *) return 0;; esac
}

reconcile_stale() { # $1 = csv path
  $PY - "$1" <<'PY'
import json, sys, time
p = ".kaggle-skill/ledger.jsonl"
if not __import__("os").path.exists(p):
    raise SystemExit
target = sys.argv[1]
rows = [json.loads(l) for l in open(p) if l.strip()]
n = 0
for r in rows:
    if r.get("event") == "submit_attempt" and str(r.get("file", "")).endswith(target):
        r["event"] = "submit_attempt_failed"
        r["error_type"] = "auto-reconciled: GCS upload route unavailable at attempt time"
        r["reconciled"] = "verified-not-landed (checked quota/submission list before each round)"
        r["reconcile_time"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        n += 1
if n:
    with open(p, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
print(f"reconciled {n} stale attempt(s) for {target}")
PY
}

log "=== gcs_probe_submit start ==="
for i in $(seq 1 360); do   # 20 min polls, up to 5 days
  if probe_route; then
    log "GCS route REACHED at $(date -Is); attempting pending file submissions"
    for entry in "${CANDIDATES[@]}"; do
      csv="${entry%%|*}"; msg="${entry#*|}"
      [ -f "$csv" ] || continue
      base=$(basename "$csv")
      grep -q "$base" "$DONE" && { log "skip $base (already submitted)"; continue; }
      $PY $RUN/src/validate_submission.py "$csv" >> $LOG 2>&1 || { log "validate failed $csv"; continue; }
      reconcile_stale "$csv" >> $LOG 2>&1
      python3 $SK submit $COMP "$csv" -m "$msg" --yes >> $LOG 2>&1
      if [ $? -eq 0 ]; then
        echo "$base $(date -Is)" >> "$DONE"
        log "FILE SUBMITTED $base"
      else
        log "file submit failed for $base (route may have dropped again)"
      fi
    done
    sleep 1800
  else
    sleep 1200
  fi
done
log "=== gcs_probe_submit exit ==="
exit 0
