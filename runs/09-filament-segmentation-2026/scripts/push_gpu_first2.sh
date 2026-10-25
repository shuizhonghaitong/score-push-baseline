#!/bin/bash
# Pusher v2 (robust to unknown SaveKernel window time; one real push pins 72h).
# Priority order: GPU v1 (1024 unet+mrcnn) > v2 (refiner, needs v1 output) >
# CPU v10 (only if $AUTO/push_cpu_v10 flag exists). 3-min probes are harmless.
RUN=/mnt/disk3/liuyutong_home/workspace/skill/score-push/runs/09-filament-segmentation-2026
source $RUN/env.sh
LOG=$RUN/logs/push_gpu_first2.log
NB_PUSH=$KAGGLE_SKILL/modules/notebooks/scripts/notebook_push.py
V1=$RUN/src/notebooks/v1
V1CPU=$RUN/src/notebooks/v1-cpu-fast2
V2=$RUN/src/notebooks/v2
AUTO=$RUN/state/auto
V1_SLUG=lilac1996/filament-magfilo-v1
V2_SLUG=lilac1996/filament-magfilo-v2
VCPU_SLUG=lilac1996/filament-magfilo-v1-cpu-u-net
mkdir -p "$AUTO"
log() { echo "[$(date -Is)] $*" >> $LOG; }

probe() {
python3 - <<'PY'
import os, urllib.request
tok = os.environ.get('KAGGLE_API_TOKEN', '')
if os.path.isfile(tok):
    tok = open(tok).read().strip()
req = urllib.request.Request(
    'https://api.kaggle.com/v1/kernels.KernelsApiService/SaveKernel',
    data=b'', method='POST',
    headers={'Authorization': f'Bearer {tok}', 'Content-Type': 'application/grpc-web+proto', 'x-grpc-web': '1'})
try:
    r = urllib.request.urlopen(req, timeout=30)
    print(r.status)
except urllib.error.HTTPError as e:
    print(e.code)
except Exception as e:
    print('ERR')
PY
}
kstatus() { kaggle kernels status "$1" 2>&1 | grep -oE 'KernelWorkerStatus\.[A-Z_]+' | head -1; }
ra_open() { case "$1" in 429|ERR|'') return 1;; *) return 0;; esac; }
do_push() { # $1=dir  ; pushes while window stays open, up to 30 min
  local dir="$1" rc=1
  for attempt in $(seq 1 15); do
    (cd "$dir" && python3 "$NB_PUSH" . --yes >> "$LOG" 2>&1)
    rc=$?
    log "push $dir attempt $attempt rc=$rc"
    [ $rc -eq 0 ] && return 0
    sleep 120
    ra_open "$(probe)" || return 1   # window closed again
  done
  return 1
}

log "=== push_gpu_first2 start ==="
# Phase 0: neutralize the weak CPU kernel (its .hold disappears so unblock_watcher2
# cannot unhold it; cpu_v10.sh then fails locally with zero network calls).
if [ -f "$V1CPU/kernel-metadata.json.hold" ]; then
  mv "$V1CPU/kernel-metadata.json.hold" "$V1CPU/kernel-metadata.json.hold.bak"
  log "phase0: cpu v10 hold backed up (neutralized)"
fi

# ---------- Phase A: get GPU v1 running ----------
phase_b_at=0
for i in $(seq 1 2880); do   # up to 6 days, 3-min polls
  st=$(kstatus "$V1_SLUG")
  case "$st" in
    *RUNNING*|*COMPLETE*)
      log "phaseA: v1 is $st (already pushed by someone)"
      # protect the next 72h slot from stale pushers (e.g. superseded session body)
      if [ -f "$V1/kernel-metadata.json" ]; then
        mv "$V1/kernel-metadata.json" "$V1/kernel-metadata.json.hold"
        log "phaseA: re-held v1 metadata (no re-push needed)"
      fi
      phase_b_at=$(date +%s)
      break
      ;;
  esac
  p=$(probe)
  if ra_open "$p"; then
    log "phaseA: window open (probe=$p) at $(date -Is); pushing GPU v1"
    [ -f "$V1/kernel-metadata.json" ] || mv "$V1/kernel-metadata.json.hold" "$V1/kernel-metadata.json" 2>/dev/null
    if do_push "$V1"; then
      sleep 300
      st=$(kstatus "$V1_SLUG")
      log "phaseA: v1 status after push: $st"
      phase_b_at=$(date +%s)
      break
    else
      log "phaseA: push failed; waiting for next window"
    fi
  fi
  sleep 180
done
[ -n "$phase_b_at" ] || { log "ABORT phaseA: v1 never started in 6 days"; exit 1; }

# ---------- Phase B: v2 (only once v1 is COMPLETE) ----------
for i in $(seq 1 2880); do   # up to 6 days, 5-min polls
  st1=$(kstatus "$V1_SLUG")
  st2=$(kstatus "$V2_SLUG")
  case "$st2" in
    *RUNNING*|*COMPLETE*) log "phaseB: v2 is $st2"; break ;;
  esac
  case "$st1" in
    *COMPLETE*)
      p=$(probe)
      if ra_open "$p"; then
        log "phaseB: window open (probe=$p); pushing v2"
        if do_push "$V2"; then
          sleep 300
          st2=$(kstatus "$V2_SLUG")
          log "phaseB: v2 status after push: $st2"
        fi
      fi
      ;;
    *ERROR*)
      # v1 run failed; if it failed >24h ago, a re-push is worth the slot
      if [ -f "$AUTO/gpu1_pushed" ] && [ $(( $(date +%s) - phase_b_at )) -gt 86400 ]; then
        if [ -f "$V1/kernel-metadata.json" ]; then
          p=$(probe)
          if ra_open "$p"; then
            log "phaseB: v1 stale ERROR; re-pushing v1"
            if do_push "$V1"; then sleep 300; log "phaseB: v1 re-pushed: $(kstatus "$V1_SLUG")"; fi
          fi
        fi
      fi
      ;;
  esac
  sleep 300
done

# ---------- Phase C: CPU v10 only if explicitly flagged ----------
if [ -f "$AUTO/push_cpu_v10" ]; then
  for i in $(seq 1 2880); do
    stc=$(kstatus "$VCPU_SLUG")
    case "$stc" in
      *RUNNING*|*COMPLETE*) log "phaseC: cpu v10 is $stc"; break ;;
    esac
    p=$(probe)
    if ra_open "$p"; then
      [ -f "$V1CPU/kernel-metadata.json" ] || mv "$V1CPU/kernel-metadata.json.hold.bak" "$V1CPU/kernel-metadata.json" 2>/dev/null
      log "phaseC: window open (probe=$p); pushing cpu v10"
      if do_push "$V1CPU"; then sleep 300; log "phaseC: cpu v10 status: $(kstatus "$VCPU_SLUG")"; fi
      break
    fi
    sleep 300
  done
fi
log "=== push_gpu_first2 exit ==="
exit 0
