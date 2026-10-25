#!/bin/bash
# Dedicated SaveKernel-window pusher (one real push pins the 72h window).
# Window 1 (10-26 ~07:46 local): preempt the weaker CPU kernel, then push GPU v1
#   with 1-minute probing to beat the 9-10-minute-cycle pushers (cpu_v10.sh,
#   autopilot backup role).
# Window 2 (10-29 ~07:46): safety net for v2 (autopilot phase C also pushes it).
# Window 3 (11-01 ~07:46): push CPU v10 only if $AUTO/push_cpu_v10 flag exists.
RUN=/mnt/disk3/liuyutong_home/workspace/skill/score-push/runs/09-filament-segmentation-2026
source $RUN/env.sh
LOG=$RUN/logs/push_gpu_first.log
NB_PUSH=$KAGGLE_SKILL/modules/notebooks/scripts/notebook_push.py
V1=$RUN/src/notebooks/v1
V1CPU=$RUN/src/notebooks/v1-cpu-fast2
V2=$RUN/src/notebooks/v2
AUTO=$RUN/state/auto
mkdir -p "$AUTO"
log() { echo "[$(date -Is)] $*" >> $LOG; }

sleep_until() {
  local target; target=$(date -d "$1" +%s)
  local now; now=$(date +%s)
  if [ $now -lt $target ]; then log "sleeping until $1 ($((target - now))s)"; sleep $((target - now)); fi
}

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

log "=== push_gpu_first start ==="

# ---------------- Window 1: GPU v1 (10-26) ----------------
sleep_until "2026-10-26 07:36:00"
log "window1: (body v2 10-23) superseded by push_gpu_first2.sh; this session exits"
exit 0
