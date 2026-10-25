#!/bin/bash
# Watches the SaveKernel 429 Retry-After countdown. Our pipelines' pushes are
# held (metadata .hold) because every 429'd SaveKernel attempt extends the
# account penalty window. When the window is close (RA < 6h or no longer 429),
# unhold the metadata so the running pipelines push immediately.
source /mnt/disk3/liuyutong_home/workspace/skill/score-push/runs/09-filament-segmentation-2026/env.sh
RUN=/mnt/disk3/liuyutong_home/workspace/skill/score-push/runs/09-filament-segmentation-2026
LOG=$RUN/logs/unblock_watcher2.log
echo "=== unblock_watcher start $(date -Is) ===" >> $LOG
for i in $(seq 1 480); do
  {
    echo "=== $(date -Is) cycle $i ==="
    python3 - <<'PY'
import os, time, urllib.request
tok = os.environ.get('KAGGLE_API_TOKEN', '')
if os.path.isfile(tok): tok = open(tok).read().strip()
req = urllib.request.Request('https://api.kaggle.com/v1/kernels.KernelsApiService/SaveKernel',
    data=b'', method='POST',
    headers={'Authorization': f'Bearer {tok}', 'Content-Type': 'application/grpc-web+proto', 'x-grpc-web': '1'})
try:
    r = urllib.request.urlopen(req, timeout=30)
    print('PROBE status', r.status, 'RA', r.headers.get('Retry-After'))
except urllib.error.HTTPError as e:
    ra = e.headers.get('Retry-After')
    end = time.strftime('%m-%d %H:%M UTC', time.gmtime(time.time() + int(ra))) if ra else '?'
    print('PROBE status', e.code, 'RA', ra, '->', end)
except Exception as e:
    print('PROBE err', type(e).__name__, str(e)[:120])
PY
  } >> $LOG 2>&1
  # sibling run 07 pusher activity (last push attempt within 25 min => window pinned)
  recent=$(find /mnt/disk3/liuyutong_home/workspace/skill/score-push/runs/07-2026-ieee-big-data-traffic-flow-bench/logs -name "queue_*.log" -mmin -25 2>/dev/null | wc -l)
  echo "sibling07_recent_logs=$recent" >> $LOG
  h1=$RUN/src/notebooks/v1/kernel-metadata.json.hold
  if [ -f "$h1" ]; then
    ra=$(tail -8 $LOG | grep -oE "PROBE status 429 RA [0-9]+" | tail -1 | grep -oE "[0-9]+$")
    open=$(tail -4 $LOG | grep -cE "PROBE status (200|400|401|403|404)")
    if [ "${open:-0}" -ge 1 ] || { [ -n "$ra" ] && [ "$ra" -lt 21600 ]; }; then
      mv "$h1" $RUN/src/notebooks/v1/kernel-metadata.json
      mv $RUN/src/notebooks/v1-cpu-fast2/kernel-metadata.json.hold $RUN/src/notebooks/v1-cpu-fast2/kernel-metadata.json
      echo "*** UNHOLDED metadata at RA=$ra (window open or <6h); pipelines will push" >> $LOG
    fi
  fi
  sleep 600
done
echo "=== unblock_watcher exit $(date -Is) ===" >> $LOG
