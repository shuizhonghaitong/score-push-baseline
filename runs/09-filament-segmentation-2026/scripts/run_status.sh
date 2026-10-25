#!/bin/bash
# One-shot run status for filament-segmentation-2026.
source /mnt/disk3/liuyutong_home/workspace/skill/score-push/runs/09-filament-segmentation-2026/env.sh
RUN=/mnt/disk3/liuyutong_home/workspace/skill/score-push/runs/09-filament-segmentation-2026
echo "== $(date -Is) =="
python3 - <<'PY'
import os, time, urllib.request
tok = os.environ.get('KAGGLE_API_TOKEN', '')
if os.path.isfile(tok): tok = open(tok).read().strip()
def probe(url):
    req = urllib.request.Request(url, data=b'', method='POST',
        headers={'Authorization': f'Bearer {tok}', 'Content-Type': 'application/grpc-web+proto', 'x-grpc-web': '1'})
    try:
        r = urllib.request.urlopen(req, timeout=30)
        return r.status, r.headers.get('Retry-After')
    except urllib.error.HTTPError as e:
        return e.code, e.headers.get('Retry-After')
    except Exception as e:
        return 'ERR', str(e)[:60]
for tag, url in (("SaveKernel", 'https://api.kaggle.com/v1/kernels.KernelsApiService/SaveKernel'),
                 ("Download  ", 'https://api.kaggle.com/v1/competitions.CompetitionApiService/DownloadDataFiles')):
    st, ra = probe(url)
    end = time.strftime('%m-%d %H:%M UTC', time.gmtime(time.time() + int(ra))) if str(ra).isdigit() else '?'
    print(f"{tag}: {st} RA={ra} -> {end}")
PY
for k in lilac1996/filament-magfilo-v1 lilac1996/filament-magfilo-v1-cpu-u-net lilac1996/filament-magfilo-v2; do
  echo -n "$k: "; kaggle kernels status "$k" 2>&1 | grep -oE 'KernelWorkerStatus\.[A-Z_]+' | head -1
done
python $KAGGLE_SKILL/scripts/kaggle_skill.py status filament-segmentation-2026 2>&1 | grep -E "rank|submissions|GPU this week|latest"
echo "held: $(ls src/notebooks/v1/kernel-metadata.json.hold src/notebooks/v1-cpu-fast2/kernel-metadata.json.hold 2>/dev/null | wc -l)/2 metadata held"
echo "submits today: $(cat state/auto/submits_$(date -u +%Y%m%d) 2>/dev/null || echo 0)"
echo "--- autopilot ---"; tail -3 $RUN/logs/autopilot.log
echo "--- unblock_watcher ---"; tail -3 $RUN/logs/unblock_watcher.log
echo "--- pipelines ---"; tail -2 $RUN/logs/watch.log | head -1 | cut -c1-260
