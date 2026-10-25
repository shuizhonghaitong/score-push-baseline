#!/bin/bash
# Robust downloader v2: routes via env.sh (unmetered proxy).
set -u
url="$1"; want="$2"; out="$3"
source /mnt/disk3/liuyutong_home/workspace/skill/score-push/runs/18-climx-main-track/env.sh
part="$out.part"
[ -f "$part" ] || : > "$part"
while :; do
  have=$(stat -c%s "$part" 2>/dev/null || echo 0)
  if [ "$have" -ge "$want" ]; then
    if [ "$have" -eq "$want" ]; then mv -f "$part" "$out"; echo "DONE $out"; exit 0; fi
    truncate -s "$want" "$part"
  fi
  curl -sSL --max-time 300 -C - -o "$part" "$url" 2>&1 | tail -1
  sleep 5
done
