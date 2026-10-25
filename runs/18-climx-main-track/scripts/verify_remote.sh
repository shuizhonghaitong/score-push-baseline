#!/bin/bash
# usage: verify_remote.sh url size localfile  -> prints OK or MISMATCH/SHORT
set -u
source /mnt/disk3/liuyutong_home/workspace/skill/score-push/runs/18-climx-main-track/env.sh
url="$1"; want="$2"; loc="$3"
have=$(stat -c%s "$loc" 2>/dev/null || echo 0)
[ "$have" -eq "$want" ] || { echo "SHORT $loc ($have/$want)"; exit 1; }
curl -sSL --max-time 120 -r 0-99999 -o /tmp/vr_head.bin "$url" 2>/dev/null
curl -sSL --max-time 120 -r $((want-100000))-$((want-1)) -o /tmp/vr_tail.bin "$url" 2>/dev/null
head -c 100000 "$loc" > /tmp/vl_head.bin
tail -c 100000 "$loc" > /tmp/vl_tail.bin
if cmp -s /tmp/vr_head.bin /tmp/vl_head.bin && cmp -s /tmp/vr_tail.bin /tmp/vl_tail.bin; then
  echo "OK $loc"
else
  echo "MISMATCH $loc"
fi
