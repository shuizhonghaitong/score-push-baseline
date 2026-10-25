#!/bin/bash
source /mnt/disk3/liuyutong_home/workspace/skill/score-push/runs/09-filament-segmentation-2026/env.sh
cd /mnt/disk3/liuyutong_home/workspace/skill/score-push/runs/09-filament-segmentation-2026
ZIP=data/magfilo-v1-0-segmentation-of-solar-filaments.zip
rm -f "$ZIP" data/*.zip.kaggle-partial
for i in $(seq 1 30); do
  echo "=== attempt $i $(date -Is) ==="
  kaggle datasets download esairlab/magfilo-v1-0-segmentation-of-solar-filaments -p data/ -q 2>&1 | tail -3
  SZ=$(stat -c%s "$ZIP" 2>/dev/null || echo 0)
  echo "size=$SZ"
  if unzip -t "$ZIP" >/dev/null 2>&1; then
    echo "DOWNLOAD COMPLETE attempt=$i size=$SZ"
    break
  fi
  echo "incomplete, wait 60s"
  sleep 60
done
if unzip -t "$ZIP" >/dev/null 2>&1; then
  unzip -q -o "$ZIP" -d data/extracted && echo "EXTRACT OK"
  find data/extracted -maxdepth 2 | head -20
else
  echo "DOWNLOAD FAILED after 30 attempts"
fi
echo "ALL DONE $(date -Is)"
