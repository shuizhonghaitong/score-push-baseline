#!/bin/bash
# 3-fold CV for V6 with retry on intermittent crashes
cd /mnt/disk3/liuyutong_home/workspace/skill/score-push/runs/18-climx-main-track
source env.sh >/dev/null 2>&1
export OMP_NUM_THREADS=2
for ho in ssp370 ssp585 ssp126; do
  for i in $(seq 1 20); do
    timeout 2500 ./venv3/bin/python -u scripts/train_v6.py --holdout $ho > /tmp/v6_${ho}.log 2>&1
    if grep -q "V6 CV" /tmp/v6_${ho}.log; then
      echo "=== $ho OK (attempt $i) ==="
      tail -20 /tmp/v6_${ho}.log
      break
    fi
    echo "$ho attempt $i failed, retrying..."
    sleep 3
  done
  if ! grep -q "V6 CV" /tmp/v6_${ho}.log; then echo "$ho FAILED after 20 attempts"; tail -5 /tmp/v6_${ho}.log; fi
done
