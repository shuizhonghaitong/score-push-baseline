#!/bin/bash
cd /mnt/disk3/liuyutong_home/workspace/skill/score-push/runs/13-enveda-CASMI26-molecule-id-mass-spectra
while true; do
  if [ -s work/v465_pushed.txt ] || grep -qE 'READY_TO_SUBMIT|NO PUSH' work/auto_v465.log 2>/dev/null || grep -qE 'night_push done|PUSH_ERR' work/night_push.log 2>/dev/null; then
    {
      echo "=== EVENT $(date) ==="
      cat work/v465_pushed.txt 2>/dev/null
      grep -E 'READY_TO_SUBMIT|NO PUSH' work/auto_v465.log 2>/dev/null | tail -2
      tail -3 work/night_push.log 2>/dev/null
      echo "=== EVENT END ==="
    } >> work/event.log
    break
  fi
  sleep 120
done
