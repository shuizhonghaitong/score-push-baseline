#!/bin/bash
cd /mnt/disk3/liuyutong_home/workspace/skill/score-push/runs/13-enveda-CASMI26-molecule-id-mass-spectra
while true; do
  {
    echo "--- $(date '+%F %T')"
    echo "train2=$(du -sh data/enveda-casmi26-molecule-id-mass-spectra/train2.parquet.parts 2>/dev/null | cut -f1) ice=$(du -sh data/local/iceberg 2>/dev/null | cut -f1) gl=$(du -sh data/local/glacier 2>/dev/null | cut -f1)"
    [ -s work/v465_pushed.txt ] && echo "PUSHED: $(cat work/v465_pushed.txt)"
    tail -1 work/night_push.log 2>/dev/null | sed 's/^/np: /'
    tail -1 work/auto_v465.log 2>/dev/null | sed 's/^/auto: /'
    grep -oE 'READY_TO_SUBMIT.*' work/auto_v465.log 2>/dev/null | tail -1
    tail -1 work/train_dl2clean.log 2>/dev/null | sed 's/^/dl: /'
    tail -1 work/dl_sims.log 2>/dev/null | sed 's/^/sims: /'
  } >> work/status2.log
  sleep 600
done
