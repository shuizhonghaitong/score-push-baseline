#!/bin/bash
cd /mnt/disk3/liuyutong_home/workspace/skill/score-push/runs/13-enveda-CASMI26-molecule-id-mass-spectra
echo "TIME $(date +%H:%M:%S)"
echo "-- spectral CPU push: $(grep -E 'cycle [0-9]' work/push_retry.log | tail -1)  which=$(cat work/push_which.txt 2>/dev/null)"
echo "-- v465 GPU push:     $(grep -E 'cycle [0-9]' work/push_v465.log | tail -1)  pushed=$(cat work/v465_pushed.txt 2>/dev/null)"
echo "-- spectral PUSHED? $(test -s work/push_which.txt && echo YES || echo no)   v465 PUSHED? $(test -s work/v465_pushed.txt && echo YES || echo no)"
du -sh data/enveda-casmi26-molecule-id-mass-spectra/train.parquet.parts/ 2>/dev/null | awk '{print "-- train dl parts: "$1" / 3.0GB"}'
test -f data/enveda-casmi26-molecule-id-mass-spectra/train.parquet && echo "-- train.parquet ASSEMBLED: $(ls -la data/enveda-casmi26-molecule-id-mass-spectra/train.parquet | awk '{print $5}')"
tmux ls 2>/dev/null | sed 's/^/tmux: /'
