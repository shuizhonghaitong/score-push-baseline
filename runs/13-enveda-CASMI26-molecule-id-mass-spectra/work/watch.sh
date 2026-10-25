#!/bin/bash
cd /mnt/disk3/liuyutong_home/workspace/skill/score-push/runs/13-enveda-CASMI26-molecule-id-mass-spectra
while true; do
  {
    echo "--- $(date '+%F %T')"
    for f in v465_pushed nb_sub_pushed nb_sub_ready nb_sub_submitted nb_sub_score v465_submitted v465_score labels_ready; do
      [ -f "work/$f.txt" ] && echo "FLAG $f: $(head -1 work/$f.txt)"
    done
    echo "np: $(tail -1 work/night_push.log 2>/dev/null)"
    echo "tb: $(tail -1 work/track_b.log 2>/dev/null)"
    echo "lbl: $(tail -1 work/labels.log 2>/dev/null)"
    echo "rdkit_dl: $(tail -1 work/dl_rdkit.log 2>/dev/null)"
    echo "train: $(du -sh data/enveda-casmi26-molecule-id-mass-spectra/train.parquet 2>/dev/null | cut -f1) parts:$(du -sh data/enveda-casmi26-molecule-id-mass-spectra/train2.parquet.parts 2>/dev/null | cut -f1)"
  } >> work/watch.log
  sleep 600
done
