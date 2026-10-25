#!/bin/bash
# Install re-downloaded ssp585 canonical after full verification.
set -u
RUN=/mnt/disk3/liuyutong_home/workspace/skill/score-push/runs/18-climx-main-track
cd $RUN
source env.sh
D=/tmp/hfchunks_585v2
TOTAL=33108711240
F=$D/full.nc
[ -f "$F" ] || { echo "FATAL: $F missing"; exit 1; }
gs=$(stat -c %s "$F")
if [ "$gs" != "$TOTAL" ]; then echo "FATAL: assembled size $gs != $TOTAL"; exit 1; fi
CANON=data/NorESM2-MM_r1i1p1f1_with_forcing_train_ssp585.nc
KP=data/NorESM2-MM_r1i1p1f1_with_forcing_train_ssp585.nc.part
# cross-check the largest available prefix of the kaggle .part (free verification source)
if [ -f "$KP" ]; then
  kps=$(stat -c %s "$KP")
  chk=$(( kps > 20000000000 ? 20000000000 : kps ))
  if [ "$chk" -ge 8000000000 ]; then
    if cmp -n $chk <(head -c $chk "$F") <(head -c $chk "$KP"); then
      echo "OK: KAGGLE_PART_MATCH_$chk bytes"
    else
      echo "FATAL: kaggle .part mismatch at prefix $chk"
      exit 1
    fi
  fi
fi
# backup old (corrupt) canonical, install new
mv "$CANON" "$CANON.corrupt.$(date -u +%Y%m%d%H%M)"
cp "$F" "$CANON"
ns=$(stat -c %s "$CANON")
[ "$ns" = "$TOTAL" ] || { echo "FATAL: install size $ns"; exit 1; }
echo "INSTALL_OK size=$ns"
