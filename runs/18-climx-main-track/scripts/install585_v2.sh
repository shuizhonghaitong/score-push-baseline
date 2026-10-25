#!/bin/bash
# Assemble ssp585 from mixed provenance parts, with overserving-tolerant snapshots.
set -u
RUN=/mnt/disk3/liuyutong_home/workspace/skill/score-push/runs/18-climx-main-track
cd $RUN
D=/tmp/hfchunks_585v2
SZ=4138588905
TOTAL=33108711240
deadline=$(( $(date +%s) + 252000 ))  # 70h
snap() { # $1 src $2 dst : copy first SZ bytes once src has >= SZ
  local s
  while :; do
    s=$(stat -c %s "$1" 2>/dev/null || echo 0)
    if [ "$s" -ge "$SZ" ]; then head -c $SZ "$1" > "$2" && echo "SNAP $(basename $2) from $(basename $1) (src size $s)" && return 0; fi
    if [ "$(date +%s)" -gt "$deadline" ]; then echo "DEADLINE $(basename $2) size=$s"; return 1; fi
    sleep 30
  done
}
ok=1
# straight parts (their own range-limited curls)
for i in 0 1 3 4 6 7; do snap $D/p$i $D/final_p$i || ok=0; done
# overserved mega-part: first SZ bytes = part 2 content
snap $D/p2 $D/final_p2 || ok=0
# p5 via p5alt (parallel range download)
snap $D/p5alt $D/final_p5 || ok=0
[ $ok -eq 1 ] || { echo "INSTALL585_V2_FAILED"; exit 1; }
cat $D/final_p0 $D/final_p1 $D/final_p2 $D/final_p3 $D/final_p4 $D/final_p5 $D/final_p6 $D/final_p7 > $D/full.nc
gs=$(stat -c %s $D/full.nc)
echo "ASSEMBLED size=$gs (expect $TOTAL)"
[ "$gs" = "$TOTAL" ] || { echo "SIZE_MISMATCH"; exit 1; }
# consistency: p3 own-content vs mega-part continuation (bytes SZ..2SZ of p2)
if cmp -n 100000000 <(tail -c +$((SZ+1)) $D/p2 | head -c 100000000) <(head -c 100000000 $D/p3); then
  echo "CROSSCHECK_P3_OK"
else
  echo "CROSSCHECK_P3_MISMATCH"
fi
# kaggle .part prefix cross-check
KP=data/NorESM2-MM_r1i1p1f1_with_forcing_train_ssp585.nc.part
if [ -f "$KP" ]; then
  kps=$(stat -c %s "$KP"); chk=$(( kps > 20000000000 ? 20000000000 : kps ))
  if [ "$chk" -ge 8000000000 ]; then
    cmp -n $chk <(head -c $chk $D/full.nc) <(head -c $chk "$KP") && echo "KAGGLE_PART_MATCH $chk" || { echo "KAGGLE_PART_MISMATCH $chk"; exit 1; }
  fi
fi
CANON=data/NorESM2-MM_r1i1p1f1_with_forcing_train_ssp585.nc
mv "$CANON" "$CANON.corrupt.bak"
cp $D/full.nc "$CANON"
echo "INSTALL585_V2_DONE size=$(stat -c %s $CANON)"
