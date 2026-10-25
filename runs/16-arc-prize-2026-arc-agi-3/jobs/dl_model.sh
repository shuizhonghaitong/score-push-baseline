#!/bin/bash
# Clean resumable chunked download of vrfai/Qwen3.6-27B-FP8/model.safetensors (35923194376 B).
# N=8 fixed-size chunks, each resumed via HTTP range. Assembles into assets/qwen36-27b-fp8/model.safetensors.
cd "$(dirname "$0")/.."
TOTAL=35923194376
N=8
CH=$((TOTAL / N))
OUT=assets/qwen36-27b-fp8
mkdir -p assets/.chunks "$OUT"
URL="https://hf-mirror.com/vrfai/Qwen3.6-27B-FP8/resolve/main/model.safetensors"
for i in $(seq 0 $((N-1))); do
  start=$((i*CH)); end=$(( (i+1)*CH - 1 )); [ $i -eq $((N-1)) ] && end=$((TOTAL-1))
  f="assets/.chunks/chunk_$i"
  want=$((end-start+1))
  if [ -f "$f" ] && [ "$(stat -c %s "$f")" = "$want" ]; then echo "chunk $i complete"; continue; fi
  r=0; [ -s "$f.part" ] && r=$(stat -c %s "$f.part")
  [ "$r" -gt "$start" ] || r=$start
  [ "$r" -ge "$want" ] && { echo "chunk $i part overlong, restarting"; r=$start; find assets/.chunks -name "chunk_$i.part" -delete; }
  echo "[$(date +%H:%M:%S)] chunk $i resume from $r to $((end-start+1))"
  for a in $(seq 1 50); do
    curl -sL --retry 0 -r "$r-$end" -o "$f.part" "$URL"
    cur=$(stat -c %s "$f.part" 2>/dev/null || echo $r)
    [ "$cur" -ge "$want" ] && break
    r=$cur; sleep 5
  done
  [ "$(stat -c %s "$f.part" 2>/dev/null || echo 0)" = "$want" ] && mv -f "$f.part" "$f"
  echo "[$(date +%H:%M:%S)] chunk $i -> $(stat -c %s "$f" 2>/dev/null || echo FAILED)"
done
# assemble
allok=1
for i in $(seq 0 $((N-1))); do
  f="assets/.chunks/chunk_$i"
  st2=$((i*CH)); en2=$(( (i+1)*CH - 1 )); [ $i -eq $((N-1)) ] && en2=$((TOTAL-1))
  want2=$((en2-st2+1))
  [ -f "$f" ] && [ "$(stat -c %s "$f")" = "$want2" ] || { echo "missing/short chunk $i"; allok=0; }
done
if [ "$allok" = 1 ]; then
  cat assets/.chunks/chunk_* > "$OUT/model.safetensors"
  sz=$(stat -c %s "$OUT/model.safetensors")
  echo "[$(date +%H:%M:%S)] assembled $sz / $TOTAL"
  [ "$sz" = "$TOTAL" ] && { echo "MODEL COMPLETE OK"; touch "$OUT/.model_ok"; }
fi
