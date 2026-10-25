#!/bin/bash
set -u
RUN=/mnt/disk3/liuyutong_home/workspace/skill/score-push/runs/18-climx-main-track
mkdir -p "$RUN/ref/ClimX/src"
base_raw=https://raw.githubusercontent.com/IPL-UV/ClimX/main
base_html=https://github.com/IPL-UV/ClimX/tree/main
get_raw() { # path
  local rel="$1" dest="$RUN/ref/ClimX/$1"
  [ -s "$dest" ] && { echo "skip $rel"; return; }
  mkdir -p "$(dirname "$dest")"
  local tries=0
  while :; do
    if curl -sS --max-time 120 -o "$dest.part" "$base_raw/$rel" 2>"$dest.err"; then
      [ -s "$dest.part" ] && mv -f "$dest.part" "$dest" && { echo "OK $rel"; return; }
    fi
    tries=$((tries+1))
    [ $tries -ge 40 ] && { echo "GAVE_UP $rel"; return; }
    sleep 15
  done
}
get_html() { # path dest
  local rel="$1" dest="$2"
  local tries=0
  while :; do
    if curl -sS --max-time 60 -o "$dest" "$base_html/$rel" 2>>"$dest.err"; then
      [ -s "$dest" ] && { echo "HTML $rel"; return; }
    fi
    tries=$((tries+1))
    [ $tries -ge 40 ] && { echo "GAVE_UP html $rel"; return; }
    sleep 15
  done
}
for f in README.md train.py data_processing.py environment.yml rnNSE.py playground.ipynb visualizations.md \
         src/index_metadata.py src/indices_xclim.py src/metrics.py; do get_raw "$f"; done
for d in src/data_preprocessing src/models src/utils src/visualization docs; do
  get_html "$d" "/tmp/climx_$RANDOM.html"
done
ls -la "$RUN/ref/ClimX" /tmp/climx_*.html
