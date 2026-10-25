#!/bin/bash
# Fetch py3.12 wheels from tsinghua simple index into models/pypi_cache
set -u
cd /mnt/disk3/liuyutong_home/workspace/skill/score-push/runs/14-gemma-4-developer-agent
CACHE=models/pypi_cache
mkdir -p "$CACHE"
BASE=https://pypi.tuna.tsinghua.edu.cn
for pkg in "$@"; do
  html=$(curl -s --max-time 30 "$BASE/simple/$pkg/") || { echo "FAIL index $pkg"; continue; }
  urls=$(echo "$html" | grep -oE 'href="[^"]+"' | sed 's/href="//;s/"//' | grep -E "x86_64.*\.whl|py3-none-any.*\.whl|py2.py3-none-any.*\.whl")
  url=$(echo "$urls" | grep -E "cp312-cp312-manylinux" | sort | tail -1)
  if [ -z "$url" ]; then
    url=$(echo "$urls" | grep -E "abi3-manylinux" | sort | tail -1)
  fi
  if [ -z "$url" ]; then
    url=$(echo "$urls" | grep -E "py2?\.?py3-none-any" | sort | tail -1)
  fi
  if [ -z "$url" ]; then echo "NOURL $pkg"; continue; fi
  case "$url" in
    http*) full="$url";;
    /*) full="$BASE$url";;
    *) full="$BASE/packages/${url%%#*}";;
  esac
  full="${full%%#*}"
  fn=$(basename "$full")
  [ -f "$CACHE/$fn" ] && { echo "skip $fn"; continue; }
  size=$(curl -sI "$full" | grep -i content-length | tr -dc '0-9')
  echo "get $fn ${size:-?}"
  if [ "${size:-0}" -gt 20000000 ]; then
    n=4; chunk=$(( (size + n - 1) / n )); parts=""
    for i in $(seq 0 $((n-1))); do
      s=$(( i * chunk )); e=$(( s + chunk - 1 )); [ $e -ge $size ] && e=$((size-1))
      p="$CACHE/.part_$fn.$i"
      curl -s --max-time 1200 -r "$s-$e" -o "$p" "$full" &
      parts="$parts $p"
    done
    wait
    cat $parts > "$CACHE/$fn" && mv $parts /tmp/ 2>/dev/null
    echo "done $fn"
  else
    curl -s --max-time 300 -o "$CACHE/$fn" "$full" && echo "done $fn"
  fi
done
echo ALL_FETCH_DONE
