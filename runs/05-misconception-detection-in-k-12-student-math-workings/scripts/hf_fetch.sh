#!/bin/bash
RUN=/mnt/disk3/liuyutong_home/workspace/skill/score-push/runs/05-misconception-detection-in-k-12-student-math-workings
cd $RUN
LOG=$RUN/logs/hf_fetch.log
say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $LOG; }
export HF_ENDPOINT=https://hf-mirror.com
$RUN/.venv/bin/python - <<'PY'
import os, time, sys
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
from huggingface_hub import hf_hub_download
dst = hf_hub_download("colearnlab/misconception-detection", "competition_data_v2.zip", repo_type="dataset")
print("DOWNLOADED", dst, os.path.getsize(dst), flush=True)
PY
rc=$?
if [ $rc -ne 0 ]; then say "HF download rc=$rc (will retry)"; sleep 120; exit 3; fi
say "zip ready: $(ls -la $RUN/data/competition_data_v2.zip 2>/dev/null | awk '{print $5}') or $(ls -la $(dirname $(find ~/.cache/huggingface -name 'competition_data_v2.zip' 2>/dev/null | head -1)) 2>/dev/null | wc -l)"
# extract images into data/images (train, test, questions, unlabelled)
Z=$(find ~/.cache/huggingface -name 'competition_data_v2.zip' 2>/dev/null | head -1)
cp -n "$Z" $RUN/data/competition_data_v2.zip 2>/dev/null
mkdir -p /tmp/miscc_v2 && (cd /tmp/miscc_v2 && unzip -o -q -d . $RUN/data/competition_data_v2.zip "competition_data/images/train/*" "competition_data/images/test/*" "competition_data/images/questions/*" "competition_data/images/unlabelled/*" 2>/dev/null || unzip -o -q $RUN/data/competition_data_v2.zip -d /tmp/miscc_v2)
SRC=$(find /tmp/miscc_v2 -maxdepth 3 -type d -name images | head -1)
if [ -n "$SRC" ]; then
  for sub in train test questions unlabelled; do
    mkdir -p $RUN/data/images/$sub
    cp -rn $SRC/$sub/. $RUN/data/images/$sub/ 2>/dev/null
  done
  say "images extracted"
fi
say "totals: train=$(ls $RUN/data/images/train 2>/dev/null | wc -l) test=$(ls $RUN/data/images/test 2>/dev/null | wc -l) questions=$(ls $RUN/data/images/questions 2>/dev/null | wc -l) unlabelled=$(ls $RUN/data/images/unlabelled 2>/dev/null | wc -l)"
say "HF_FETCH OK"
