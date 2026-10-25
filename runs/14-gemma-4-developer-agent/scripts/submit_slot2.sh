#!/bin/bash
# Submit the best-evidence variant for the 10-23 slot.
# Usage: bash scripts/submit_slot2.sh [variant]   # variant in v1 v2 t04 v2t (default: auto)
set -e
cd /mnt/disk3/liuyutong_home/workspace/skill/score-push/runs/14-gemma-4-developer-agent
SK=/mnt/disk3/liuyutong_home/workspace/skill/score-push/skills/kaggle/scripts/kaggle_skill.py
PY=/mnt/disk3/liuyutong_home/miniconda3/envs/kaggle/bin/python

auto_pick() {
  # Kaggle A/B output?
  if [ -f downloads/kaggle_ab/ab_summary.csv ]; then
    python3 - <<'EOF'
import csv
rows = list(csv.DictReader(open('downloads/kaggle_ab/ab_summary.csv')))
v1 = sum(int(r['v1']) for r in rows if r['v1'] in '01')
v2 = sum(int(r['v2']) for r in rows if r['v2'] in '01')
print('v2' if v2 > v1 else ('v1' if v1 > v2 else 'v2'))
EOF
    return
  fi
  # local A/B output? (require both runs complete: 16 dev tasks; wait <=45m)
  if [ -f results/ab_v1/progress.jsonl ]; then
    /mnt/disk3/liuyutong_home/miniconda3/envs/kaggle/bin/python /mnt/disk3/liuyutong_home/workspace/skill/score-push/runs/14-gemma-4-developer-agent/scripts/ab_pick.py
    return
  fi
  echo "v2"
}

V=${1:-$(auto_pick)}
echo "submitting variant: $V"
case $V in
  v1) ZIP=submissions/submission_v1.zip;;
  v2) ZIP=submissions/submission_v2.zip;;
  t04) ZIP=submissions/submission_t04.zip;;
  v2t) ZIP=submissions/submission_v2t.zip;;
  *) echo "bad variant $V"; exit 1;;
esac
cp "$ZIP" submission.zip
echo "sha256: $(sha256sum submission.zip | cut -d' ' -f1) (variant $V)"
$PY $SK submit gemma-4-developer-agent submission.zip -m "run14 $V: $(date -u +%F) auto-selection" --yes
