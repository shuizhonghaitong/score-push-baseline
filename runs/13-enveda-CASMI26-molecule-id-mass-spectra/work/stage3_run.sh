#!/bin/bash
# Stage-3: run after the V465 scored rerun outputs + labels are available.
cd /mnt/disk3/liuyutong_home/workspace/skill/score-push/runs/13-enveda-CASMI26-molecule-id-mass-spectra
export PATH=/mnt/disk3/liuyutong_home/miniconda3/envs/kaggle/bin:$PATH
export HTTP_PROXY=http://172.28.4.16:18999 HTTPS_PROXY=http://172.28.4.16:18999
PY=./.venv/bin/python

echo "### fetch artifacts"
bash work/fetch_artifacts.sh | tail -8

ART=$(find work/v465_out -name 'e004_candidate_inputs.json' | head -1)
ARTDIR=$(dirname "$ART")
[ -n "$ARTDIR" ] || { echo "NO ARTIFACTS"; exit 1; }
echo "ARTDIR=$ARTDIR"
ls -la "$ARTDIR" | head -20

echo "### labels"
LAB=$(find work/labels_out -name 'labels.json' | head -1)
if [ -z "$LAB" ]; then
  echo "labels not ready; waiting (max 2h)"
  for i in $(seq 1 24); do
    LAB=$(find work/labels_out -name 'labels.json' | head -1)
    [ -n "$LAB" ] && break
    sleep 300
  done
fi
[ -n "$LAB" ] || { echo "NO LABELS"; exit 2; }
echo "LAB=$LAB"
$PY -c "import json; d=json.load(open('$LAB')); print('labels:', len(d))"

echo "### failure analysis (as-submitted config)"
$PY src/failure_analysis.py "$ARTDIR" "$LAB" 2>&1 | tail -25

echo "### rerank sweep (ice x gl)"
$PY src/rerank_local.py --artifacts "$ARTDIR" --labels "$LAB" --sweep 2>&1 | tail -30

echo "### window sweep around best"
$PY - <<PYEOF
import json, sys, itertools
sys.path.insert(0, 'src')
from rerank_local import evaluate
art = "$ARTDIR"
labels = json.load(open("$LAB"))
for w in (25, 40, 60, 80):
    for il in (0.5, 1.0, 1.5):
        for gl in (0.5, 1.0, 1.5):
            m, _ = evaluate(art, labels, il, gl, window=w)
            print(f'w={w} ice={il} gl={gl} mrr={m:.4f}', flush=True)
PYEOF
echo "### stage3 done"
