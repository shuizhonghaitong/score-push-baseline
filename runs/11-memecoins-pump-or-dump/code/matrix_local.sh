#!/bin/bash
# Local walk-forward matrix over precomputed features + local labels.
# Usage:
#   code/matrix_local.sh DATA_DIR FEATS OUT_ROOT "tag:KEY=VAL KEY=VAL" [more...]
# DATA_DIR: dir with train.csv/test.csv/... (labels)
# FEATS:    features.csv|parquet (mint index col 0)
set -u
cd "$(dirname "$0")/.."
PY=./venv/bin/python
DATA_DIR=$1; shift
FEATS=$1; shift
OUT_ROOT=$1; shift
mkdir -p "$OUT_ROOT"
SUMMARY=$OUT_ROOT/summary.csv
echo "tag,n_days,n_trades_mean,pnl_ev_sum,pnl_ev_min,pnl_ev_lo,pnl_ev_hi,c10,c20,c31,top1_5,best1_5,cascade,avg_auc5_raw" > "$SUMMARY"
for spec in "$@"; do
  tag=${spec%%:*}; rest=${spec#*:}
  OUT=$OUT_ROOT/$tag
  echo "=== $tag : $rest"
  ENVSTR=""
  for kv in $rest; do ENVSTR="$ENVSTR $kv"; done
  env MODE=wf OUT_DIR=$OUT DATA_DIR=$DATA_DIR RESUME_FEATS=$FEATS NJOBS=${NJOBS:-6} $ENVSTR \
    $PY code/pipeline.py > "$OUT.log" 2>&1 || { echo "  FAILED (see $OUT.log)"; continue; }
  python3 - "$OUT/wf_results.csv" "$tag" "$SUMMARY" <<'PY'
import sys, pandas as pd, numpy as np
f, tag, summary = sys.argv[1:4]
try:
    df = pd.read_csv(f)
except Exception:
    print("  no wf_results"); sys.exit(0)
if not len(df):
    print("  empty wf"); sys.exit(0)
row = dict(tag=tag, n_days=len(df), n_trades_mean=round(df.n_trades.mean(),1),
           pnl_ev_sum=round(df.pnl_ev.sum(),2), pnl_ev_min=round(df.pnl_ev.min(),2),
           pnl_ev_lo=round(df.get('pnl_ev_lo',0).sum(),2), pnl_ev_hi=round(df.get('pnl_ev_hi',0).sum(),2),
           c10=round(df.get('pnl_ev_c10',0).sum(),2), c20=round(df.get('pnl_ev_c20',0).sum(),2),
           c31=round(df.get('pnl_ev_c31',0).sum(),2), top1_5=round(df.get('pnl_top1_5',0).sum(),2),
           best1_5=round(df.get('pnl_best1_5',0).sum(),2), cascade=round(df.get('pnl_cascade_top1_5',0).sum(),2),
           avg_auc5_raw=round(df.get('auc5_raw',0).mean(),4))
print("  ", {k: row[k] for k in list(row)[1:12]})
line = ",".join(str(row[k]) for k in ["tag","n_days","n_trades_mean","pnl_ev_sum","pnl_ev_min",
    "pnl_ev_lo","pnl_ev_hi","c10","c20","c31","top1_5","best1_5","cascade","avg_auc5_raw"])
open(summary, "a").write(line + "\n")
PY
done
echo "=== SUMMARY ==="
python3 -c "
import pandas as pd
df = pd.read_csv('$SUMMARY')
print(df.sort_values('pnl_ev_sum', ascending=False).to_string(index=False))
"
