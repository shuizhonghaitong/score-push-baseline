#!/bin/bash
cd /mnt/disk3/liuyutong_home/workspace/skill/score-push/runs/13-enveda-CASMI26-molecule-id-mass-spectra
export HTTP_PROXY=http://172.28.4.16:18999 HTTPS_PROXY=http://172.28.4.16:18999 http_proxy=http://172.28.4.16:18999 https_proxy=http://172.28.4.16:18999
SK=/mnt/disk3/liuyutong_home/workspace/skill/score-push/skills/kaggle/scripts/kaggle_skill.py
LOG=work/auto_v465.log
K=lilac1996/casmi26-v465-fork
echo "=== auto_v465 start $(date) ===" > $LOG
for i in $(seq 1 1200); do
  [ -s work/v465_pushed.txt ] && break
  sleep 60
done
[ -s work/v465_pushed.txt ] || { echo "NO PUSH after 20h" >> $LOG; exit 1; }
echo "push seen $(date)" >> $LOG
python3 $SK notebook-wait $K --out downloads/v465 --timeout 10800 --interval 60 >> $LOG 2>&1
rc=$?
echo "notebook-wait rc=$rc $(date)" >> $LOG
SUB=$(find downloads -name submission.csv 2>/dev/null | head -1)
echo "SUB=$SUB" >> $LOG
vrc=1
if [ -n "$SUB" ] && [ -f "$SUB" ]; then
  ./.venv/bin/python - "$SUB" >> $LOG 2>&1 <<'PY'
import sys, pandas as pd
s=pd.read_csv(sys.argv[1])
assert list(s.columns)==['molecule_id','smiles'], list(s.columns)
assert len(s)==400, len(s)
assert s.smiles.notna().all() and (s.smiles.astype(str).str.len()>0).all()
nreal=sum(1 for x in s.smiles if x!='CCO')
print("verify OK rows=",len(s)," non-CCO=",nreal)
PY
  vrc=$?
fi
VER=$(grep -oE "Kernel version [0-9]+ successfully pushed" work/push_v465.log | grep -oE "[0-9]+" | tail -1)
echo "READY_TO_SUBMIT VERSION=$VER SUB=$SUB verify_rc=$vrc $(date)" | tee -a $LOG
