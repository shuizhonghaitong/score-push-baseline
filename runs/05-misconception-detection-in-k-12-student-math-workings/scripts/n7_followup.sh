#!/bin/bash
# Self-sufficient followup:
#  - when n7 push is ACCEPTED (rc=0) -> detach-launch the n8 pusher (once)
#  - when n7 is COMPLETE + FETCHED   -> build & (slot-permitting) submit v6a/v6b
export HTTP_PROXY=http://172.28.4.16:18999 HTTPS_PROXY=http://172.28.4.16:18999 http_proxy=http://172.28.4.16:18999 https_proxy=http://172.28.4.16:18999
export PATH=/mnt/disk3/liuyutong_home/miniconda3/envs/kaggle/bin:$PATH
RUN=/mnt/disk3/liuyutong_home/workspace/skill/score-push/runs/05-misconception-detection-in-k-12-student-math-workings
cd $RUN
LOG=$RUN/logs/n7_followup.log
say(){ echo "[$(date '+%m-%d %H:%M:%S')] $*" | tee -a $LOG; }
PY=$RUN/.venv/bin/python
SK=/mnt/disk3/liuyutong_home/workspace/skill/score-push/skills/kaggle/scripts/kaggle_skill.py
START=$(date +%s)
N8_LAUNCHED=0

# wait for either accepted or complete marker
while true; do
  if [ $(( $(date +%s) - START )) -gt 259200 ]; then say "GIVEUP waiting 72h"; exit 1; fi
  if grep -qE "PUSH rc=0" logs/push_n7_v4.log 2>/dev/null && [ $N8_LAUNCHED -eq 0 ]; then
    say "n7 push accepted -> launching n8 pusher (detached)"
    setsid bash $RUN/scripts/push_n8.sh > $RUN/logs/push_n8.out 2>&1 < /dev/null &
    echo $! > $RUN/logs/push_n8.pid
    N8_LAUNCHED=1
  fi
  if grep -qE "N7_V[34] COMPLETE_TAGGED_FETCHED|N7_V[34] COMPLETE \\\\+ FETCHED" logs/push_n7_v3.log logs/push_n7_v4.log 2>/dev/null; then
    say "n7 COMPLETE + FETCHED marker seen"; break
  fi
  if [ -f outputs/ds_lora/vlm_diag_test.csv ] && [ $(stat -c%Y outputs/ds_lora/vlm_diag_test.csv) -gt $(date -d "2026-10-22 20:00" +%s) ]; then
    say "fresh diag csv detected"; break
  fi
  sleep 180
done

# report in-sample quality
if [ -f logs/n7_v4_kernel.log ]; then
  say "kernel log highlights:"
  grep -E "N7_TAG|final loss|LoRA train MAP|DATASET|ABORT|BUDGET" logs/n7_v4_kernel.log | tail -12 | sed 's/^/    | /' | tee -a $LOG
fi

left=$(timeout 60 python3 $SK status misconception-detection-in-k-12-student-math-workings 2>&1 | grep -o "[0-9]* left of [0-9]* a day" | grep -o "^[0-9]*")
say "slots left today: ${left:-unknown}"

if [ "${left:-0}" -ge 1 ] 2>/dev/null && [ -f outputs/ds_lora/vlm_diag_test.csv ]; then
  $PY scripts/build_v6_lora.py --mode pure --diag-te outputs/ds_lora/vlm_diag_test.csv --out outputs/submission_v6a_purelora.csv || say "v6a build FAILED"
  if [ -f outputs/submission_v6a_purelora.csv ]; then
    $PY scripts/submit_manual.py outputs/submission_v6a_purelora.csv "v6a: pure n7-LoRA top3 (448px ep1)" | tee -a $LOG
  fi
fi
if [ "${left:-0}" -ge 2 ] 2>/dev/null && [ -f outputs/ds_lora/vlm_diag_test.csv ]; then
  TR=outputs/ds_lora/vlm_diag_train.csv
  if [ ! -f "$TR" ]; then TR=outputs/ds_diag/vlm_diag_train.csv; fi
  $PY scripts/build_v6_lora.py --mode blend --diag-tr "$TR" --diag-te outputs/ds_lora/vlm_diag_test.csv --w-diag 0.4 --w-knn 0.3 --dy 1.5 --dn 0.2 --out outputs/submission_v6b_blend.csv || say "v6b build FAILED"
  if [ -f outputs/submission_v6b_blend.csv ]; then
    $PY scripts/submit_manual.py outputs/submission_v6b_blend.csv "v6b: v4b recipe + 0.4*n7-LoRA diag" | tee -a $LOG
  fi
fi
say "N7_FOLLOWUP done (submitted or skipped for slots)"
