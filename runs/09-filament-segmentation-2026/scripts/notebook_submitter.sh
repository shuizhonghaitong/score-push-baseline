#!/bin/bash
# Notebook submissions for COMPLETED kernels (kaggle.com API path, no GCS
# upload). Version numbers come from push receipts ("Kernel version N
# successfully pushed ... code/<slug>") in the push logs.
RUN=/mnt/disk3/liuyutong_home/workspace/skill/score-push/runs/09-filament-segmentation-2026
source $RUN/env.sh
cd $RUN
LOG=$RUN/logs/notebook_submitter.log
SK=$KAGGLE_SKILL/scripts/kaggle_skill.py
COMP=filament-segmentation-2026
SUBS=$RUN/state/auto/nb_submitted.txt
touch "$SUBS"
log() { echo "[$(date -Is)] $*" >> $LOG; }

kstatus() { kaggle kernels status "$1" 2>&1 | grep -oE 'KernelWorkerStatus\.[A-Z_]+' | head -1; }

receipts() { # "slug<TAB>version" lines from all push logs
  cat $RUN/logs/push_gpu_first2.log $RUN/logs/vpush.log $RUN/logs/gpu_pipeline.log \
      $RUN/logs/autopilot.log $RUN/logs/push_gpu_first.log 2>/dev/null \
  | grep -oE "Kernel version [0-9]+ successfully pushed\.  Please check progress at https://www.kaggle.com/code/[a-z0-9-]+/[a-z0-9-]+" \
  | sed -E 's/Kernel version ([0-9]+) successfully pushed\.  Please check progress at https:\/\/www\.kaggle\.com\/code\/([^ ]+)$/\2\t\1/'
}

log "=== notebook_submitter start ==="
for i in $(seq 1 4320); do   # 10 min polls, up to 30 days
  while IFS=$'\t' read -r slug ver; do
    [ -n "$slug" ] || continue
    case "$slug" in
      lilac1996/filament-magfilo-v1) msg="v1 gpu unet12+mrcnn10 4flip TTA (auto)";;
      lilac1996/filament-magfilo-v2) msg="v2 crop refiner TTA (auto)";;
      *) continue;;
    esac
    grep -q "^$slug $ver$" "$SUBS" && continue
    st=$(kstatus "$slug")
    case "$st" in
      *COMPLETE*)
        log "notebook-submit $slug v$ver (status $st)"
        if python3 $SK submit $COMP --notebook "$slug" --version "$ver" -m "$msg" --yes >> $LOG 2>&1; then
          echo "$slug $ver" >> "$SUBS"
          log "NB SUBMITTED $slug v$ver"
        else
          log "NB submit FAILED $slug v$ver (retry next cycle)"
        fi
        ;;
      *RUNNING*) : ;;
      *) log "$slug v$ver status=$st (not submittable yet)";;
    esac
  done < <(receipts | sort -u)
  sleep 600
done
log "=== notebook_submitter exit ==="
exit 0
