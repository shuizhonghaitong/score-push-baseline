#!/bin/bash
# Conductor (run 06, agent-2): supervises the old-namespace watchers from this session.
# Read-centric: never kills old-namespace procs; launches only: retry2 push, manual C2 pushes,
# manual G push (if v2 wedged), dir restore (if v1 dead / overdue), c2_harvest.
cd /mnt/disk3/liuyutong_home/workspace/skill/score-push/runs/06-name-that-llm || exit 1
source env.sh
S=/mnt/disk3/liuyutong_home/workspace/skill/score-push/skills/kaggle/scripts/kaggle_skill.py
PY=./venv/bin/python
LOG=work/conductor.log
log() { echo "$(date '+%F %T') $*" | tee -a "$LOG"; }
once() { [ -f "work/.cond_$1" ] || { touch "work/.cond_$1"; return 0; }; return 1; }  # 0 first time only

kernel_exists() { timeout 90 kaggle kernels status "$1" 2>/dev/null | grep -q "KernelWorkerStatus"; }
kernel_state() { timeout 90 kaggle kernels status "$1" 2>&1 | grep -o 'KernelWorkerStatus\.[A-Z_]*' | tail -1; }

log "=== conductor started (pid $$) $(date -u '+%FT%TZ') ==="
T1600=$(date -d "2026-10-23 16:00" +%s)

while true; do
  now=$(date +%s)

  # ---------- 1. C2 push chain ----------
  if [ ! -f work/.c2_pushed ]; then
    if grep -q "GAVE UP" work/push_c2_loop.log 2>/dev/null && once retry2_launched; then
      log "loop1 exhausted unpushed -> launching push_c2_retry2 (30min x7)"
      nohup bash code/push_c2_retry2.sh >> work/push_c2_loop2.log 2>&1 &
    fi
    if grep -q "GAVE UP" work/push_c2_loop2.log 2>/dev/null; then
      last=$(cat work/.c2_manual_last 2>/dev/null || echo 0)
      if [ $(( now - last )) -gt 2700 ]; then
        echo $now > work/.c2_manual_last
        out=$(timeout 600 python3 $S notebook-push notebookC2 --yes 2>&1)
        ver=$(echo "$out" | grep -oE 'KernelVersion: [0-9]+' | grep -oE '[0-9]+' | tail -1)
        if echo "$out" | grep -qE "429|Too Many|ProxyError|Max retries"; then
          log "manual C2 push: 429, retry in 45m"
        elif [ -n "$ver" ]; then
          touch work/.c2_pushed; echo "$ver" > work/pushver_ntl-run06-v3-cpu
          log "manual C2 push OK ver=$ver"
        else
          log "manual C2 push: unknown result: $(echo "$out" | tail -2 | tr '\n' ' ')"
        fi
      fi
    fi
  else
    st=$(kernel_state lilac1996/ntl-run06-v3-cpu)
    case "$st" in
      *COMPLETE*)
        if once c2_harvest; then log "C2 COMPLETE -> running c2_harvest.sh"; nohup bash code/c2_harvest.sh >> work/c2_harvest.log 2>&1 & fi ;;
      *CANCEL*|*ERROR*|*FAILED*) if once c2_fail; then log "ALERT C2 terminal non-complete: $st"; fi ;;
      "") if once c2_status_alert; then log "ALERT C2 pushed but status unparseable (404/403?) - check manually"; fi ;;
    esac
  fi

  # ---------- 2. watcher liveness (stale-log alerts) ----------
  for spec in "work/fe_b_early2.log:1500:v2" "work/fe_b_early.log:1500:v1" "work/sup_gap.log:1500:supgap"; do
    f=${spec%%:*}; rest=${spec#*:}; max=${rest%%:*}; name=${rest#*:}
    [ -f "$f" ] || continue
    age=$(( now - $(stat -c %Y "$f") ))
    if [ $age -gt $max ] && once "stale_${name}_$(date -u +%Y%m%d_%H)"; then
      log "ALERT $name log $f stale ${age}s (> ${max}s)"
    fi
  done

  # ---------- 3. 16:00 events ----------
  if [ $now -gt $T1600 ]; then
    # 3a. verify v2 pushed G (allow 25 min for its 15-min poll cycle)
    if [ $now -gt $((T1600 + 1500)) ] && ! grep -q "PUSH notebookG" work/fe_b_early2.log 2>/dev/null && once g_manual_decision; then
      rem=$(timeout 90 kaggle quota 2>/dev/null | awk '/^GPU/ {print $3}')
      if awk "BEGIN{exit !(${rem:-0} >= 8)}"; then
        log "v2 did not push G by +25m with quota $rem h -> pushing G myself"
        out=$(timeout 900 python3 $S notebook-push notebookG --yes 2>&1)
        ver=$(echo "$out" | grep -oE 'KernelVersion: [0-9]+' | grep -oE '[0-9]+' | tail -1)
        echo "$out" >> "$LOG"
        if [ -n "$ver" ]; then echo "$ver" > work/pushver_ntl-run06-forkv21; log "manual G push OK ver=$ver"; else log "manual G push FAILED or 429"; fi
      else
        log "G manual check: quota only $rem h (<8) - waiting"
      fi
    fi
    # 3b. G registered check after 16:40 (sentinel should cover; backup alert)
    if [ $now -gt $((T1600 + 2400)) ] && grep -q "PUSH notebookG" work/fe_b_early2.log 2>/dev/null && once g_registered; then
      if kernel_exists lilac1996/ntl-run06-forkv21; then log "G registered (backup check)"; else log "ALERT G not registered 40m after v2 push (sentinel should be rescuing)"; fi
    fi
    # 3c. dir restore backup: v1 dead (silent >60m since 16:30) or after 18:00
    for n in F E2 B; do
      if [ -d "work/hold_notebook$n" ] && [ ! -d "notebook$n" ]; then
        v1age=$(( now - $(stat -c %Y work/fe_b_early.log 2>/dev/null || echo 0) ))
        if { [ $now -gt $((T1600 + 1800)) ] && [ $v1age -gt 3600 ]; } || [ $now -gt $((T1600 + 7200)) ]; then
          if once restore_$n; then
            mv "work/hold_notebook$n" "notebook$n"
            log "backup-restore notebook$n (v1 silent ${v1age}s)"
          fi
        fi
      fi
    done
  fi
  date "+%F %T" > work/conductor_hb
  sleep 300
done
