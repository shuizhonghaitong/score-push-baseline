#!/usr/bin/env python
"""Run-16 multi-day orchestrator v2 (day-boundary safe, verification-first).

Responsibilities:
  * Day 1 (2026-10-22): legacy monitors push the duckv2 canary; orch2 backs up
    the duckv2 submission and takes over if the canary dies (23:30 CST fallback).
  * Day 2+: push/submit the candidate chain heliosli -> ms2 -> duckv3.
  * Neutralize legacy loops at the 23:50 CST day boundary (dead refs in
    tools/orch_state.json, push markers, disabled duckv2 dir) and, once the v2
    score is posted, delete the duckv2 kernel so no legacy loop can submit it.
  * At most one submission per UTC day; verified via the submissions API.
  * Hourly leaderboard/submission evidence snapshots + RESULT.json refresh.

Usage: orch2.py tick | loop | status
State: tools/orch2_state.json
"""
import json, os, subprocess, sys, time, datetime, fcntl
from pathlib import Path

RUN = Path("/mnt/disk3/liuyutong_home/workspace/skill/score-push/runs/16-arc-prize-2026-arc-agi-3")
STATE = RUN / "tools" / "orch2_state.json"
LEGACY_STATE = RUN / "tools" / "orch_state.json"
LOG = RUN / "orch2.log"
EVIDENCE = RUN / "evidence"
KAGGLE_BIN = "/mnt/disk3/liuyutong_home/miniconda3/envs/kaggle/bin"
COMP = "arc-prize-2026-arc-agi-3"
SKILL = "/mnt/disk3/liuyutong_home/workspace/skill/score-push/skills/kaggle/scripts/kaggle_skill.py"
PY = sys.executable
INTERVAL_S = 300
PUSH_BACKOFF_S = 900
SUBMIT_RETRY_S = 1800

# key: (notebook_dir, kernel_ref, title)
CANDS = {
    "duckv2": ("notebooks/duck_v2", "lilac1996/arc-agi-3-duck-harness-run16-v2",
               "run16 duck v2 (anim TAAF + Qwen3.6-27B, rank-9 clone)"),
    "heliosli": ("notebooks/heliosli_v1", "lilac1996/arc-agi-3-heliosli-merged-run16-v1",
                 "run16 heliosli V8 merged (Qwen3.8-Flash-Next + spec decode)"),
    "ms2": ("notebooks/ms2_v1", "lilac1996/arc-agi-3-ms2-harness-run16-v1",
            "run16 milestone-2 harness (Qwen3.8-Flash-Next)"),
    "duckv3": ("notebooks/duck_v3", "lilac1996/arc-agi-3-duck-harness-run16-v3",
               "run16 duck v3 (tuned: conc 36, 9600s/game)"),
}
QUEUE = ["heliosli", "ms2", "duckv3"]  # day-2+ push order
DAY1_KEY = "duckv2"
DAY1_DATE = "2026-10-22"

def now_utc(): return datetime.datetime.now(datetime.timezone.utc)
def utc_day(): return now_utc().strftime("%Y-%m-%d")
def cst(): return datetime.datetime.now().astimezone()
def cst_day(): return cst().strftime("%Y-%m-%d")
def cst_hm(): return cst().strftime("%H:%M")

def log(msg):
    line = f"[{cst():%m-%d %H:%M:%S}] {msg}"
    with open(LOG, "a") as f:
        f.write(line + "\n")
    print(line, flush=True)

def default_state():
    return {"version": 2, "pushed": {}, "submitted": {}, "flags": {}, "ledger": []}

def load_state():
    try:
        if STATE.exists() and STATE.stat().st_size > 0:
            st = json.loads(STATE.read_text())
            for k, v in default_state().items():
                st.setdefault(k, v)
            return st
    except Exception as e:
        log(f"state corrupt ({e!r}); resetting (backup kept)")
        try:
            STATE.rename(STATE.with_suffix(f".corrupt-{int(time.time())}.bak"))
        except Exception:
            pass
    return default_state()

def save_state(st):
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(st, indent=1))
    os.replace(tmp, STATE)

def note(st, event, detail=""):
    st["ledger"].append({"ts": cst().strftime("%Y-%m-%d %H:%M:%S"), "event": event,
                         "detail": str(detail)[:400]})
    st["ledger"] = st["ledger"][-300:]
    log(f"{event}: {detail}")

def kaggle_cli(*args, timeout=240):
    env = dict(os.environ, PATH=f"{KAGGLE_BIN}:{os.environ.get('PATH','')}")
    try:
        r = subprocess.run([KAGGLE_BIN + "/kaggle", *args], capture_output=True,
                           text=True, timeout=timeout, env=env, cwd=str(RUN))
        return (r.stdout or "") + (r.stderr or "")
    except subprocess.TimeoutExpired:
        return "TIMEOUT"

def skill_submit(ref, version, msg, timeout=300):
    env = dict(os.environ, PATH=f"{KAGGLE_BIN}:{os.environ.get('PATH','')}")
    try:
        r = subprocess.run([PY, SKILL, "submit", COMP, "--notebook", ref,
                            "--version", str(version), "-m", msg, "--yes"],
                           capture_output=True, text=True, timeout=timeout, env=env, cwd=str(RUN))
        return (r.stdout or "") + (r.stderr or "")
    except subprocess.TimeoutExpired:
        return "TIMEOUT"

def kernel_status(ref):
    try:
        sys.path.insert(0, str(RUN / "tools"))
        from kaggle_api import client, _req
        from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelSessionStatusRequest
        u, s = ref.split("/", 1)
        st = client().kernels.kernels_api_client.get_kernel_session_status(
            _req(ApiGetKernelSessionStatusRequest, u, s))
        return str(st.status)
    except Exception:
        return "ERR"

def kernel_exists(ref):
    out = kaggle_cli("kernels", "status", ref).lower()
    if not out.strip() or "not found" in out or "no kernel" in out or "404" in out:
        return False
    return any(w in out for w in ("running", "complete", "failed", "error", "cancelling", "idle"))

def list_submissions():
    """Return list of dicts via SDK (fallback: None)."""
    try:
        sys.path.insert(0, str(RUN / "tools"))
        from kaggle_api import client
        from kagglesdk.competitions.types.competition_api_service import ApiListSubmissionsRequest
        r = ApiListSubmissionsRequest()
        r.competition_name = COMP
        r.page_size = 30
        out = client().competitions.competition_api_client.list_submissions(r)
        subs = []
        for s in (getattr(out, "submissions", None) or []):
            d = getattr(s, "date", None)
            subs.append({
                "date": d.isoformat() if d else None,
                "description": getattr(s, "description", "") or "",
                "public_score": getattr(s, "public_score", "") or "",
                "status": str(getattr(s, "status", "")),
                "error": getattr(s, "error_description", "") or "",
            })
        return subs
    except Exception as e:
        log(f"list_submissions error: {e!r}")
        return None

def token(key):
    return f"[run16-{key}]"

def sub_for(st, key):
    subs = list_submissions()
    if subs is None:
        return None
    for s in subs:
        if token(key) in s["description"]:
            return s
    return None

def slot_used_today():
    subs = list_submissions()
    if subs is None:
        return False
    day = utc_day()
    for s in subs:
        if s["date"] and s["date"].startswith(day):
            return True
    return False

def try_push(st, key):
    ndir, ref, _ = CANDS[key]
    if not (RUN / ndir).exists():
        note(st, "push-skip", f"{key}: dir {ndir} missing/disabled")
        return False
    out = kaggle_cli("kernels", "push", "-p", ndir)
    if "successfully pushed" not in out:
        tail = (out.strip().splitlines() or ["empty"])
        st["_next_push_ts"] = time.time() + PUSH_BACKOFF_S
        note(st, "push-fail", f"{key}: {tail[-1][:200]}")
        return False
    st["pushed"][key] = {"ref": ref, "version": 1, "time": cst().strftime("%Y-%m-%d %H:%M:%S")}
    note(st, "pushed", f"{key} -> {ref} v1")
    return True

def snapshot(st, reason):
    try:
        ts = cst().strftime("%Y%m%d_%H%M")
        lb = kaggle_cli("competitions", "leaderboard", COMP, "-s")
        (EVIDENCE / f"lb_{ts}_{reason}.txt").write_text(lb[:20000])
        subs = list_submissions()
        (EVIDENCE / f"subs_{ts}_{reason}.txt").write_text(json.dumps(subs, indent=1, default=str) if subs is not None else "list_submissions failed")
        st["_last_snapshot_ts"] = time.time()
    except Exception as e:
        log(f"snapshot error: {e!r}")

def best_public_score():
    subs = list_submissions()
    if not subs:
        return None
    best = None
    for s in subs:
        try:
            v = float(s["public_score"])
            if best is None or v > best:
                best = v
        except ValueError:
            pass
    return best

def write_result(st):
    best = best_public_score()
    res = {
        "run": "16-arc-prize-2026-arc-agi-3",
        "competition": COMP,
        "account": "lilac1996",
        "updated": cst().strftime("%Y-%m-%d %H:%M:%S %Z"),
        "status": "in_progress",
        "this_run_best_score": best,
        "submissions": [
            {"key": k, "ref": v["ref"], "utc_day": v["utc_day"], "time": v["time"],
             "verified": v.get("verified", False), "score": v.get("score")}
            for k, v in st["submitted"].items() if k != "_utc_day"
        ],
        "pushed": st["pushed"],
        "notes": "See orch2.log, experiments.csv and evidence/ for details.",
    }
    (RUN / "RESULT.json").write_text(json.dumps(res, indent=1))

def disable_legacy(st):
    if st["flags"].get("legacy_disabled"):
        return
    (RUN / ".pushed_arc-agi-3-duck-harness-run16-v1").touch(exist_ok=True)
    (RUN / ".pushed_arc-agi-3-duck-harness-run16-v2").touch(exist_ok=True)
    src = RUN / "notebooks" / "duck_v2"
    dst = RUN / "notebooks" / "duck_v2.disabled"
    if src.exists() and not dst.exists():
        src.rename(dst)
    LEGACY_STATE.write_text(json.dumps({
        "pushed": {
            "v2": {"kernel": "lilac1996/run16-disabled-never-submitted-v2", "version": 1},
            "v1": {"kernel": "lilac1996/run16-disabled-never-submitted-v1", "version": 1},
        },
        "submitted": {"date": cst_day(), "disabled_by": "orch2"},
        "kernel_status": {},
        "_last_push_ts": time.time() + 86400,
    }, indent=1))
    st["flags"]["legacy_disabled"] = cst().strftime("%Y-%m-%d %H:%M:%S")
    note(st, "legacy-disabled", "push markers + duckv2 dir + legacy orch_state neutralized")

def delete_duckv2_kernel(st):
    if st["flags"].get("duckv2_kernel_deleted"):
        return
    info = st["submitted"].get(DAY1_KEY)
    if not info or not info.get("score"):
        return
    ref = CANDS[DAY1_KEY][1]
    out = kaggle_cli("kernels", "delete", "-p", ref, "-y")
    low = out.lower()
    if "delet" in low or "success" in low or not out.strip():
        st["flags"]["duckv2_kernel_deleted"] = cst().strftime("%Y-%m-%d %H:%M:%S")
        note(st, "duckv2-kernel-deleted", f"score posted; kernel removed (out={out.strip()[:120]})")
    else:
        note(st, "duckv2-kernel-delete-failed", out.strip()[:200])

def do_submit(st, key):
    """Attempt to submit candidate. Returns True if the UTC day slot is consumed by us."""
    ndir, ref, title = CANDS[key]
    version = st["pushed"].get(key, {}).get("version", 1)
    log(f"SUBMITTING {key} {ref} v{version}")
    out = skill_submit(ref, version, f"{token(key)} {title}"[:250])
    day = utc_day()
    time.sleep(10)
    sub = sub_for(st, key)
    cli_ok = "Submitted" in out
    verified = (sub is not None and sub["error"] == "") or cli_ok
    transient = any(w in out.lower() for w in ("limit", "429", "too many", "per day", "quota", "timeout", "error"))
    if verified:
        st["submitted"][key] = {"ref": ref, "version": version, "utc_day": day,
                                "time": cst().strftime("%Y-%m-%d %H:%M:%S"), "verified": True}
        st["submitted"]["_utc_day"] = day
        note(st, "submitted", f"{key} verified in ledger :: {out.strip()[-300:]}")
    elif transient and not cli_ok:
        st["_next_submit_retry_ts"] = time.time() + SUBMIT_RETRY_S
        note(st, "submit-transient", f"{key} (slot likely taken or throttled) :: {out.strip()[-300:]}")
        return False
    else:
        st["submitted"][key] = {"ref": ref, "version": version, "utc_day": day,
                                "time": cst().strftime("%Y-%m-%d %H:%M:%S"), "verified": False}
        st["submitted"]["_utc_day"] = day
        note(st, "submit-unverified", f"{key} :: {out.strip()[-400:]}")
    snapshot(st, f"submit-{key}")
    return True

def pass_once(st):
    day = utc_day()
    ch, cm = (int(x) for x in cst_hm().split(":"))
    st["_pass_count"] = st.get("_pass_count", 0) + 1
    if st["_pass_count"] % 6 == 1:
        log(f"heartbeat pass={st['_pass_count']} utc_day={day} pushed={list(st['pushed'])} submitted={[k for k in st['submitted'] if k != '_utc_day']}")

    # ---------- score tracking for submitted keys ----------
    for key in [k for k in st["submitted"] if k != "_utc_day"]:
        info = st["submitted"][key]
        if info.get("score"):
            continue
        sub = sub_for(st, key)
        if sub:
            try:
                if sub["public_score"]:
                    info["score"] = float(sub["public_score"])
                    info["verified"] = True
                    note(st, "score-posted", f"{key} = {info['score']}")
            except ValueError:
                pass

    # ---------- day-1 fallback (23:30 CST): if v2 canary dead, take over ----------
    if cst_day() == DAY1_DATE and ch >= 23 and cm >= 30 and not st["flags"].get("day1_fallback_done"):
        st["flags"]["day1_fallback_done"] = True
        if DAY1_KEY not in st["submitted"] and sub_for(st, DAY1_KEY) is None:
            v2st = kernel_status(CANDS[DAY1_KEY][1])
            if v2st != "KernelWorkerStatus.COMPLETE":
                note(st, "day1-fallback", f"duckv2 canary {v2st}; pushing heliosli canary now")
                if "heliosli" not in st["pushed"] and time.time() >= st.get("_next_push_ts", 0):
                    try_push(st, "heliosli")

    # ---------- day-boundary neutralization (23:50 CST) ----------
    if ch >= 23 and cm >= 50 and st["flags"].get("neutralized_day") != cst_day():
        if cst_day() == DAY1_DATE:
            if kernel_exists(CANDS[DAY1_KEY][1]):
                disable_legacy(st)
                st["flags"]["neutralized_day"] = cst_day()
            elif DAY1_KEY in st["pushed"]:
                if time.time() >= st.get("_next_push_ts", 0):
                    if try_push(st, "duckv2"):
                        st["flags"]["neutralized_day"] = cst_day()  # pushed this pass; disable next pass once it exists
                # retry disable on a later pass once kernel exists
            else:
                # v2 never pushed at all: push it ourselves, don't disable dir yet
                if time.time() >= st.get("_next_push_ts", 0):
                    try_push(st, "duckv2")
        else:
            disable_legacy(st)
            st["flags"]["neutralized_day"] = cst_day()

    # ---------- push the day-2+ canary (from 23:50 CST on day1, or later days) ----------
    day2_push_ok = (cst_day() > DAY1_DATE) or (ch >= 23 and cm >= 50)
    if day2_push_ok:
        for key in QUEUE:
            if key in st["submitted"] or st["flags"].get(f"{key}_dead"):
                continue
            if key not in st["pushed"]:
                if time.time() >= st.get("_next_push_ts", 0):
                    try_push(st, key)
                break
            if kernel_status(CANDS[key][1]) in ("KernelWorkerStatus.ERROR", "KernelWorkerStatus.FAILED"):
                note(st, "canary-failed", key)
                st["pushed"].pop(key, None)
                st["flags"][f"{key}_dead"] = True
            break

    # ---------- submit: once per UTC day, priority order ----------
    if st["submitted"].get("_utc_day") != day and time.time() >= st.get("_next_submit_retry_ts", 0):
        if slot_used_today():
            st["submitted"]["_utc_day"] = day
            note(st, "slot-already-used", f"UTC {day} submission already exists (legacy/other); skipping")
        else:
            priority = []
            if DAY1_KEY not in st["submitted"] and cst_day() == DAY1_DATE:
                priority.append(DAY1_KEY)
            for key in QUEUE:
                if key not in st["submitted"] and not st["flags"].get(f"{key}_dead"):
                    priority.append(key)
            for key in priority:
                pushed = st["pushed"].get(key) or (key == DAY1_KEY and kernel_exists(CANDS[key][1]))
                if not pushed:
                    continue
                if kernel_status(CANDS[key][1]) == "KernelWorkerStatus.COMPLETE":
                    if do_submit(st, key):
                        break
    delete_duckv2_kernel(st)

    # ---------- hourly evidence + RESULT ----------
    if time.time() - st.get("_last_snapshot_ts", 0) > 3600:
        snapshot(st, "hourly")
        write_result(st)
    save_state(st)

def main():
    EVIDENCE.mkdir(exist_ok=True)
    mode = sys.argv[1] if len(sys.argv) > 1 else "tick"
    if mode == "status":
        print(json.dumps(load_state(), indent=1, default=str))
        return
    if mode == "loop":
        while True:
            try:
                with STATE.open("a+") as lf:
                    fcntl.flock(lf, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    try:
                        pass_once(load_state())
                    finally:
                        fcntl.flock(lf, fcntl.LOCK_UN)
            except (BlockingIOError, OSError) as e:
                log(f"another instance holds lock: {e!r}")
            except Exception as e:
                log(f"pass error: {e!r}")
            time.sleep(INTERVAL_S)
    else:
        pass_once(load_state())

if __name__ == "__main__":
    main()
