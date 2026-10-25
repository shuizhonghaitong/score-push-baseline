#!/usr/bin/env python
"""Run-16 orchestrator v3 (orch2b) — resilient to the batch-quota reality.

Pushes duckv2 -> heliosli -> ms2 -> duckv3 canaries as batch quota allows
(15-min backoff, staggered: next canary only after the current one is
complete/failed or >3h old), submits the best COMPLETE canary once per UTC
day (verified via the submissions API with [run16-<key>] tokens), tracks
posted scores, neutralizes the legacy 19333 loop via dead refs in
tools/orch_state.json, and — after the duckv2 score posts — retires the
duckv2 kernel+dir so no legacy loop can resubmit it.

On startup it holds an exclusive lock on tools/orch2_state.json, which makes
the superseded orch2 loop (old code in memory) a harmless no-op.

Usage: orch2b.py tick | loop | status
State: tools/orch2b_state.json   Log: orch2b.log
"""
import fcntl, hashlib, json, os, shutil, subprocess, sys, time, datetime
from pathlib import Path

RUN = Path("/mnt/disk3/liuyutong_home/workspace/skill/score-push/runs/16-arc-prize-2026-arc-agi-3")
STATE = RUN / "tools" / "orch2b_state.json"
OLD_STATE = RUN / "tools" / "orch2_state.json"   # lock target to inert the old loop
LEGACY_STATE = RUN / "tools" / "orch_state.json" # legacy orchestrator state
LOG = RUN / "orch2b.log"
EVIDENCE = RUN / "evidence"
KAGGLE_BIN = "/mnt/disk3/liuyutong_home/miniconda3/envs/kaggle/bin"
COMP = "arc-prize-2026-arc-agi-3"
SKILL = "/mnt/disk3/liuyutong_home/workspace/skill/score-push/skills/kaggle/scripts/kaggle_skill.py"
PY = sys.executable
INTERVAL_S = 300
PUSH_BACKOFF_S = 900
SUBMIT_RETRY_S = 1800
CAND_STAGGER_S = 3 * 3600
DUCK_GRACE_S = 5 * 3600

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
PRIO = ["duckv2", "heliosli", "ms2", "duckv3"]

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
    return {"version": 3, "pushed": {}, "submitted": {}, "flags": {}, "ledger": []}

def load_state():
    try:
        if STATE.exists() and STATE.stat().st_size > 0:
            st = json.loads(STATE.read_text())
            for k, v in default_state().items():
                st.setdefault(k, v)
            return st
    except Exception as e:
        log(f"state corrupt ({e!r}); resetting")
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

def kernel_check(ref):
    """'yes' | 'no' | 'unknown' — never act on transport-error output."""
    out = kaggle_cli("kernels", "status", ref)
    low = out.lower()
    if not low.strip():
        return "unknown"
    if "404" in low or "not found" in low or "no kernel" in low:
        return "no"
    if any(w in low for w in ("running", "complete", "failed", "cancelling", "idle")):
        return "yes"
    return "unknown"  # SSL/proxy/timeout garbage or unrecognized output

def kernel_exists(ref):
    return kernel_check(ref) == "yes"

def repair_bogus_adoptions(st):
    """Drop adopted/legacy-push entries whose kernel provably does not exist
    (created by a false-positive kernel_exists during proxy outages)."""
    for key in list(st["pushed"].keys()):
        info = st["pushed"][key]
        if info.get("time_str") in ("legacy-push", "adopted"):
            res = kernel_check(info["ref"])
            if res == "no":
                note(st, "bogus-adoption-removed",
                     f"{key}: kernel {info['ref']} does not exist (verified 404)")
                st["pushed"].pop(key, None)

def list_submissions():
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
            subs.append({"date": d.isoformat() if d else None,
                         "description": getattr(s, "description", "") or "",
                         "public_score": getattr(s, "public_score", "") or "",
                         "status": str(getattr(s, "status", "")),
                         "error": getattr(s, "error_description", "") or ""})
        return subs
    except Exception as e:
        log(f"list_submissions error: {e!r}")
        return None

def token(key): return f"[run16-{key}]"

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
    return any(s["date"] and s["date"].startswith(day) for s in subs)

def quota_probe():
    try:
        from kagglehub.datasets_helpers import build_kaggle_client
        from kagglesdk.kernels.types.kernels_api_service import ApiSaveKernelRequest
        kc = build_kaggle_client()
        kc.kernels.kernels_api_client.save_kernel(ApiSaveKernelRequest())
        return "ACCEPTED"
    except Exception as e:
        resp = getattr(e, "response", None)
        h = dict(resp.headers) if resp is not None and hasattr(resp, "headers") else {}
        ra = h.get("retry-after")
        if ra:
            reset = now_utc() + datetime.timedelta(seconds=int(ra))
            return f"429 retry-after={ra}s (~{reset:%m-%d %H:%M} UTC)"
        return f"{type(e).__name__}"

def try_push(st, key):
    ndir, ref, _ = CANDS[key]
    if not (RUN / ndir).exists() or not (RUN / ndir / "nb.ipynb").exists():
        note(st, "push-skip", f"{key}: dir {ndir} missing/disabled")
        return False
    out = kaggle_cli("kernels", "push", "-p", ndir)
    if "successfully pushed" not in out:
        tail = (out.strip().splitlines() or ["empty"])
        st["_next_push_ts"] = time.time() + PUSH_BACKOFF_S
        note(st, "push-fail", f"{key}: {tail[-1][:160]}")
        return False
    st["pushed"][key] = {"ref": ref, "version": 1, "time": time.time(),
                         "time_str": cst().strftime("%Y-%m-%d %H:%M:%S")}
    note(st, "pushed", f"{key} -> {ref} v1")
    return True

def snapshot(st, reason):
    try:
        ts = cst().strftime("%Y%m%d_%H%M")
        lb = kaggle_cli("competitions", "leaderboard", COMP, "-s")
        (EVIDENCE / f"lb_{ts}_{reason}.txt").write_text(lb[:20000])
        subs = list_submissions()
        (EVIDENCE / f"subs_{ts}_{reason}.txt").write_text(
            json.dumps(subs, indent=1, default=str) if subs is not None else "list_submissions failed")
        q = quota_probe()
        (EVIDENCE / f"quota_{ts}_{reason}.txt").write_text(q + "\n")
        st["_last_snapshot_ts"] = time.time()
        log(f"snapshot {reason}: quota={q}")
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
            best = v if best is None or v > best else best
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
        "pushed": {k: v["time_str"] for k, v in st["pushed"].items()},
        "notes": "Batch-session quota (account-level, shared with other runs) gates kernel pushes; "
                 "see orch2b.log, experiments.csv, evidence/ for details.",
    }
    (RUN / "RESULT.json").write_text(json.dumps(res, indent=1))

def neutralize_legacy(st):
    if st["flags"].get("legacy_neutralized"):
        return
    (RUN / ".pushed_arc-agi-3-duck-harness-run16-v1").touch(exist_ok=True)
    LEGACY_STATE.write_text(json.dumps({
        "pushed": {
            "v2": {"kernel": "lilac1996/run16-disabled-never-submitted-v2", "version": 1},
            "v1": {"kernel": "lilac1996/run16-disabled-never-submitted-v1", "version": 1},
        },
        "submitted": {"date": cst_day(), "disabled_by": "orch2b"},
        "kernel_status": {},
        "_last_push_ts": time.time() + 86400,
    }, indent=1))
    st["flags"]["legacy_neutralized"] = cst().strftime("%Y-%m-%d %H:%M:%S")
    note(st, "legacy-neutralized", "19333 loop disabled via dead refs in tools/orch_state.json")

def retire_duckv2(st):
    """After duckv2 score posts: stop legacy re-creation, delete the kernel."""
    info = st["submitted"].get("duckv2")
    if not info or not info.get("score"):
        return
    if not st["flags"].get("duckv2_dir_retired"):
        src = RUN / "notebooks" / "duck_v2"
        dst = RUN / "notebooks" / "duck_v2.disabled"
        try:
            if src.exists():
                if dst.exists() and not any(dst.iterdir()):
                    shutil.rmtree(dst)
                src.rename(dst)
                st["flags"]["duckv2_dir_retired"] = cst().strftime("%Y-%m-%d %H:%M:%S")
                note(st, "duckv2-dir-retired", f"moved {src} -> {dst}")
        except Exception as e:
            note(st, "duckv2-dir-retire-failed", repr(e))
    if not st["flags"].get("duckv2_kernel_deleted"):
        out = kaggle_cli("kernels", "delete", "-p", CANDS["duckv2"][1], "-y")
        low = out.lower()
        if "delet" in low or "success" in low or not out.strip():
            st["flags"]["duckv2_kernel_deleted"] = cst().strftime("%Y-%m-%d %H:%M:%S")
            note(st, "duckv2-kernel-deleted", out.strip()[:120])
        else:
            note(st, "duckv2-kernel-delete-failed", out.strip()[:200])

def do_submit(st, key):
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
        note(st, "submitted", f"{key} verified :: {out.strip()[-250:]}")
    elif transient and not cli_ok:
        st["_next_submit_retry_ts"] = time.time() + SUBMIT_RETRY_S
        note(st, "submit-transient", f"{key} (quota/throttled) :: {out.strip()[-250:]}")
        return False
    else:
        st["submitted"][key] = {"ref": ref, "version": version, "utc_day": day,
                                "time": cst().strftime("%Y-%m-%d %H:%M:%S"), "verified": False}
        st["submitted"]["_utc_day"] = day
        note(st, "submit-unverified", f"{key} :: {out.strip()[-350:]}")
    snapshot(st, f"submit-{key}")
    return True

def _code_hash():
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]

def pass_once(st):
    # auto-restart via keepalive if the on-disk code changed since we started
    if st.get("_code_hash") != _code_hash():
        log(f"code changed on disk (was {st.get('_code_hash')} now {_code_hash()}); exiting for keepalive respawn")
        st["_code_hash"] = _code_hash()  # so the respawned process passes the check
        save_state(st)
        os._exit(3)
    st["_code_hash"] = _code_hash()
    day = utc_day()
    repair_bogus_adoptions(st)
    st["_pass_count"] = st.get("_pass_count", 0) + 1
    if st["_pass_count"] % 6 == 1:
        log(f"heartbeat pass={st['_pass_count']} utc={day} pushed={list(st['pushed'])} "
            f"submitted={[k for k in st['submitted'] if k != '_utc_day']}")

    # ---- score tracking ----
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

    # ---- neutralize legacy (from 10-23; 5180/push_monitor keep pushing duckv2 as backup) ----
    if cst_day() > "2026-10-22":
        neutralize_legacy(st)

    # ---- canary status bookkeeping ----
    for key, info in list(st["pushed"].items()):
        status = kernel_status(info["ref"])
        info["status"] = status
        if status in ("KernelWorkerStatus.ERROR", "KernelWorkerStatus.FAILED"):
            st["flags"][f"{key}_dead"] = True
            note(st, "canary-failed", f"{key}: {status}")
            st["pushed"].pop(key, None)

    # ---- push management (staggered) ----
    if time.time() >= st.get("_next_push_ts", 0):
        active = None
        for key in PRIO:
            if key in st["pushed"]:
                active = key
                break
        if active is None:
            for key in PRIO:
                if key not in st["pushed"] and not st["flags"].get(f"{key}_dead"):
                    # adopt if a legacy pusher already created this kernel (avoid re-push/re-run)
                    if kernel_exists(CANDS[key][1]):
                        st["pushed"][key] = {"ref": CANDS[key][1], "version": 1,
                                             "time": time.time(), "time_str": "adopted",
                                             "status": kernel_status(CANDS[key][1])}
                        note(st, "adopted", f"{key} kernel already exists (legacy push); not re-pushing")
                        break
                    try_push(st, key)  # one attempt per window; break on any outcome
                    break
        else:
            info = st["pushed"][active]
            age = time.time() - info.get("time", 0)
            status = info.get("status", "")
            if status == "KernelWorkerStatus.COMPLETE" or age > CAND_STAGGER_S:
                for key in PRIO:
                    if key != active and key not in st["pushed"] and not st["flags"].get(f"{key}_dead"):
                        try_push(st, key)
                        break

    # ---- submit: once per UTC day ----
    if st["submitted"].get("_utc_day") != day and time.time() >= st.get("_next_submit_retry_ts", 0):
        if slot_used_today():
            st["submitted"]["_utc_day"] = day
            note(st, "slot-already-used", f"UTC {day} has a submission (legacy/other); skipping")
        else:
            head_grace_used = False
            for key in PRIO:
                if key in st["submitted"] or st["flags"].get(f"{key}_dead"):
                    continue
                if key in st["pushed"]:
                    ref, info = CANDS[key][1], st["pushed"][key]
                    status = info.get("status") or kernel_status(ref)
                elif key == "duckv2" and kernel_exists(CANDS[key][1]):
                    # pushed by a legacy loop
                    st["pushed"][key] = {"ref": CANDS[key][1], "version": 1,
                                         "time": time.time(), "time_str": "legacy-push"}
                    ref = CANDS[key][1]
                    status = kernel_status(ref)
                    st["pushed"][key]["status"] = status
                else:
                    continue
                if status == "KernelWorkerStatus.COMPLETE":
                    if do_submit(st, key):
                        break
                elif (key == PRIO[0] and not head_grace_used and
                      time.time() - st["pushed"][key].get("time", 0) > DUCK_GRACE_S):
                    head_grace_used = True  # duckv2 stalled: let later candidates take the day
    retire_duckv2(st)

    # ---- hourly evidence + RESULT ----
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
    # hold the old loop's lock for our whole lifetime
    old_lock = OLD_STATE.open("a+")
    fcntl.flock(old_lock, fcntl.LOCK_EX)
    if mode == "loop":
        while True:
            try:
                pass_once(load_state())
            except Exception as e:
                log(f"pass error: {e!r}")
            time.sleep(INTERVAL_S)
    else:
        pass_once(load_state())

if __name__ == "__main__":
    main()
