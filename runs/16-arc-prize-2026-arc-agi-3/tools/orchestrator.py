#!/usr/bin/env python
"""Run-16 orchestrator: push kernels -> watch -> submit (idempotent, stateful).

Usage:
  orchestrator.py tick          # one pass, then exit
  orchestrator.py loop          # pass every INTERVAL_S until killed
  orchestrator.py status        # print state and exit

State: tools/orch_state.json  (survives exec-session death)
Priority: v2 (strongest) -> v1 (insurance). Submits at most once per calendar day.
"""
import json, os, subprocess, sys, time, datetime
from pathlib import Path

RUN = Path("/mnt/disk3/liuyutong_home/workspace/skill/score-push/runs/16-arc-prize-2026-arc-agi-3")
STATE = RUN / "tools" / "orch_state.json"
LOG = RUN / "orchestrator.log"
KAGGLE_BIN = "/mnt/disk3/liuyutong_home/miniconda3/envs/kaggle/bin"
COMP = "arc-prize-2026-arc-agi-3"
SKILL = "/mnt/disk3/liuyutong_home/workspace/skill/score-push/skills/kaggle/scripts/kaggle_skill.py"
PY = sys.executable

# priority-ordered candidates: (key, notebook_dir, kernel_ref, title)
CANDS = [
    ("v2", "notebooks/duck_v2", "lilac1996/arc-agi-3-duck-harness-run16-v2", "duck v2 (anim TAAF + Qwen3.6/3.8, rank-9 config)"),
    ("v1", "notebooks/duck_v1", "lilac1996/arc-agi-3-duck-harness-run16-v1", "duck v1 (TAAF + Qwen3.6)"),
]
INTERVAL_S = 300
PUSH_EVERY_S = 900


def log(msg):
    line = f"[{datetime.datetime.now():%m-%d %H:%M:%S}] {msg}"
    with open(LOG, "a") as f:
        f.write(line + "\n")
    print(line, flush=True)


def load_state():
    if STATE.exists():
        return json.loads(STATE.read_text())
    return {"pushed": {}, "submitted": {}, "kernel_status": {}}


def save_state(st):
    STATE.write_text(json.dumps(st, indent=1))


def kaggle(*args, timeout=240):
    env = dict(os.environ, PATH=f"{KAGGLE_BIN}:{os.environ.get('PATH','')}")
    try:
        r = subprocess.run([KAGGLE_BIN + "/kaggle", *args], capture_output=True,
                           text=True, timeout=timeout, env=env, cwd=str(RUN))
        return (r.stdout or "") + (r.stderr or "")
    except subprocess.TimeoutExpired:
        return "TIMEOUT"


def kernel_status(ref):
    """Return status string for a kernel's latest session via SDK."""
    try:
        sys.path.insert(0, str(RUN / "tools"))
        from kaggle_api import client, _req
        from kagglesdk.kernels.types.kernels_api_service import ApiGetKernelSessionStatusRequest
        u, s = ref.split("/", 1)
        st = client().kernels.kernels_api_client.get_kernel_session_status(
            _req(ApiGetKernelSessionStatusRequest, u, s))
        return str(st.status)
    except Exception as e:
        return f"ERR:{type(e).__name__}"


def kernel_version(ref):
    """Latest pushed version number (integer) or None if kernel doesn't exist."""
    out = kaggle("kernels", "status", ref)
    if "not found" in out.lower() or "no kernel" in out.lower() or "404" in out:
        return None
    return 1  # we only ever push v1 of our own kernels


def try_push(st, key, ndir, ref):
    if key in st["pushed"]:
        return True
    out = kaggle("kernels", "push", "-p", ndir)
    last = out.strip().splitlines()[-1] if out.strip() else "empty"
    # success requires positive confirmation; anything else (429, cap, invalid, timeout) = not pushed
    if "successfully pushed" not in out:
        log(f"push {key} not yet: {last}")
        return False
    st["pushed"][key] = {"kernel": ref, "version": 1, "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    save_state(st)
    log(f"PUSHED {key} -> {ref} v1 :: {last}")
    return True


def today():
    return time.strftime("%Y-%m-%d")


def pass_once(st):
    # 1) push in priority order (only attempt first unpushed; avoid stacking)
    last_push = st.get("_last_push_ts", 0)
    if time.time() - last_push >= PUSH_EVERY_S:
        st["_last_push_ts"] = time.time()
        save_state(st)
        for key, ndir, ref, _ in CANDS:
            if key not in st["pushed"]:
                try_push(st, key, ndir, ref)
                break  # one push attempt per window (slots are scarce)
    # 2) refresh status of pushed kernels
    for key in list(st["pushed"]):
        ref = st["pushed"][key]["kernel"]
        st["kernel_status"][key] = kernel_status(ref)
    save_state(st)
    # 3) submit: once per day, when highest-priority pushed kernel is COMPLETE
    if st.get("submitted", {}).get("date") != today():
        for key, _, ref, title in CANDS:
            if key in st["pushed"] and st["kernel_status"].get(key) == "KernelWorkerStatus.COMPLETE":
                ver = st["pushed"][key]["version"]
                log(f"SUBMITTING {key} {ref} v{ver}")
                sk = subprocess.run([PY, SKILL, "submit", COMP, "--notebook", ref,
                                     "--version", str(ver), "-m", title, "--yes"],
                                    capture_output=True, text=True, timeout=300)
                combined = (sk.stdout or "") + (sk.stderr or "")
                st["submitted"] = {"date": today(), "kernel": ref, "version": ver,
                                   "key": key, "time": time.strftime("%Y-%m-%d %H:%M:%S")}
                save_state(st)
                log(f"SUBMIT result ({key}):\n{combined[-1500:]}")
                # append to experiments.csv
                with open(RUN / "experiments.csv", "a") as f:
                    f.write(f"{time.strftime('%Y-%m-%dT%H:%M')},run16/submit-{key},submitted,-,{ref} v{ver},,\n")
                return
        # log progress if nothing submittable yet
        statuses = {k: st["kernel_status"].get(k) for k in st["pushed"]}
        if st["pushed"]:
            log(f"state: pushed={list(st['pushed'])} status={statuses} submitted={st.get('submitted',{}).get('date')}")


def main():
    # ensure a dummy file exists for kaggle CLI --file (notebook submit needs a file arg in some CLIs)
    dummy = RUN / "dummy_submit.csv"
    if not dummy.exists():
        dummy.write_text("row_id,game_id,end_of_game,score\n1_0,1,True,1\n")
    mode = sys.argv[1] if len(sys.argv) > 1 else "tick"
    if mode == "status":
        st = load_state()
        print(json.dumps(st, indent=1))
        return
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
