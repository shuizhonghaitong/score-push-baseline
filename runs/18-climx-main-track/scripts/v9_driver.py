"""Worker driver: run train_v9.py for given holdout SSPs, saving cross-SSP lite predictions."""
import os, subprocess, sys, time

RUN = "/mnt/disk3/liuyutong_home/workspace/skill/score-push/runs/18-climx-main-track"
SP = f"{RUN}/venv3/lib/python3.11/site-packages"
sys.path.insert(0, SP)
env = dict(os.environ)
env["OMP_NUM_THREADS"] = "8"
env["PYTHONPATH"] = SP + os.pathsep + env.get("PYTHONPATH", "")

for ho in sys.argv[1:]:
    ok = False
    for attempt in range(1, 4):
        t0 = time.time()
        r = subprocess.run([sys.executable, "-u", f"{RUN}/scripts/train_v9.py", "--holdout", ho],
                           cwd=RUN, env=env, timeout=5400)
        print(f"[v9driver] {ho} attempt {attempt} rc={r.returncode} {time.time()-t0:.0f}s", flush=True)
        if r.returncode == 0:
            ok = True
            break
        time.sleep(20)
    if not ok:
        print(f"[v9driver] FATAL {ho}", flush=True)
        sys.exit(1)
print(f"[v9driver] V9_DONE {sys.argv[1:]}", flush=True)
