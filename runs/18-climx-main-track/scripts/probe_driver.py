import os, subprocess, sys
RUN = "/mnt/disk3/liuyutong_home/workspace/skill/score-push/runs/18-climx-main-track"
SP = f"{RUN}/venv3/lib/python3.11/site-packages"
sys.path.insert(0, SP)
env = dict(os.environ)
env["OMP_NUM_THREADS"] = "2"
env["PYTHONPATH"] = SP + os.pathsep + env.get("PYTHONPATH", "")
r = subprocess.run([sys.executable, "-u", f"{RUN}/scripts/probe_integrity.py"], cwd=RUN, env=env, timeout=7200)
print(f"[probedriver] rc={r.returncode}", flush=True)
sys.exit(r.returncode)
