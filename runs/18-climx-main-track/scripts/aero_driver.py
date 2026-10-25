"""Worker driver for aerosol extraction."""
import os, subprocess, sys, time
RUN = "/mnt/disk3/liuyutong_home/workspace/skill/score-push/runs/18-climx-main-track"
SP = f"{RUN}/venv3/lib/python3.11/site-packages"
sys.path.insert(0, SP)
env = dict(os.environ)
env["OMP_NUM_THREADS"] = "4"
env["PYTHONPATH"] = SP + os.pathsep + env.get("PYTHONPATH", "")
env.pop("HTTP_PROXY", None); env.pop("HTTPS_PROXY", None)
env.pop("http_proxy", None); env.pop("https_proxy", None)
for scen in sys.argv[1:]:
    r = subprocess.run([sys.executable, "-u", f"{RUN}/scripts/extract_aerosol.py", scen],
                       cwd=RUN, env=env, timeout=7200)
    print(f"[aerodriver] {scen} rc={r.returncode}", flush=True)
    if r.returncode != 0:
        sys.exit(1)
print("[aerodriver] ALL_DONE", flush=True)
