"""Worker driver: after ssp585 install -> integrity probe, pass2 extract, cv3, v3 test."""
import os, subprocess, sys, time

RUN = "/mnt/disk3/liuyutong_home/workspace/skill/score-push/runs/18-climx-main-track"
SP = f"{RUN}/venv3/lib/python3.11/site-packages"
sys.path.insert(0, SP)
env = dict(os.environ)
env["OMP_NUM_THREADS"] = "8"
env["PYTHONPATH"] = SP + os.pathsep + env.get("PYTHONPATH", "")
env.pop("HTTP_PROXY", None); env.pop("HTTPS_PROXY", None)
env.pop("http_proxy", None); env.pop("https_proxy", None)
PY = sys.executable

def run(name, cmd, timeout):
    for attempt in range(1, 4):
        t0 = time.time()
        r = subprocess.run(cmd, cwd=RUN, env=env, timeout=timeout)
        print(f"[chain585] {name} attempt {attempt} rc={r.returncode} {time.time()-t0:.0f}s", flush=True)
        if r.returncode == 0:
            return
        time.sleep(15)
    print(f"[chain585] FATAL {name}", flush=True)
    sys.exit(1)

# 1) real integrity probe (small spatial patch per variable per 2000-day slice)
run("probe", [PY, "-u", f"{RUN}/scripts/corrupt_map.py"], 7200)
# 2) base extract (means+lbc) then pass2 lite indices for ssp585
run("base585", [PY, "-u", f"{RUN}/scripts/extract_full.py", "--scen", "ssp585"], 28800)
run("pass2", [PY, "-u", f"{RUN}/scripts/extract_full.py", "--scen", "ssp585", "--pass2", "--basemode", "lite"], 28800)
# 3) 3-fold CV (winning config: linear, no gate/quad)
run("cv3", [PY, "-u", f"{RUN}/scripts/cv_run.py"], 7200)
# 4) 3-SSP test prediction
run("v3", [PY, "-u", f"{RUN}/scripts/fullres_linear.py", "--mode", "test",
          "--scens", "ssp126,ssp370,ssp585", "--out", f"{RUN}/submissions/submission_fullres_v3.parquet"], 7200)
# 5) cone blend of v3 (50/50)
run("blend", [PY, "-u", f"{RUN}/scripts/cone_interp.py",
              f"{RUN}/submissions/submission_cone_v3.parquet", "blend",
              f"{RUN}/submissions/submission_fullres_v3.parquet"], 3600)
print("[chain585] ALL_DONE", flush=True)
