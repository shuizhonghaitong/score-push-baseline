"""Worker driver: after hist_late install -> probe, full-res base, pass2 full for all SSPs,
cv3, v3 full-base test prediction, cone blend."""
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
HIST = f"{RUN}/data/NorESM2-MM_r1i1p1f1_with_forcing_hist_train_1932_2014.nc"

def run(name, cmd, timeout, retries=3):
    for attempt in range(1, retries + 1):
        t0 = time.time()
        r = subprocess.run(cmd, cwd=RUN, env=env, timeout=timeout)
        print(f"[chainhist] {name} attempt {attempt} rc={r.returncode} {time.time()-t0:.0f}s", flush=True)
        if r.returncode == 0:
            return
        time.sleep(15)
    print(f"[chainhist] FATAL {name}", flush=True)
    sys.exit(1)

run("probe", [PY, "-u", f"{RUN}/scripts/corrupt_map.py", HIST], 7200)
run("make_base", [PY, "-u", f"{RUN}/scripts/extract_full.py", "--make-base"], 28800)
for s in ["ssp126", "ssp370", "ssp585"]:
    run(f"pass2full_{s}", [PY, "-u", f"{RUN}/scripts/extract_full.py", "--scen", s, "--pass2", "--basemode", "full"], 28800)
run("cv3", [PY, "-u", f"{RUN}/scripts/cv_run.py"], 7200)
run("v3fb", [PY, "-u", f"{RUN}/scripts/fullres_linear.py", "--mode", "test",
            "--scens", "ssp126,ssp370,ssp585", "--out", f"{RUN}/submissions/submission_fullres_v3fb.parquet"], 7200)
run("blend", [PY, "-u", f"{RUN}/scripts/cone_interp.py",
              f"{RUN}/submissions/submission_cone_v3fb.parquet", "blend",
              f"{RUN}/submissions/submission_fullres_v3fb.parquet"], 3600)
print("[chainhist] ALL_DONE", flush=True)
