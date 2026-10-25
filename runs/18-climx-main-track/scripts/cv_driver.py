"""Worker driver for CV / test prediction steps.
Usage: cv_driver.py <step> ...   step in {smoke, cv3, v1, v1check}
"""
import os, subprocess, sys, time

RUN = "/mnt/disk3/liuyutong_home/workspace/skill/score-push/runs/18-climx-main-track"
SP = f"{RUN}/venv3/lib/python3.11/site-packages"
sys.path.insert(0, SP)
env = dict(os.environ)
env["OMP_NUM_THREADS"] = "8"
env["PYTHONPATH"] = SP + os.pathsep + env.get("PYTHONPATH", "")
PY = sys.executable

STEPS = {
    "smoke": [PY, "-u", f"{RUN}/scripts/cv_smoke.py"],
    "cv3": [PY, "-u", f"{RUN}/scripts/cv_run.py"],
    "v1": [PY, "-u", f"{RUN}/scripts/fullres_linear.py", "--mode", "test",
           "--scens", "ssp126,ssp370", "--out", f"{RUN}/submissions/submission_fullres_v1.parquet"],
    "cv2": [PY, "-u", f"{RUN}/scripts/cv_run_v2.py"],
    "diag2": [PY, "-u", f"{RUN}/scripts/cv_diag2.py"],
    "diag3": [PY, "-u", f"{RUN}/scripts/diag3.py"],
    "temporal": [PY, "-u", f"{RUN}/scripts/cv_temporal.py"],
    "diagscale": [PY, "-u", f"{RUN}/scripts/diag_scale.py"],
    "diag4": [PY, "-u", f"{RUN}/scripts/diag4.py"],
    "diag5": [PY, "-u", f"{RUN}/scripts/diag5.py"],
    "v3": [PY, "-u", f"{RUN}/scripts/fullres_linear.py", "--mode", "test",
           "--scens", "ssp126,ssp370,ssp585", "--out", f"{RUN}/submissions/submission_fullres_v3.parquet"],
    "v2test2": [PY, "-u", f"{RUN}/scripts/fullres_linear.py", "--mode", "test",
                "--scens", "ssp126,ssp370", "--block",
                "--out", f"{RUN}/submissions/submission_fullres_v2.parquet"],
    "v2test3": [PY, "-u", f"{RUN}/scripts/fullres_linear.py", "--mode", "test", "--block",
                "--out", f"{RUN}/submissions/submission_fullres_v2_3ssp.parquet"],
}

argv = sys.argv[1:]
i = 0
while i < len(argv):
    step = argv[i]
    if step not in STEPS:
        i += 1
        continue
    j = i + 1
    while j < len(argv) and argv[j] not in STEPS:
        j += 1
    cmd = STEPS[step] + argv[i + 1:j]
    ok = False
    for attempt in range(1, 4):
        t0 = time.time()
        r = subprocess.run(cmd, cwd=RUN, env=env, timeout=7200)
        print(f"[cvdriver] {step} attempt {attempt} rc={r.returncode} {time.time()-t0:.0f}s", flush=True)
        if r.returncode == 0:
            ok = True
            break
        time.sleep(20)
    if not ok:
        print(f"[cvdriver] FATAL {step}", flush=True)
        sys.exit(1)
    i = j
print(f"[cvdriver] ALL_STEPS_DONE {sys.argv[1:]}", flush=True)
