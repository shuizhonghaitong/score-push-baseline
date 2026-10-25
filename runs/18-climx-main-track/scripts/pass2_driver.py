"""Worker driver: run extract_full.py --pass2 (lite base) for given scenarios, with retries."""
import os, subprocess, sys, time

RUN = "/mnt/disk3/liuyutong_home/workspace/skill/score-push/runs/18-climx-main-track"
SP = f"{RUN}/venv3/lib/python3.11/site-packages"
sys.path.insert(0, SP)
env = dict(os.environ)
env["OMP_NUM_THREADS"] = "2"
env["PYTHONPATH"] = SP + os.pathsep + env.get("PYTHONPATH", "")
env.pop("HTTP_PROXY", None); env.pop("HTTPS_PROXY", None)
env.pop("http_proxy", None); env.pop("https_proxy", None)

scens = sys.argv[1:]
assert scens, "usage: pass2_driver.py ssp126 [ssp370 ...]"
for scen in scens:
    ok = False
    for attempt in range(1, 5):
        t0 = time.time()
        r = subprocess.run(
            [sys.executable, "-u", f"{RUN}/scripts/extract_full.py",
             "--scen", scen, "--pass2", "--basemode", "lite"],
            cwd=RUN, env=env, timeout=14400,
        )
        print(f"[driver] {scen} attempt {attempt} rc={r.returncode} {time.time()-t0:.0f}s", flush=True)
        if r.returncode == 0:
            ok = True
            break
        time.sleep(15)
    if not ok:
        print(f"[driver] FATAL {scen} failed all attempts", flush=True)
        sys.exit(1)
print(f"[driver] WORKER_PASS2_DONE {scens}", flush=True)
