import subprocess, time, os, re, sys
# self-contained env
E = {
 "HTTP_PROXY":"http://172.28.4.16:18999","HTTPS_PROXY":"http://172.28.4.16:18999",
 "http_proxy":"http://172.28.4.16:18999","https_proxy":"http://172.28.4.16:18999",
 "PATH":"/mnt/disk3/liuyutong_home/miniconda3/envs/kaggle/bin:"+os.environ.get("PATH",""),
}
os.environ.update(E)
RUN = "/mnt/disk3/liuyutong_home/workspace/skill/score-push/runs/05-misconception-detection-in-k-12-student-math-workings"
SKILL = "/mnt/disk3/liuyutong_home/workspace/skill/score-push/skills/kaggle/scripts/kaggle_skill.py"
N5 = "misconception-detection-vlm-diagnosis-v1"
N3B2 = "misconception-detection-vlm-answer-hr-v1"
N7 = "misconception-detection-vlm-lora-v1"
LOG = open(f"{RUN}/logs/worker.log","a",buffering=1)
def log(msg):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True, file=LOG)
def sh(cmd, timeout=180):
    for a in range(3):
        try:
            r = subprocess.run(cmd, shell=True, capture_output=True, text=True, timeout=timeout)
            return r.returncode, (r.stdout or "") + (r.stderr or "")
        except subprocess.TimeoutExpired:
            log(f"  timeout on: {cmd[:60]} (attempt {a})")
        time.sleep(20)
    return 999, "timeout"
def kstatus(slug):
    rc, out = sh(f"kaggle kernels status lilac1996/{slug} 2>&1 | tail -3", timeout=90)
    m = re.search(r'KernelWorkerStatus\.(\w+)', out)
    return m.group(1) if m else f"UNK(rc{rc})"
def terminal(st):
    return st in ("COMPLETE","ERROR","CANCELLED")

DL5 = DL3B2 = PUSHED7 = False
t_end = time.time() + 6*3600
while time.time() < t_end:
    try:
        s5, s3 = kstatus(N5), kstatus(N3B2)
        log(f"n5={s5} n3b2={s3} dl5={DL5} dl3b2={DL3B2} pushed7={PUSHED7}")
        if terminal(s5) and not DL5:
            rc,out = sh(f"kaggle datasets download lilac1996/miscc-vlm-diag-v1 -p {RUN}/outputs/ds_diag 2>&1 && cd {RUN}/outputs/ds_diag && unzip -o -q *.zip 2>&1 && ls -la", timeout=600)
            DL5 = rc==0
            log(f"  n5 dataset download rc={rc} :: {out[-160:]}")
        if terminal(s3) and not DL3B2:
            rc,out = sh(f"kaggle datasets download lilac1996/miscc-vlm-ans-hr-v1 -p {RUN}/outputs/ds_vlm_hr 2>&1 && cd {RUN}/outputs/ds_vlm_hr && unzip -o -q *.zip 2>&1 && ls -la", timeout=600)
            DL3B2 = rc==0
            log(f"  n3b2 dataset download rc={rc} :: {out[-160:]}")
        if not PUSHED7:
            rc,out = sh(f"python3 {SKILL} notebook-push {RUN}/notebooks/n7-vlm-lora --yes 2>&1", timeout=240)
            if rc==0:
                PUSHED7=True; log("  N7 PUSHED OK")
            elif "Maximum batch GPU" in out:
                log("  n7 push: 2 slots still full, will retry")
            else:
                log(f"  n7 push rc={rc} :: {out[-200:]}")
        if DL5 and DL3B2 and PUSHED7:
            log("WORKER COMPLETE: n5+n3b2 downloaded, n7 pushed"); break
    except Exception as e:
        log(f"ERR {type(e).__name__}: {e}")
    time.sleep(300)
log("worker loop end")
