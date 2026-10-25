import subprocess, time, os
E={"HTTP_PROXY":"http://172.28.4.16:18999","HTTPS_PROXY":"http://172.28.4.16:18999","http_proxy":"http://172.28.4.16:18999","https_proxy":"http://172.28.4.16:18999","PATH":"/mnt/disk3/liuyutong_home/miniconda3/envs/kaggle/bin:"+os.environ.get("PATH","")}
os.environ.update(E)
RUN="/mnt/disk3/liuyutong_home/workspace/skill/score-push/runs/05-misconception-detection-in-k-12-student-math-workings"
D=f"{RUN}/outputs/ds_vlm_hr"; os.makedirs(D,exist_ok=True)
LOG=open(f"{RUN}/logs/fetch_n3b2.log","a",buffering=1)
def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}",flush=True,file=LOG)
log("fetch_n3b2 start")
t0=time.time()
while time.time()-t0 < 3*3600:
    if any(f.endswith(".csv") for f in os.listdir(D)):
        log("CSV already present, stop"); break
    try:
        r=subprocess.run(f"kaggle datasets download lilac1996/miscc-vlm-ans-hr-v1 -p {D}",shell=True,capture_output=True,text=True,timeout=200)
        out=(r.stdout or "")+(r.stderr or "")
        z=f"{D}/miscc-vlm-ans-hr-v1.zip"
        if os.path.exists(z):
            subprocess.run(f"cd {D} && unzip -o -q *.zip",shell=True,capture_output=True,text=True,timeout=120)
            if any(f.endswith(".csv") for f in os.listdir(D)):
                log("N3B2 FETCHED OK"); break
        if "404" in out: log("still 404 (propagating), retry 90s")
        else: log(f"rc={r.returncode} :: {out.strip()[-120:]}")
    except Exception as e: log(f"ERR {type(e).__name__}: {e}")
    time.sleep(90)
log("fetch_n3b2 end")
