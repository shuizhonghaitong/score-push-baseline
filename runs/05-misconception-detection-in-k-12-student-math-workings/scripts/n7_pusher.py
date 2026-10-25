import subprocess, time, os
E={"HTTP_PROXY":"http://172.28.4.16:18999","HTTPS_PROXY":"http://172.28.4.16:18999","http_proxy":"http://172.28.4.16:18999","https_proxy":"http://172.28.4.16:18999","PATH":"/mnt/disk3/liuyutong_home/miniconda3/envs/kaggle/bin:"+os.environ.get("PATH","")}
os.environ.update(E)
RUN="/mnt/disk3/liuyutong_home/workspace/skill/score-push/runs/05-misconception-detection-in-k-12-student-math-workings"
DIR=f"{RUN}/notebooks/n7-push"
LOG=open(f"{RUN}/logs/n7_pusher.log","a",buffering=1)
def log(m): print(f"[{time.strftime('%H:%M:%S')}] {m}",flush=True,file=LOG)
log("n7 pusher start")
t0=time.time()
while time.time()-t0 < 10*3600:
    try:
        r=subprocess.run(f"kaggle kernels push -p {DIR}",shell=True,capture_output=True,text=True,timeout=180)
        out=(r.stdout or "")+(r.stderr or "")
        if r.returncode==0 and "push error" not in out:
            log("N7 PUSHED OK"); break
        if "Maximum batch GPU" in out:
            log("slots full, retry in 45s")
        else:
            log(f"push rc={r.returncode} :: {out.strip()[-150:]}")
    except Exception as e:
        log(f"ERR {type(e).__name__}: {e}")
    time.sleep(45)
log("n7 pusher end")
