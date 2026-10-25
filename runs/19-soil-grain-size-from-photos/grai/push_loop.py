"""Retry pushing the paper-fetch kernel until a session slot frees, then wait + download."""
import os, sys, time, warnings, json
warnings.filterwarnings("ignore")
import requests

_orig = requests.Session.send
def rewrite_send(self, request, **kw):
    kw.setdefault("timeout", (15, 300))
    if "www.googleapis.com" in request.url:
        request.url = request.url.replace("www.googleapis.com", "storage.googleapis.com")
    return _orig(self, request, **kw)
requests.Session.send = rewrite_send

from kaggle.api.kaggle_api_extended import KaggleApi
api = KaggleApi()
api.with_retry = lambda f, *a, **k: f
api.authenticate()

os.makedirs("downloads/grai3-paper-fetch-output", exist_ok=True)
for attempt in range(60):
    try:
        r = api.kernels_push("notebooks/grai3-paper-fetch")
        err = r._error if r and r._error else None
    except Exception as e:
        err = f"exc {type(e).__name__} {str(e)[:100]}"
    print(f"[{time.strftime('%H:%M:%S')}] attempt {attempt+1}: {err or 'ok'}", flush=True)
    if err:
        time.sleep(300)
        continue
    # running; poll
    for i in range(40):
        time.sleep(45)
        st = api.kernels_status("lilac1996/grai3-paper-fetch")
        print(f"[{time.strftime('%H:%M:%S')}] status: {st}", flush=True)
        if st.get("status") in ("COMPLETE", "ERROR", "CANCELLED"):
            api.kernels_output("lilac1996/grai3-paper-fetch", path="downloads/grai3-paper-fetch-output")
            for f in sorted(os.listdir("downloads/grai3-paper-fetch-output")):
                p = os.path.join("downloads/grai3-paper-fetch-output", f)
                print("OUT:", f, os.path.getsize(p), flush=True)
            print("DONE", flush=True)
            sys.exit(0)
    print("timeout waiting; will retry push", flush=True)
print("gave up", flush=True)
sys.exit(1)
