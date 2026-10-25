"""Run kaggle submit with request-level tracing to find stalls."""
import sys, time
import requests

_orig = requests.Session.request
def traced(self, method, url, **kw):
    t0 = time.time()
    print(f"[REQ] {method} {url} (timeout={kw.get('timeout')})", flush=True)
    try:
        r = _orig(self, method, url, **kw)
        print(f"[RSP] {r.status_code} in {time.time()-t0:.1f}s  {url}", flush=True)
        return r
    except Exception as e:
        print(f"[ERR] {type(e).__name__}: {e} in {time.time()-t0:.1f}s  {url}", flush=True)
        raise
requests.Session.request = traced

from kaggle.cli import main
sys.argv = ["kaggle", "competitions", "submit", "-c", "squares-in-squares",
            "-f", "outputs/submission_run20a.csv.zip",
            "-m", "run20: friedman+deotte+skarin certified"]
main()
