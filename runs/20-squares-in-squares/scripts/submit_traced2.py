import sys, time, http.client
import requests

_t = http.client.HTTPConnection
_orig_send = http.client.HTTPConnection.send
def traced_send(self, data):
    print(f"[SEND] {len(data)}B to {self.host}:{self.port} {self._tunnel_host or ''}", flush=True)
    try:
        r = _orig_send(self, data)
        print(f"[SEND ok]", flush=True)
        return r
    except Exception as e:
        print(f"[SEND ERR] {type(e).__name__}: {e}", flush=True)
        raise
http.client.HTTPConnection.send = traced_send

_orig = requests.Session.request
def traced(self, method, url, **kw):
    t0 = time.time()
    print(f"[REQ] {method} {url[:130]}", flush=True)
    try:
        r = _orig(self, method, url, **kw)
        print(f"[RSP] {r.status_code} in {time.time()-t0:.1f}s", flush=True)
        return r
    except Exception as e:
        print(f"[REQ ERR] {type(e).__name__}: {e} in {time.time()-t0:.1f}s", flush=True)
        raise
requests.Session.request = traced

from kaggle.cli import main
sys.argv = ["kaggle", "competitions", "submit", "-c", "squares-in-squares",
            "-f", "outputs/submission_run20a.csv.zip",
            "-m", "run20: friedman+deotte+skarin certified"]
main()
