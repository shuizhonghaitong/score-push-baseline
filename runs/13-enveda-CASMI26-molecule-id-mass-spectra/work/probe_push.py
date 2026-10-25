import sys
import requests
_orig = requests.Session.request
def patched(self, method, url, **kw):
    r = _orig(self, method, url, **kw)
    if 'SaveKernel' in str(url):
        print('PROBE req', r.status_code, {k: v for k, v in r.headers.items()}, flush=True)
    return r
requests.Session.request = patched
try:
    import httpx
    _hs = httpx.Client.send
    def hs(self, request, **kw):
        r = _hs(self, request, **kw)
        if 'SaveKernel' in str(request.url):
            print('PROBE httpx', r.status_code, dict(r.headers), flush=True)
        return r
    httpx.Client.send = hs
except Exception as e:
    print('no httpx hook', repr(e)[:100])
import kaggle
try:
    kaggle.api.kernels_push(sys.argv[1])
    print('PUSH_OK')
except Exception as e:
    print('PUSH_ERR', repr(e)[:300])
