import json, os, sys, time, traceback
sys.path.insert(0, os.environ.get("KAGGLE_PKG_DIR", "/mnt/disk3/liuyutong_home/miniconda3/envs/kaggle/lib/python3.11/site-packages"))
from kagglesdk.kaggle_http_client import KaggleHttpClient
from kagglesdk.kernels.types.kernels_api_service import ApiSaveKernelRequest as SaveKernelRequest

with open("/tmp/nbbuild_18x/knb/kernel-metadata.json") as f:
    meta = json.load(f)
with open(f"/tmp/nbbuild_18x/knb/{meta['code_file']}", encoding="utf-8") as f:
    code = f.read()

http = KaggleHttpClient(verbose=False)
http._init_session()
req = SaveKernelRequest()
req.slug = meta["id"]
req.new_title = meta["title"]
req.text = code
req.language = meta["language"]
req.kernel_type = meta["kernel_type"]
req.is_private = True
req.enable_internet = True
req.competition_data_sources = meta.get("competition_sources", [])
body = SaveKernelRequest.to_dict(req)
url = "https://api.kaggle.com/v1/kernels.KernelsApiService/SaveKernel"
t0 = time.time()
try:
    r = http._session.post(url, json=body, headers=http._session.headers, auth=http._session.auth, timeout=120)
    print(f"STATUS={r.status_code} dt={time.time()-t0:.1f}s")
    for k in ("Retry-After", "X-Ratelimit-Limit", "X-Ratelimit-Remaining", "X-Ratelimit-Reset", "Server", "Via", "CF-RAY"):
        if k in r.headers:
            print(f"HDR {k}: {r.headers[k]}")
    print("BODY:", r.text[:3000])
except Exception as e:
    print(f"EXC {type(e).__name__} dt={time.time()-t0:.1f}s: {e}")
    if hasattr(e, "response") and e.response is not None:
        print("BODY:", e.response.text[:3000])
