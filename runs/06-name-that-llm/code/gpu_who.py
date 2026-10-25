import os, json, urllib.request
tok = os.environ["KAGGLE_API_TOKEN"]
base = "https://api.kaggle.com/v1"
def post(path, payload):
    req = urllib.request.Request(base+path, data=json.dumps(payload).encode(),
        headers={"Authorization": f"Bearer {tok}", "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        return {"__err": e.code, "body": e.read().decode()[:300]}
ks = post("/kernels.KernelsApiService/ListKernels", {"username":"lilac1996", "limit": 40})
kernels = ks.get("kernels", [])
print(f"total recent: {len(kernels)}")
active = []
for k in kernels:
    owner = k.get("ownerName") or (k.get("author") or {}).get("username") or "?"
    slug = k.get("slug") or ""
    title = (k.get("title") or "")[:30]
    status = post("/kernels.KernelsApiService/GetKernelSessionStatus", {"ownerName":owner,"kernelSlug":slug})
    st = status.get("status","?")
    flag = ""
    if st in ("KernelWorkerStatus.RUNNING","KernelWorkerStatus.QUEUED","KernelWorkerStatus.PENDING","KernelWorkerStatus.PROVISIONING"):
        flag = "  <== ACTIVE"
        active.append((owner,slug,st))
    print(f"{owner}/{slug:40s} {st:32s} {title}{flag}")
print("\n=== ACTIVE (GPU slot holders) ===")
for a in active: print(a)
