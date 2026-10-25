import json, os, sys, time, requests
sys.path.insert(0, "/mnt/disk3/liuyutong_home/miniconda3/envs/kaggle/lib/python3.11/site-packages")
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.blobs.types.blob_api_service import ApiBlobType, ApiStartBlobUploadRequest
from kagglesdk.datasets.types.dataset_api_service import (
    ApiCreateDatasetVersionRequest, ApiCreateDatasetVersionRequestBody, ApiDatasetNewFile)

RUN = "/mnt/disk3/liuyutong_home/workspace/skill/score-push/runs/06-name-that-llm"
ZIP_PATH = os.path.join(RUN, "work", "svc4.zip")
api = KaggleApi()

token = None
for attempt in range(3):
    with api.build_kaggle_client() as client:
        req = ApiStartBlobUploadRequest()
        req.type = ApiBlobType.DATASET
        req.name = "svc4.zip"
        req.content_length = os.path.getsize(ZIP_PATH)
        req.last_modified_epoch_seconds = int(os.path.getmtime(ZIP_PATH))
        resp = client.blobs.blob_api_client.start_blob_upload(req)
    put_url = resp.create_url.replace("www.googleapis.com", "storage.googleapis.com")
    r = requests.put(put_url, data=open(ZIP_PATH, "rb").read(), timeout=900)
    print("attempt", attempt, "PUT:", r.status_code)
    if r.status_code == 200:
        token = resp.token
        break
    time.sleep(8)
assert token, "PUT failed"

body = ApiCreateDatasetVersionRequestBody()
body.version_notes = "run06 prefit 4-member SVC ensemble artifacts"
nf = ApiDatasetNewFile()
nf.token = token
body.files = [nf]
request = ApiCreateDatasetVersionRequest()
request.owner_slug = "lilac1996"
request.dataset_slug = "ntl-run06-svc4"
request.body = body

with api.build_kaggle_client() as client:
    out = client.datasets.dataset_api_client.create_dataset_version(request)
d = out.to_dict()
print("create version:", {k: v for k, v in d.items() if v and k in ("status", "error", "error_message", "version", "owner_slug", "dataset_slug")})
