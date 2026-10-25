import json, os, sys, time, zipfile
from urllib.parse import urlparse

sys.path.insert(0, "/mnt/disk3/liuyutong_home/miniconda3/envs/kaggle/lib/python3.11/site-packages")
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.blobs.types.blob_api_service import ApiBlobType, ApiStartBlobUploadRequest
from kagglesdk.datasets.types.dataset_api_service import ApiCreateDatasetRequest, ApiDatasetNewFile

import requests

RUN = "/mnt/disk3/liuyutong_home/workspace/skill/score-push/runs/06-name-that-llm"
FOLDER = os.path.join(RUN, "work", "svc5")
ZIP_PATH = os.path.join(RUN, "work", "svc5b.zip")

meta = json.load(open(os.path.join(FOLDER, "dataset-metadata.json")))
owner_slug, slug = meta["id"].split("/")
title = meta["title"]
license_name = meta["licenses"][0]["name"]
keywords = meta.get("keywords", [])
description = meta.get("description")

if not os.path.exists(ZIP_PATH):
    with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(FOLDER):
            for fn in files:
                full = os.path.join(root, fn)
                arc = os.path.relpath(full, FOLDER)
                if arc != "dataset-metadata.json":
                    zf.write(full, arc)
print("zip size:", os.path.getsize(ZIP_PATH))

api = KaggleApi()
api._authenticate_with_access_token()
size = os.path.getsize(ZIP_PATH)
mtime = int(os.path.getmtime(ZIP_PATH))

with api.build_kaggle_client() as client:
    req = ApiStartBlobUploadRequest()
    req.type = ApiBlobType.DATASET
    req.name = "svc5b.zip"
    req.content_length = size
    req.last_modified_epoch_seconds = mtime
    resp = client.blobs.blob_api_client.start_blob_upload(req)
create_url = resp.create_url
print("create_url host:", urlparse(create_url).netloc)
put_url = create_url.replace("www.googleapis.com", "storage.googleapis.com")
t0 = time.time()
r = requests.put(put_url, data=open(ZIP_PATH, "rb").read(), timeout=900)
print("PUT status:", r.status_code, "in", round(time.time()-t0, 1), "s")
assert r.status_code == 200, r.text[:500]

cr = ApiCreateDatasetRequest()
cr.title = title
cr.slug = slug
cr.owner_slug = owner_slug
cr.license_name = license_name
cr.description = description
cr.is_private = True
cr.category_ids = keywords
nf = ApiDatasetNewFile()
nf.token = resp.token
cr.files = [nf]

with api.build_kaggle_client() as client:
    out = client.datasets.dataset_api_client.create_dataset(cr)
print("create_dataset:", out.to_dict() if hasattr(out, "to_dict") else out)
