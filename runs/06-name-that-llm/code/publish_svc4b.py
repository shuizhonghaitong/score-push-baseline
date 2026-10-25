import json, os, sys, time, traceback
sys.path.insert(0, "/mnt/disk3/liuyutong_home/miniconda3/envs/kaggle/lib/python3.11/site-packages")
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.blobs.types.blob_api_service import ApiBlobType, ApiStartBlobUploadRequest
from kagglesdk.datasets.types.dataset_api_service import ApiCreateDatasetRequest, ApiDatasetNewFile
import requests

RUN = "/mnt/disk3/liuyutong_home/workspace/skill/score-push/runs/06-name-that-llm"
ZIP_PATH = os.path.join(RUN, "work", "svc4.zip")
meta = json.load(open(os.path.join(RUN, "work", "svc4", "dataset-metadata.json")))
owner_slug, slug = meta["id"].split("/")
api = KaggleApi()

def create(url, token):
    cr = ApiCreateDatasetRequest()
    cr.title = meta["title"]; cr.slug = slug; cr.owner_slug = owner_slug
    cr.license_name = meta["licenses"][0]["name"]; cr.description = meta.get("description")
    cr.is_private = True; cr.category_ids = meta.get("keywords", [])
    nf = ApiDatasetNewFile(); nf.token = token; cr.files = [nf]
    # hit the raw endpoint to see status/body
    import kagglesdk.kaggle_http_client as khc
    from kagglesdk.datasets.types.dataset_api_service import ApiCreateDatasetResponse

for attempt in range(4):
    try:
        with api.build_kaggle_client() as client:
            req = ApiStartBlobUploadRequest()
            req.type = ApiBlobType.DATASET
            req.name = "svc4.zip"
            req.content_length = os.path.getsize(ZIP_PATH)
            req.last_modified_epoch_seconds = int(os.path.getmtime(ZIP_PATH))
            resp = client.blobs.blob_api_client.start_blob_upload(req)
        create_url = resp.create_url
        put_url = create_url.replace("www.googleapis.com", "storage.googleapis.com")
        r = requests.put(put_url, data=open(ZIP_PATH, "rb").read(), timeout=900)
        print("attempt", attempt, "PUT:", r.status_code)
        if r.status_code != 200:
            time.sleep(5); continue
        with api.build_kaggle_client() as client:
            import inspect
            cr = ApiCreateDatasetRequest()
            cr.title = meta["title"]; cr.slug = slug; cr.owner_slug = owner_slug
            cr.license_name = meta["licenses"][0]["name"]; cr.description = meta.get("description")
            cr.is_private = True; cr.category_ids = meta.get("keywords", [])
            nf = ApiDatasetNewFile(); nf.token = resp.token; cr.files = [nf]
            out = client.datasets.dataset_api_client.create_dataset(cr)
        d = out.to_dict()
        print("attempt", attempt, "create ok:", {k: v for k, v in d.items() if v and k in ("status", "error", "error_message", "owner_slug", "dataset_slug", "version")})
        if not (out.error or out.error_message):
            print("SUCCESS")
            break
    except Exception as e:
        print("attempt", attempt, "error:", repr(e)[:300])
    time.sleep(10)
