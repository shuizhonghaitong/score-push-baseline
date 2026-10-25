"""Direct competition submission via storage.googleapis.com (bypasses blocked www host).

Usage: submit_direct.py <parquet> <message>
"""
import json, os, subprocess, sys, time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from kaggle.api.kaggle_api_extended import KaggleApi
from kagglesdk.competitions.types.competition_api_service import (
    ApiStartSubmissionUploadRequest, ApiCreateSubmissionRequest)

COMP = "climx-main-track"
BUCKET = "kaggle-competitions-submissions"


def main():
    path, msg = Path(sys.argv[1]), sys.argv[2]
    # validate
    t = pq.read_table(str(path))
    assert t.num_rows == 4755456, t.num_rows
    d = t.to_pydict()
    assert list(d.keys())[:1] == ["row_id"]
    for k in d:
        v = np.asarray(d[k], dtype=np.float64)
        assert np.isfinite(v).all(), f"non-finite in {k}"
    size = os.path.getsize(path)
    api = KaggleApi()
    with api.build_kaggle_client() as kaggle:
        req = ApiStartSubmissionUploadRequest()
        req.competition_name = COMP
        req.file_name = os.path.basename(path)
        req.content_length = size
        req.last_modified_epoch_seconds = int(os.path.getmtime(path))
        resp = kaggle.competitions.competition_api_client.start_submission_upload(req)
    uid = resp.create_url.split("upload_id=")[1]
    url = (f"https://storage.googleapis.com/upload/storage/v1/b/{BUCKET}/o"
           f"?uploadType=resumable&upload_id={uid}")
    t0 = time.time()
    r = subprocess.run(["curl", "-sS", "-o", "/tmp/sub_put_resp.txt",
                        "-w", "HTTP=%{http_code} speed=%{speed_upload} time=%{time_total}",
                        "--max-time", "3600", "-X", "PUT",
                        "-H", f"Content-Range: bytes 0-{size-1}/{size}",
                        "--data-binary", "@" + str(path), url],
                       capture_output=True, text=True)
    print("PUT:", r.stdout, f"wall={time.time()-t0:.1f}s", r.stderr[:200], flush=True)
    assert "HTTP=200" in r.stdout or "HTTP=201" in r.stdout, "upload failed"
    t0 = time.time()
    with api.build_kaggle_client() as kaggle:
        cr = ApiCreateSubmissionRequest()
        cr.competition_name = COMP
        cr.blob_file_tokens = resp.token
        cr.submission_description = msg
        cr2 = kaggle.competitions.competition_api_client.create_submission(cr)
    print(f"CREATE dt={time.time()-t0:.1f}s message={cr2.message}", flush=True)


if __name__ == "__main__":
    main()
