import os, sys, time, zipfile, io, random
sys.path.insert(0, "/mnt/disk3/liuyutong_home/miniconda3/envs/kaggle/lib/python3.11/site-packages")
from kagglehub.clients import build_kaggle_client
from kagglesdk.competitions.types.competition_api_service import (
    ApiDownloadDataFileRequest, ApiListDataFilesRequest)

COMP = "memecoins-pump-or-dump"
OUT = "/mnt/disk3/liuyutong_home/workspace/skill/score-push/runs/11-memecoins-pump-or-dump/data"
os.makedirs(OUT, exist_ok=True)

def get_client():
    return build_kaggle_client()

def call_with_retry(fn, max_wait=28800, base_delay=20, max_rounds=6):
    delay = base_delay
    for _round in range(max_rounds):
        try:
            return fn()
        except Exception as e:
            msg = str(e)
            status = getattr(getattr(e, "response", None), "status_code", None)
            if status == 429 or "429" in msg or "Too Many" in msg:
                ra = None
                try:
                    ra = e.response.headers.get("Retry-After")
                except Exception:
                    pass
                wait = float(ra) if (ra and ra.replace(".","",1).isdigit()) else delay
                wait = min(wait, max_wait)
                print(f"  429 rate-limited (retry-after {ra}); waiting {wait:.0f}s, round {_round+1}/{max_rounds}", flush=True)
                time.sleep(wait + random.uniform(0, 5))
                delay = min(delay * 1.5, 600)
            else:
                raise
    raise RuntimeError("still 429 after %d full waits; aborting to avoid extending throttle" % max_rounds)

def main():
    only = sys.argv[1] if len(sys.argv) > 1 else None
    client = get_client()
    svc = client.competitions.competition_api_client
    files = []
    page_token = None
    while True:
        r = ApiListDataFilesRequest()
        r.competition_name = COMP
        r.page_size = 200
        if page_token:
            r.page_token = page_token
        resp = call_with_retry(lambda: svc.list_data_files(r))
        files.extend(resp.files)
        page_token = getattr(resp, "next_page_token", None) or None
        if not page_token:
            break
    print(f"listed {len(files)} files", flush=True)
    for f in files:
        name = f.name
        if only and only not in name:
            continue
        dest = os.path.join(OUT, name)
        size = getattr(f, "total_bytes", None) or 0
        if os.path.exists(dest) and (size == 0 or os.path.getsize(dest) == size):
            print(f"skip {name} (exists)", flush=True)
            continue
        print(f"downloading {name} ({size} bytes)", flush=True)
        url = getattr(f, "url", None)
        if not url:
            req = ApiDownloadDataFileRequest()
            req.competition_name = COMP
            req.file_name = name
            redir = call_with_retry(lambda: svc.download_data_file(req))
            url = redir.url
        # download bytes
        import requests
        d = 5
        while True:
            try:
                with requests.get(url, stream=True, timeout=120) as r2:
                    r2.raise_for_status()
                    tmp = dest + ".part"
                    with open(tmp, "wb") as fo:
                        for chunk in r2.iter_content(1 << 20):
                            fo.write(chunk)
                if name.endswith(".zip"):
                    with zipfile.ZipFile(tmp) as zf:
                        zf.extractall(OUT)
                    os.remove(tmp)
                else:
                    os.replace(tmp, dest)
                print(f"done {name}", flush=True)
                break
            except Exception as e:
                if d <= 0:
                    raise
                d -= 1
                print(f"  transfer error {e}; retry in 30s", flush=True)
                time.sleep(30)
    print("ALL DONE", flush=True)

if __name__ == "__main__":
    main()
