#!/usr/bin/env python3
"""High-parallelism snapshot downloader. Skips files already in progress or done."""
import concurrent.futures, json, os, sys, time, urllib.request
from pathlib import Path
SKILL = "/mnt/disk3/liuyutong_home/workspace/skill/score-push/skills/kaggle"
sys.path.insert(0, SKILL)
from shared import mcp_client, credentials  # noqa: E402
COMP = "gemma-4-developer-agent"
OUT = Path(__file__).resolve().parents[1] / "data"
TOKEN = credentials.bearer_token()
WORKERS = int(os.environ.get("SNAP_WORKERS", "16"))

def list_files():
    files, token = [], None
    while True:
        req = {"competitionName": COMP, "pageSize": 200}
        if token: req["pageToken"] = token
        r = mcp_client.request("list_competition_data_files", req, token=TOKEN)
        obj = json.loads(r.response["result"]["content"][0]["text"])
        files += obj.get("files", [])
        token = obj.get("next_page_token")
        if not token: break
    return [f for f in files if f["name"].startswith("snapshots/")]

def fetch(name):
    r = mcp_client.request("download_competition_data_file", {"competitionName": COMP, "fileName": name}, token=TOKEN)
    return json.loads(r.response["result"]["content"][0]["text"])["url"]

def one(item):
    name, size = item["name"], int(item["total_bytes"])
    dest = OUT / name
    if dest.exists() and dest.stat().st_size == size:
        return name, "skip"
    if dest.with_suffix(".tgz.part").exists():   # main downloader is on it
        return name, "busy"
    tmp = dest.with_suffix(".tgz.partB")
    if tmp.exists():
        tmp.unlink()
    for attempt in range(4):
        try:
            url = fetch(name)
            with urllib.request.urlopen(urllib.request.Request(url), timeout=300) as resp, open(tmp, "wb") as fh:
                while True:
                    c = resp.read(1024*1024)
                    if not c: break
                    fh.write(c)
            if tmp.stat().st_size != size:
                raise IOError("size mismatch")
            os.replace(tmp, dest)
            return name, "ok"
        except Exception as e:  # noqa: BLE001
            time.sleep(2+2*attempt)
    tmp.unlink(missing_ok=True)
    return name, "FAIL"

def main():
    files = list_files()
    print(f"{len(files)} snapshots, {WORKERS} workers", flush=True)
    done = 0; t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as pool:
        for name, status in pool.map(one, files):
            done += 1
            if status not in ("ok","skip","busy"):
                print(name, status, flush=True)
            if done % 10 == 0:
                el = time.time()-t0
                print(f"[{done}/{len(files)}] {el:.0f}s", flush=True)
    print("SNAP PARALLEL DONE", flush=True)

if __name__ == "__main__":
    main()
