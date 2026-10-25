#!/usr/bin/env python3
"""Download competition data via the Kaggle MCP signed-URL route."""
import concurrent.futures
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

SKILL = "/mnt/disk3/liuyutong_home/workspace/skill/score-push/skills/kaggle"
sys.path.insert(0, SKILL)
from shared import mcp_client, credentials  # noqa: E402

COMP = "gemma-4-developer-agent"
OUT = Path(sys.argv[1] if len(sys.argv) > 1 else "data")
WORKERS = int(os.environ.get("DL_WORKERS", "6"))
TOKEN = credentials.bearer_token()


def list_files():
    files, token = [], None
    while True:
        req = {"competitionName": COMP, "pageSize": 200}
        if token:
            req["pageToken"] = token
        r = mcp_client.request("list_competition_data_files", req, token=TOKEN)
        obj = json.loads(r.response["result"]["content"][0]["text"])
        files += obj.get("files", [])
        token = obj.get("next_page_token")
        if not token:
            break
    return files


def fetch_url(name):
    r = mcp_client.request(
        "download_competition_data_file",
        {"competitionName": COMP, "fileName": name},
        token=TOKEN,
    )
    obj = json.loads(r.response["result"]["content"][0]["text"])
    return obj["url"]


def download_one(item):
    name = item["name"]
    size = int(item["total_bytes"])
    dest = OUT / name
    if dest.exists() and dest.stat().st_size == size:
        return name, "skip"
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    last_err = None
    for attempt in range(4):
        try:
            url = fetch_url(name)
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=300) as resp, open(tmp, "wb") as fh:
                while True:
                    chunk = resp.read(1024 * 1024)
                    if not chunk:
                        break
                    fh.write(chunk)
            got = tmp.stat().st_size
            if got != size:
                raise IOError(f"size mismatch {got} != {size}")
            os.replace(tmp, dest)
            return name, "ok"
        except Exception as e:  # noqa: BLE001
            last_err = repr(e)[:120]
            time.sleep(2 + 2 * attempt)
    try:
        tmp.unlink(missing_ok=True)
    except OSError:
        pass
    return name, f"FAIL {last_err}"


def main():
    files = list_files()
    print(f"{len(files)} files", flush=True)
    total = sum(int(f["total_bytes"]) for f in files)
    print(f"total {total/2**30:.1f} GiB", flush=True)
    done = 0
    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {pool.submit(download_one, f): f for f in files}
        for fut in concurrent.futures.as_completed(futures):
            name, status = fut.result()
            done += 1
            if status != "ok" and status != "skip":
                print(f"[{done}/{len(files)}] {name}: {status}", flush=True)
            elif done % 25 == 0 or done == len(files):
                el = time.time() - t0
                print(f"[{done}/{len(files)}] {name}: {status}  ({el:.0f}s)", flush=True)
    print("ALL DONE" if done == len(files) else "DONE WITH FAILURES", flush=True)


if __name__ == "__main__":
    main()
