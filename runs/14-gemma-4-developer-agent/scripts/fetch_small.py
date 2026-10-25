#!/usr/bin/env python3
"""Fetch specific non-snapshot files directly (complements the main downloader)."""
import concurrent.futures
import json
import sys
import time
import urllib.request
from pathlib import Path

SKILL = "/mnt/disk3/liuyutong_home/workspace/skill/score-push/skills/kaggle"
sys.path.insert(0, SKILL)
from shared import mcp_client, credentials  # noqa: E402

COMP = "gemma-4-developer-agent"
OUT = Path(__file__).resolve().parents[1] / "data"
TOKEN = credentials.bearer_token()

SKIP_PREFIX = ("snapshots/", "embeddings/", "graphs/")


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


def fetch(name):
    r = mcp_client.request(
        "download_competition_data_file", {"competitionName": COMP, "fileName": name}, token=TOKEN)
    return json.loads(r.response["result"]["content"][0]["text"])["url"]


def one(item):
    name, size = item["name"], int(item["total_bytes"])
    dest = OUT / name
    if dest.exists() and dest.stat().st_size == size:
        return name, "skip"
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part2")
    try:
        url = fetch(name)
        with urllib.request.urlopen(urllib.request.Request(url), timeout=120) as resp, open(tmp, "wb") as fh:
            while True:
                c = resp.read(1024 * 256)
                if not c:
                    break
                fh.write(c)
        os_replace = tmp.rename
        import os
        os.replace(tmp, dest)
        return name, "ok"
    except Exception as e:  # noqa: BLE001
        tmp.unlink(missing_ok=True)
        return name, f"FAIL {e!r}"[:100]


def main():
    files = [f for f in list_files() if not f["name"].startswith(SKIP_PREFIX)]
    print(f"{len(files)} small files", flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
        for name, status in pool.map(one, files):
            if status not in ("ok", "skip"):
                print(name, status, flush=True)
    print("SMALL FILES DONE", flush=True)


if __name__ == "__main__":
    main()
