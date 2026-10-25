#!/usr/bin/env python3
"""Chunked parallel snapshot downloader (per-file range splits)."""
import concurrent.futures, json, os, subprocess, sys, time
from pathlib import Path
SKILL = "/mnt/disk3/liuyutong_home/workspace/skill/score-push/skills/kaggle"
sys.path.insert(0, SKILL)
from shared import mcp_client, credentials  # noqa: E402
COMP = "gemma-4-developer-agent"
OUT = Path(__file__).resolve().parents[1] / "data"
TOKEN = credentials.bearer_token()
FILES = int(os.environ.get("SNAP_FILES", "6"))
CHUNKS = int(os.environ.get("SNAP_CHUNKS", "4"))


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
    return [f for f in files if f["name"].startswith("snapshots/")]


def url_for(name):
    r = mcp_client.request("download_competition_data_file", {"competitionName": COMP, "fileName": name}, token=TOKEN)
    return json.loads(r.response["result"]["content"][0]["text"])["url"]


def one(item):
    name, size = item["name"], int(item["total_bytes"])
    dest = OUT / name
    if dest.exists() and dest.stat().st_size == size:
        return name, "skip"
    if dest.with_suffix(".tgz.partB").exists():
        return name, "busy"
    try:
        u = url_for(name)
    except Exception as e:
        return name, f"url-err {e!r}"
    parts = []
    chunk = -(-size // CHUNKS)
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=CHUNKS) as ex:
            futs = []
            for i in range(CHUNKS):
                s = i * chunk
                e = min(s + chunk, size) - 1
                if s >= size:
                    break
                p = dest.with_suffix(f".tgz.partB.{i}")
                parts.append(p)
                futs.append(ex.submit(subprocess.run,
                                      ["curl", "-s", "--max-time", "900", "-r", f"{s}-{e}", "-o", str(p), u],
                                      check=True))
            for f in futs:
                f.result()
        got = sum(p.stat().st_size for p in parts)
        if got != size:
            raise IOError(f"size {got} != {size}")
        with open(dest, "wb") as out:
            for p in parts:
                out.write(p.read_bytes())
                p.unlink()
        return name, "ok"
    except Exception as e:
        for p in parts:
            p.unlink(missing_ok=True)
        return name, f"FAIL {e!r}"


def main():
    files = [f for f in list_files()
             if not (OUT / f["name"]).exists() or (OUT / f["name"]).stat().st_size != int(f["total_bytes"])]
    print(f"{len(files)} pending", flush=True)
    t0 = time.time()
    with concurrent.futures.ThreadPoolExecutor(max_workers=FILES) as pool:
        for name, status in pool.map(one, files):
            print(name, status, f"{time.time()-t0:.0f}s", flush=True)
    print("CHUNKED DONE", flush=True)


if __name__ == "__main__":
    main()
