#!/usr/bin/env python3
"""Upload a submission file via Kaggle MCP and (optionally) submit it.

Usage:
  python mcp_submit.py upload <file> <description>   # upload blob, print token
  python mcp_submit.py submit <file> <description>   # upload + submit_to_competition
"""
import sys, os, json, time
from pathlib import Path

SKILL = "/mnt/disk3/liuyutong_home/workspace/skill/score-push/skills/kaggle/shared"
sys.path.insert(0, SKILL)
import mcp_client, credentials, net

COMP = "2026-ieee-big-data-traffic-flow-bench"

def call(tool, req, timeout=120):
    tok = credentials.bearer_token()
    for attempt in range(4):
        resp = mcp_client.mcp_call(tool, req, token=tok, timeout=timeout)
        cls = mcp_client.classify_result(resp)
        if cls == "ok":
            return mcp_client.extract_json(resp)
        print(f"[{tool}] attempt {attempt+1}: {cls}", file=sys.stderr)
        if "connection" in cls or "timeout" in cls:
            time.sleep(3 * (attempt + 1))
            continue
        print(mcp_client.extract_text(resp)[:500], file=sys.stderr)
        raise SystemExit(f"failed: {cls}")
    raise SystemExit(f"failed after retries: {tool}")

def upload_file(path: Path, description: str):
    size = path.stat().st_size
    mtime = int(path.stat().st_mtime)
    info = call("start_competition_submission_upload", {"request": {
        "competitionName": COMP, "hasCompetitionName": True,
        "contentLength": size, "lastModifiedEpochSeconds": mtime,
        "fileName": path.name,
    }}, timeout=90)
    create_url = info["create_url"].replace("https://www.googleapis.com/", "https://storage.googleapis.com/")
    blob_token = info["token"]
    headers = {"Content-Length": "0", "Content-Range": f"bytes */{size}",
               "Content-Type": "application/octet-stream"}
    r = net.request("POST", create_url, headers=headers, data=b"", timeout=120)
    if r.status not in (200, 308):
        raise SystemExit(f"resumable init failed: HTTP {r.status} {r.text[:300]}")
    data = path.read_bytes()
    h2 = {"Content-Length": str(size), "Content-Range": f"bytes 0-{size-1}/{size}",
          "Content-Type": "application/octet-stream"}
    r2 = net.request("PUT", create_url, headers=h2, data=data, timeout=1800)
    if r2.status not in (200, 201):
        raise SystemExit(f"upload PUT failed: HTTP {r2.status} {r2.text[:300]}")
    print(json.dumps({"blob_token": blob_token, "size": size}))
    return blob_token

def main():
    mode = sys.argv[1]
    path = Path(sys.argv[2]).expanduser().resolve()
    description = sys.argv[3]
    if not path.exists():
        raise SystemExit(f"no such file: {path}")
    blob = upload_file(path, description)
    if mode == "upload":
        return
    res = call("submit_to_competition", {"request": {
        "competitionName": COMP,
        "blobFileTokens": blob,
        "submissionDescription": description,
        "hasSubmissionDescription": True,
    }}, timeout=180)
    print("SUBMITTED:", json.dumps(res)[:800])

if __name__ == "__main__":
    main()
