"""Download and verify the 3LC starter kit from the official CDN.

Mimics the plugin's downloader (3lc-compute-plugin-kaggle v1.2.15):
fetch manifest.json, download each shard archive (with Range resume),
verify sha256 of every archive, extract, then verify every file against
the manifest (bytes + sha256).
"""
import hashlib
import json
import sys
import time
import urllib.request
import zipfile
from pathlib import Path

BASE = "https://competitions.3lc.ai/kaggle/exdark-low-light/starter-kit/v3"
ROOT = Path(__file__).resolve().parent.parent
SHARDS = ROOT / "data" / "shards"
EXTRACT = ROOT / "data"
CHUNK = 1 << 20


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def fetch(name: str, dest: Path, size_hint: int | None, retries: int = 4) -> None:
    url = f"{BASE}/{name}"
    for attempt in range(retries + 1):
        start = dest.stat().st_size if dest.exists() else 0
        if size_hint and start >= size_hint:
            return
        headers = {"Range": f"bytes={start}-"} if start else {}
        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=60) as resp, dest.open("r+b" if start else "wb") as f:
                if start:
                    if resp.status != 206:
                        dest.unlink()
                        start = 0
                        req = urllib.request.Request(url)
                        with urllib.request.urlopen(req, timeout=60) as resp2, dest.open("wb") as f:
                            while True:
                                data = resp2.read(CHUNK)
                                if not data:
                                    break
                                f.write(data)
                else:
                    while True:
                        data = resp.read(CHUNK)
                        if not data:
                            break
                        f.write(data)
            if size_hint and dest.stat().st_size != size_hint:
                print(f"  {name}: size {dest.stat().st_size} != {size_hint}, redownload", flush=True)
                dest.unlink()
                continue
            print(f"  {name}: {dest.stat().st_size} bytes", flush=True)
            return
        except Exception as exc:
            print(f"  {name}: attempt {attempt} failed: {exc}", flush=True)
            time.sleep(3 * (attempt + 1))
    raise RuntimeError(f"could not download {name}")


def main() -> None:
    manifest_path = ROOT / "data" / "kit-manifest-cdn.json"
    manifest = json.loads(manifest_path.read_text())
    SHARDS.mkdir(parents=True, exist_ok=True)
    archives = manifest["archives"]
    for arc in archives:
        name = arc["name"]
        dest = SHARDS / name
        want = arc.get("sha256")
        if dest.exists():
            have = sha256_file(dest)
            if want and have == want:
                print(f"  {name}: cached, sha ok", flush=True)
                continue
            print(f"  {name}: sha mismatch, redownload", flush=True)
            dest.unlink()
        fetch(name, dest, arc.get("bytes"))
        if want:
            got = sha256_file(dest)
            if got != want:
                raise RuntimeError(f"sha256 mismatch for {name}: {got} != {want}")
            print(f"  {name}: sha256 verified", flush=True)
    # extract
    for arc in archives:
        zp = SHARDS / arc["name"]
        t0 = time.time()
        with zipfile.ZipFile(zp) as zf:
            bad = zf.testzip()
            if bad:
                raise RuntimeError(f"corrupt zip member {bad} in {zp.name}")
            zf.extractall(EXTRACT)
        print(f"  extracted {arc['name']} in {time.time()-t0:.1f}s", flush=True)
    # verify every file
    root_prefix = manifest["kit_dir_name"]
    checked = 0
    t0 = time.time()
    for entry in manifest["files"]:
        p = EXTRACT / entry["path"]
        if not p.exists():
            print("MISSING", entry["path"], flush=True)
            continue
        if p.stat().st_size != entry["bytes"]:
            print("SIZE MISMATCH", entry["path"], flush=True)
            continue
        if sha256_file(p) != entry["sha256"]:
            print("SHA MISMATCH", entry["path"], flush=True)
            continue
        checked += 1
        if checked % 2000 == 0:
            print(f"  verified {checked} files ({time.time()-t0:.0f}s)", flush=True)
    print(f"DONE: {checked}/{manifest['file_count']} files verified", flush=True)
    if checked != manifest["file_count"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
