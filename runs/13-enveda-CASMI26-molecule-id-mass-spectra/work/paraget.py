"""Robust parallel ranged downloader for signed GCS URLs (flaky proxy tolerant)."""
import json, subprocess, sys, os, shutil, time

PROXY = "http://172.28.4.16:18999"
FALLBACK_SIZE = {"train.parquet": 3033286496}

def probe_size(url):
    for attempt in range(15):
        r = subprocess.run(["curl","-sS","-D","-","-o","/dev/null","-r","0-1023","-x",PROXY, url],
                           capture_output=True, text=True, timeout=90)
        size = None
        for line in r.stdout.splitlines():
            if line.lower().startswith("content-range:"):
                size = int(line.split("/")[-1].strip())
            elif line.lower().startswith("x-goog-stored-content-length:"):
                size = int(line.split(":")[1].strip())
        if size:
            return size
        time.sleep(5)
    return None

def fetch_part(url, part, a, b):
    want = b - a + 1
    for attempt in range(100000):
        if os.path.exists(part):
            have = os.path.getsize(part)
            if have >= want:
                return True
            rng = f"{a + have}-{b}"
            cont = "-C", str(have)
        else:
            rng = f"{a}-{b}"
            cont = None
        cmd = ["curl","-sS","-f","-r",rng,"-x",PROXY,"-o",part,"-w","%{http_code}\n", url]
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)
        except subprocess.TimeoutExpired:
            r = None
        if r is not None and r.returncode == 0 and os.path.exists(part) and os.path.getsize(part) == want:
            return True
        err = (r.stderr[:100] if r else "timeout")
        print(f"[part {part}] attempt {attempt+1} failed: {err}", flush=True)
        time.sleep(10 + (attempt % 20) * 5)
    return False

def main():
    urls = json.load(open("/tmp/casmi_urls.json"))
    url, out = sys.argv[1], sys.argv[2]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    tmp = out + ".parts"
    os.makedirs(tmp, exist_ok=True)
    size = probe_size(url)
    if not size:
        size = FALLBACK_SIZE.get(os.path.basename(out.split(".parquet")[0] + ".parquet") if out.endswith(".parquet") else out)
    assert size, f"cannot determine size for {url[:80]}"
    print(f"size={size} parts={n}", flush=True)
    chunk = (size + n - 1) // n
    import concurrent.futures as cf
    def fetch(i):
        a = i * chunk
        b = min(size - 1, a + chunk - 1)
        part = os.path.join(tmp, f"{i:03d}.bin")
        ok = fetch_part(url, part, a, b)
        print(f"[{i}] {'OK' if ok else 'FAILED'}", flush=True)
        return ok
    with cf.ThreadPoolExecutor(n) as ex:
        res = list(ex.map(fetch, range(n)))
    if not all(res):
        print("SOME PARTS FAILED"); sys.exit(1)
    if os.path.exists(out):
        os.remove(out)
    with open(out, "wb") as fo:
        for i in range(n):
            with open(os.path.join(tmp, f"{i:03d}.bin"), "rb") as fi:
                shutil.copyfileobj(fi, fo, length=1 << 24)
    print("assembled", out, os.path.getsize(out), flush=True)
    shutil.rmtree(tmp, ignore_errors=True)

if __name__ == "__main__":
    main()
