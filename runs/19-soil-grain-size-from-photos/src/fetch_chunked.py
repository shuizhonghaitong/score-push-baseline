"""Download a URL in Range chunks with retries (proxy truncates large transfers)."""
import sys, time, os, requests

def fetch(url, out, chunk=250000, tries=40):
    ua = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64; rv:126.0) Gecko/20100101 Firefox/126.0"}
    s = requests.Session()
    total = None
    got = 0
    pos = 0
    if os.path.exists(out) and os.path.getsize(out) > 0:
        # restart from existing partial
        pos = os.path.getsize(out)
    for t in range(tries):
        try:
            if total is None and t == 0 and pos == 0:
                r = s.head(url, headers=ua, timeout=(15, 30))
                total = int(r.headers.get("content-length", 0)) or None
            headers = dict(ua)
            if total and pos < total:
                headers["Range"] = f"bytes={pos}-{min(pos + chunk, total - 1)}"
            r = s.get(url, headers=headers, timeout=(15, 90))
            if r.status_code not in (200, 206):
                print(f"  [{t}] http {r.status_code}", flush=True)
                time.sleep(8); continue
            data = r.content
            if r.status_code == 200 and pos > 0:
                # server ignored range; restart fresh
                pos = 0
                mode = "wb"
            else:
                mode = "ab"
            if len(data) == 0:
                time.sleep(5); continue
            with open(out, mode) as f:
                f.write(data)
            pos += len(data)
            if total is None:
                total = pos
            if total and pos >= total:
                print(f"DONE {out} {pos}B in {t+1} tries", flush=True)
                return True
        except Exception as e:
            print(f"  [{t}] {type(e).__name__} {str(e)[:80]}", flush=True)
            time.sleep(5)
    print(f"GAVE UP {out} at {pos}", flush=True)
    return False

if __name__ == "__main__":
    fetch(sys.argv[1], sys.argv[2])
