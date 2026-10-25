"""Parse kernel log for UPLOAD_URL lines, download files into outputs/fetched/."""
import json, os, re, sys, subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def main():
    log = sys.argv[1]
    outdir = sys.argv[2] if len(sys.argv) > 2 else os.path.join(ROOT, "outputs/fetched")
    os.makedirs(outdir, exist_ok=True)
    events = json.load(open(log))
    text = "".join(e.get("data", "") for e in events)
    urls = re.findall(r"UPLOAD_URL (\S+) (\S+) (https?://\S+)", text)
    if not urls:
        print("no UPLOAD_URL found in", log)
        return
    for tag, host, url in urls:
        print(tag, host, url)
        dl = url
        if "tmpfiles.org/" in url and "/dl/" not in url:
            dl = url.replace("tmpfiles.org/", "tmpfiles.org/dl/")
        dest = os.path.join(outdir, f"{tag}.bin")
        r = subprocess.run(["curl", "-sL", "--max-time", "600", "-o", dest, "-w", "%{http_code} %{size_download}", dl])
        print("  ", r.stdout.decode(), "->", dest)

if __name__ == "__main__":
    main()
