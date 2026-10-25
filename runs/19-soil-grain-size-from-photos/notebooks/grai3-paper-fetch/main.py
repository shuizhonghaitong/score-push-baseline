import hashlib, io, json, os, re, time
import urllib.parse
import requests

OUT = "/kaggle/output"
os.makedirs(OUT, exist_ok=True)

H = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "sec-ch-ua": '"Chromium";v="125", "Google Chrome";v="125", "Not.A/Brand";v="24"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"Windows"',
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
    "Upgrade-Insecure-Requests": "1",
}

def try_get(url, name, timeout=90, sess=None, headers=None, minbytes=20000):
    t0 = time.time()
    try:
        s = sess or requests
        r = s.get(url, headers=headers or H, timeout=timeout)
        ct = r.headers.get("content-type", "")
        data = r.content
        print(f"[{r.status_code}] {url}  type={ct[:40]} bytes={len(data)} time={time.time()-t0:.0f}s", flush=True)
        if r.status_code == 200 and len(data) >= minbytes and name:
            p = os.path.join(OUT, name)
            with open(p, "wb") as f:
                f.write(data)
            print("  saved ->", p, hashlib.sha256(data).hexdigest()[:16], flush=True)
        elif r.status_code == 200 and name:
            p = os.path.join(OUT, name + ".partial")
            with open(p, "wb") as f:
                f.write(data)
            print("  saved partial ->", p, flush=True)
        return r
    except Exception as e:
        print(f"[ERR] {url}: {type(e).__name__} {str(e)[:120]}", flush=True)
        return None

PII = "S0266352X26004684"
SD = f"https://www.sciencedirect.com/science/article/pii/{PII}"
SSRN = "https://papers.ssrn.com/sol3/papers.cfm?abstract_id=6361302"

print("== 1. BOKU BOR (institutional repository) ==", flush=True)
try_get("https://bor.boku.ac.at/discover?query=GRAI3", "bor_search_grai3.html", minbytes=1000)
try_get("https://bor.boku.ac.at/discover?query=particle%20size%20distribution%20smartphone", "bor_search_psd.html", minbytes=1000)
try_get("https://bor.boku.ac.at/oai/request?verb=Identify", "bor_oai_identify.xml", minbytes=200)
# Pure-based REST search (used by many university repos)
try_get("https://bor.boku.ac.at/rest/search?query=GRAI3", "bor_rest.json", minbytes=100)
# OAI search by title fragment is not standard; try ListRecords filtered (small)
try_get("https://bor.boku.ac.at/oai/request?verb=ListRecords&metadataPrefix=oai_dc&from=2026-01-01", "bor_oai_2026.xml", minbytes=100, timeout=120)

print("== 2. GRAI lab site ==", flush=True)
try_get("https://grai.boku.ac.at/", "grai_home.html", minbytes=500)
try_get("https://grai.boku.ac.at/publications", "grai_pubs.html", minbytes=500)

print("== 3. Wayback machine ==", flush=True)
for label, target in (("sd", SD), ("ssrn", SSRN)):
    api_url = "http://archive.org/wayback/available?url=" + urllib.parse.quote(target, safe="")
    r = try_get(api_url, f"wayback_avail_{label}.json", minbytes=20, headers={"User-Agent": "research-agent"})
    if r is not None and r.status_code == 200:
        try:
            j = r.json()
            snap = j.get("archived_snapshots", {}).get("closest", {})
            print(f"  wayback {label}: {snap.get('url')}", flush=True)
            if snap.get("url"):
                try_get(snap["url"] + "id_/" + target.replace("https://", ""), f"wayback_{label}.html", minbytes=20000, timeout=120)
                try_get(snap["url"].replace("http", "https", 1) + "if_/" + target.replace("https://", ""), f"wayback_{label}_if.html", minbytes=20000, timeout=120)
        except Exception as e:
            print("  wayback parse err:", e, flush=True)
    # also direct web.archive.org crawl guess
    try_get(f"https://web.archive.org/web/2026/{target}", f"wayback_direct_{label}.html", minbytes=20000, timeout=120)

print("== 4. SSRN variants ==", flush=True)
try_get(SSRN, "ssrn_page.html")
try_get("https://ssrn.com/abstract=6361302", "ssrn_alt.html", minbytes=5000)
try_get("https://papers.ssrn.com/sol3/Delivery.cfm?abstractid=6361302&mirid=1", "ssrn_delivery.pdf", minbytes=50000)
try_get("https://papers.ssrn.com/sol3/Delivery.cfm?SSRN_ID=5762069&abstractid=6361302&mirid=1", "ssrn_delivery2.pdf", minbytes=50000)

print("== 5. ScienceDirect with cookie session ==", flush=True)
sess = requests.Session()
sess.headers.update(H)
try:
    r1 = sess.get("https://www.sciencedirect.com/", timeout=60)
    print(f"[{r1.status_code}] SD landing bytes={len(r1.content)}", flush=True)
    time.sleep(3)
    r2 = sess.get(SD, timeout=90)
    print(f"[{r2.status_code}] SD article bytes={len(r2.content)}", flush=True)
    if r2.status_code == 200 and len(r2.content) > 50000:
        open(os.path.join(OUT, "sd_article.html"), "wb").write(r2.content)
        print("  saved sd_article.html", flush=True)
        time.sleep(5)
        for pdf_url in (SD + "/pdf", SD + "/pdf?isDTMRedir=true&download=true"):
            r3 = sess.get(pdf_url, timeout=120)
            print(f"[{r3.status_code}] {pdf_url} bytes={len(r3.content)} type={r3.headers.get('content-type','')[:30]}", flush=True)
            if r3.status_code == 200 and len(r3.content) > 100000 and r3.content[:4] == b"%PDF":
                open(os.path.join(OUT, "grai3_sd.pdf"), "wb").write(r3.content)
                print("  SAVED grai3_sd.pdf", flush=True)
                break
except Exception as e:
    print("SD session err:", e, flush=True)

print("== 6. doi content negotiation ==", flush=True)
try_get("https://doi.org/10.1016/j.compgeo.2026.108362", "doi_negotiated.pdf",
        headers={**H, "Accept": "application/pdf"}, minbytes=50000)

print("== 7. Semantic Scholar + unpaywall (metadata only) ==", flush=True)
r = try_get("https://api.semanticscholar.org/graph/v1/paper/DOI:10.1016/j.compgeo.2026.108362?fields=title,openAccessPdf,isOpenAccess,externalIds,tldr",
           "s2.json", minbytes=100, headers={"User-Agent": "research-agent"})
if r is not None and r.status_code == 200:
    j = r.json()
    oa = j.get("openAccessPdf") or {}
    print("S2 openAccessPdf:", oa.get("url"), flush=True)

print("=== output files ===", flush=True)
for f in sorted(os.listdir(OUT)):
    p = os.path.join(OUT, f)
    print(f, os.path.getsize(p), flush=True)
