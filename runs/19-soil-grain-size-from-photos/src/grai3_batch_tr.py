import base64, json, os, re, sys, time, urllib.request, glob

API = "https://soranz84-grai3api.hf.space/analyze_base64"
LOG = "state/grai3_api_tr.log"
OUT = "state/grai3_preds_tr.json"
testdir = 'data/Training-All_Photos_updated/Training-All_Photos_updated'
imgs = sorted(f for f in glob.glob(testdir + '/**/*', recursive=True)
              if os.path.isfile(f) and f.lower().endswith(('.jpg','.jpeg','.png')))
names = [os.path.basename(f) for f in imgs]

done = {}
if os.path.exists(LOG):
    for line in open(LOG, encoding="utf-8", errors="ignore"):
        m = re.match(r"\d+/\d+ (.+?)\s+b=([\d.]+) c=([\d.]+)", line)
        if m:
            done[m.group(1)] = {"b": float(m.group(2)), "c": float(m.group(3))}
print(f"resuming with {len(done)} already done", flush=True)

records = []
for i, (p, name) in enumerate(zip(imgs, names)):
    if name in done:
        records.append({"file": name, **done[name]})
        continue
    rec = {"file": name}
    data = base64.b64encode(open(p, 'rb').read()).decode()
    req = urllib.request.Request(API, data=json.dumps({"image": data}).encode(),
                                 headers={"Content-Type": "application/json"})
    for attempt in range(8):
        try:
            with urllib.request.urlopen(req, timeout=180) as r:
                j = json.load(r)
            if j.get("status") == "success":
                rec.update({"b": j["b_factor"], "c": j["c_factor"],
                            "d10": j["parameters"]["d10"], "d30": j["parameters"]["d30"],
                            "d60": j["parameters"]["d60"], "supported": j.get("supported_device")})
                break
            rec["err"] = str(j)[:100]
        except Exception as e:
            rec["err"] = f"{type(e).__name__} {str(e)[:80]}"
        time.sleep(20 * (attempt + 1))
    records.append(rec)
    line = f"{i+1}/{len(imgs)} {time.strftime('%H:%M:%S')} {name} b={rec.get('b')} c={rec.get('c')} {rec.get('err','')}"
    print(line, flush=True)
    if "b" in rec:
        with open(LOG, "a") as f:
            f.write(line + "\n")
    time.sleep(3)
json.dump(records, open(OUT, 'w'), indent=1)
print("DONE", len(records), flush=True)
