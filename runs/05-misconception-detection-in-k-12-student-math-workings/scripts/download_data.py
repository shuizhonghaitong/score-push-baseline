import os, sys, json, time, threading, argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd
import requests
from kaggle.api.kaggle_api_extended import KaggleApi, ApiDownloadDataFileRequest

COMP = "misconception-detection-in-k-12-student-math-workings"
PREFIX = "competition_data/"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=10)
    ap.add_argument("--root", default="data")
    ap.add_argument("--retries", type=int, default=4)
    ap.add_argument("--only", default="", help="subset filter: train,test,questions,top")
    args = ap.parse_args()

    root = os.path.abspath(args.root)
    tr = pd.read_csv(os.path.join(root, "train.csv"))
    te = pd.read_csv(os.path.join(root, "test.csv"))

    names = set()
    if not args.only or "train" in args.only:
        names.update(PREFIX + "images/train/" + f for f in tr.image_file)
    if not args.only or "test" in args.only:
        names.update(PREFIX + "images/test/" + f for f in te.image_file)
    if not args.only or "questions" in args.only:
        for f in pd.concat([tr.question_image_file, te.question_image_file]).dropna():
            if str(f).strip():
                names.add(PREFIX + "images/questions/" + str(f))
    if not args.only or "top" in args.only:
        names.update(PREFIX + n for n in ["sample_submission.csv", "taxonomy_v1.csv", "train.csv", "test.csv", "unlabelled.csv"])

    def local_path(name):
        return os.path.join(root, name[len(PREFIX):])

    todo = [n for n in sorted(names) if not (os.path.exists(local_path(n)) and os.path.getsize(local_path(n)) > 0)]
    print(f"needed={len(names)} todo={len(todo)}", flush=True)
    if not todo:
        print("nothing to do", flush=True)
        return

    api = KaggleApi()
    api.authenticate()
    client = api.build_kaggle_client()
    lock = threading.Lock()
    stats = {"ok": 0, "fail": 0}
    faillog = []

    def work(name):
        lp = local_path(name)
        os.makedirs(os.path.dirname(lp), exist_ok=True)
        last_err = None
        for attempt in range(args.retries):
            try:
                req = ApiDownloadDataFileRequest()
                req.competition_name = COMP
                req.file_name = name
                with client:
                    resp = client.competitions.competition_api_client.download_data_file(req)
                url = resp.request.url
                r = requests.get(url, stream=True, timeout=120)
                r.raise_for_status()
                tmp = lp + ".part"
                with open(tmp, "wb") as fh:
                    for chunk in r.iter_content(1 << 20):
                        fh.write(chunk)
                os.replace(tmp, lp)
                with lock:
                    stats["ok"] += 1
                return
            except Exception as e:
                last_err = repr(e)[:200]
                time.sleep(1.5 * (attempt + 1))
        with lock:
            stats["fail"] += 1
            faillog.append(f"{name}\t{last_err}")

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(work, n) for n in todo]
        for i, f in enumerate(as_completed(futs), 1):
            if i % 250 == 0 or i == len(futs):
                dt = time.time() - t0
                print(f"{i}/{len(futs)} ok={stats['ok']} fail={stats['fail']} elapsed={dt:.0f}s", flush=True)
    print(f"DONE ok={stats['ok']} fail={stats['fail']} elapsed={time.time()-t0:.0f}s", flush=True)
    if faillog:
        os.makedirs(os.path.join(root, "..", "logs"), exist_ok=True)
        with open(os.path.join(root, "..", "logs", "download_failures.txt"), "w") as fh:
            fh.write("\n".join(faillog) + "\n")
        print(f"{len(faillog)} failures logged", flush=True)

if __name__ == "__main__":
    main()
