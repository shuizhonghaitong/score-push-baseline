"""Submit a competition file with the GCS upload-host rewrite.

www.googleapis.com is unreachable on this network route but
storage.googleapis.com works, and the Kaggle signed-upload URL is accepted
by the latter host. Usage:
  python submit_rewritten.py <competition> <file> -m "message"
Prints the submission ref on success.
"""
import argparse, json, sys, time, warnings
warnings.filterwarnings("ignore")
import requests


def bounded_send(self, request, **kw):
    kw["timeout"] = (15, 120)
    if "www.googleapis.com" in request.url:
        request.url = request.url.replace("www.googleapis.com", "storage.googleapis.com")
    return _orig_send(self, request, **kw)


_orig_send = requests.Session.send
requests.Session.send = bounded_send


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("competition")
    ap.add_argument("file")
    ap.add_argument("-m", "--message", default="autonomous run submission")
    a = ap.parse_args()
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.with_retry = lambda function, *x, **kw: function
    api.authenticate()
    t0 = time.time()
    result = api.competition_submit(a.file, a.message, a.competition, quiet=True)
    print(json.dumps({"ref": result.ref, "message": result.message, "secs": round(time.time()-t0, 1)}))


if __name__ == "__main__":
    main()
