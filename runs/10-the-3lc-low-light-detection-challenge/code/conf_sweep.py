"""COCO mAP@0.5 on val across a few conf thresholds (score-side maxDets=100)."""
import json
import os
import subprocess
import sys
import time

RUN = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def run_conf(weights, conf):
    t0 = time.time()
    r = subprocess.run([
        os.path.join(RUN, "tools", "jobpython"), os.path.join(RUN, "code", "eval_coco.py"),
        "--weights", weights,
        "--images-dir", os.path.join(RUN, "data/starter_kit/images/val"),
        "--labels-dir", os.path.join(RUN, "data/starter_kit/labels/val"),
        "--conf", str(conf), "--imgsz", "640", "--max-det", "100",
    ], capture_output=True, text=True)
    out = None
    for line in r.stdout.splitlines():
        if line.startswith("EVAL_JSON "):
            out = json.loads(line[len("EVAL_JSON "):])
            break
    if out is None:
        print("FAIL", conf, r.stderr[-400:])
        return
    out["conf"] = conf
    out["elapsed_s"] = round(time.time() - t0, 1)
    print(json.dumps(out), flush=True)

def main():
    weights = sys.argv[1]
    for conf in (0.1, 0.15, 0.2, 0.25, 0.3, 0.35):
        run_conf(weights, conf)

if __name__ == "__main__":
    main()
