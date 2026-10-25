"""Wait for a training job to finish, then run another script with jobpython."""
import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

RUN = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wait", required=True, help="status.json of the job to wait for")
    ap.add_argument("--script", required=True)
    ap.add_argument("--timeout-min", type=float, default=240)
    ap.add_argument("args", nargs=argparse.REMAINDER)
    args = ap.parse_args()
    if args.args and args.args[0] == "--":
        args.args = args.args[1:]

    wait_path = (RUN / args.wait).resolve() if not os.path.isabs(args.wait) else Path(args.wait)
    script = (RUN / args.script).resolve() if not os.path.isabs(args.script) else Path(args.script)

    t0 = time.time()
    while True:
        try:
            state = json.loads(wait_path.read_text())
        except Exception as e:
            print("waiting (read error):", e, flush=True)
            state = {}
        status = state.get("status")
        if status in ("finished", "failed", "cancelled") or status is None and not Path(wait_path).exists():
            print("waited job status:", status, flush=True)
            break
        if time.time() - t0 > args.timeout_min * 60:
            print("TIMEOUT waiting for", wait_path, flush=True)
            sys.exit(5)
        time.sleep(30)

    if status != "finished":
        print("waited job did not finish cleanly:", status, flush=True)
        sys.exit(6)

    cmd = [str(RUN / "tools" / "jobpython"), str(script), *args.args]
    print("running:", " ".join(cmd), flush=True)
    r = subprocess.run(cmd)
    sys.exit(r.returncode)

if __name__ == "__main__":
    main()
