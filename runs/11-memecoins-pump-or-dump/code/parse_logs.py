"""Parse `kaggle kernels logs` concatenated-JSON output into readable streams.
Usage: kaggle kernels logs REF 2>/dev/null > /tmp/x.json; python code/parse_logs.py /tmp/x.json [stream]
"""
import json
import sys


def load(path):
    raw = open(path).read()
    dec = json.JSONDecoder()
    i = 0
    objs = []
    while i < len(raw):
        while i < len(raw) and raw[i] in " \r\n\t":
            i += 1
        if i >= len(raw):
            break
        try:
            o, j = dec.raw_decode(raw, i)
            objs.append(o)
            i = j
        except Exception:
            i += 1
    if objs and isinstance(objs[-1], list):
        objs = objs[-1]
    return [o for o in objs if isinstance(o, dict) and "stream_name" in o]


def main():
    path = sys.argv[1]
    want = sys.argv[2] if len(sys.argv) > 2 else "stdout"
    for o in load(path):
        if o.get("stream_name") == want:
            sys.stdout.write(o.get("data", ""))


if __name__ == "__main__":
    main()
