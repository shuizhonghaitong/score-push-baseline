"""Generate a Kaggle notebook (ipynb) from pipeline.py with a config header cell."""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PIPE = open(os.path.join(HERE, "pipeline.py")).read()

TEMPLATE = '''import os, sys, time

# ==== CONFIG ====
{cfg}
print("config:", {cfg_print})
t_start = time.time()
'''

def main():
    # args: output_dir mode [KEY=VAL ...]
    out_dir = sys.argv[1]
    mode = sys.argv[2]
    extra = sys.argv[3:]
    os.makedirs(out_dir, exist_ok=True)
    cfg_lines = [f'os.environ["MODE"] = "{mode}"',
                 'os.environ["OUT_DIR"] = "/kaggle/working"',
                 'os.environ["HISTORY_DAYS"] = "0"']
    for kv in extra:
        k, v = kv.split("=", 1)
        cfg_lines.append(f'os.environ["{k}"] = "{v}"')
    src = TEMPLATE.format(cfg="\n".join(cfg_lines), cfg_print='os.environ["MODE"]') + "\n" + PIPE
    nb = {
        "cells": [
            {"cell_type": "markdown", "metadata": {},
             "source": ["# Memecoins Pump-or-Dump pipeline\n",
                        "Trains from scratch on the attached window. No pretrained weights, no internet.",
                        "Modes: smoke / wf / submit / both (MODE env)."]},
            {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
             "source": src.splitlines(keepends=True)},
        ],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }
    path = os.path.join(out_dir, "solution.ipynb")
    json.dump(nb, open(path, "w"), indent=1)
    print("wrote", path, os.path.getsize(path), "bytes")

if __name__ == "__main__":
    main()
