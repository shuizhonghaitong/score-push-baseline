#!/usr/bin/env python3
"""Write the competition main.py from a policy module."""
import argparse
import re
from pathlib import Path

HEADER = '''"""Smart Buildings submission.

Policy: {cls}
"""
import sys
sys.path.append('/kaggle/input/competitions/neurips-2026-smart-buildings/kaggle_evaluation')

import numpy as np
from kaggle_evaluation import smart_buildings_inference_server

COMPETITION_DATA_FOLDER = '/kaggle/input/competitions/neurips-2026-smart-buildings/'

{code}

smart_buildings_inference_server.SmartBuildingsInferenceServer({cls}).run(
    competition_data_folder=COMPETITION_DATA_FOLDER,
)
'''


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--policy", required=True)
    p.add_argument("--class", dest="cls", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    src = Path(args.policy).read_text()
    m = re.match(r'^\s*"""(.*?)"""\s*\n', src, re.S)
    code = src[m.end():].lstrip() if m else src
    # Kaggle script mode: __future__ imports must come first; the policy body
    # is inlined after the harness imports, so drop them (not needed on 3.13).
    code = "\n".join(
        ln for ln in code.splitlines()
        if "from __future__" not in ln
    ).lstrip()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(HEADER.format(cls=args.cls, code=code.rstrip()))
    print("wrote", out)


if __name__ == "__main__":
    main()
