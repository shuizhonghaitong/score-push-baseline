#!/usr/bin/env python3
"""Assemble the Kaggle submission notebook from a policy module.

Usage:
  python tools/make_notebook.py --policy policies/v2_policy.py --class V2Policy \
      --name submission-v2 --out notebooks/submission-v2.ipynb [--message "notebook message"]
"""
import argparse
import json
import re
from pathlib import Path

HEADER = '''"""Smart Buildings submission.

Policy: {policy_class}
{docstring}
"""
import sys
sys.path.append('/kaggle/input/competitions/neurips-2026-smart-buildings/kaggle_evaluation')

import numpy as np
from kaggle_evaluation import smart_buildings_inference_server

COMPETITION_DATA_FOLDER = '/kaggle/input/competitions/neurips-2026-smart-buildings/'

{policy_code}

smart_buildings_inference_server.SmartBuildingsInferenceServer({policy_class}).run(
    competition_data_folder=COMPETITION_DATA_FOLDER,
)
'''


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--policy", required=True)
    p.add_argument("--class", dest="class_name", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    policy_src = Path(args.policy).read_text()
    doc_match = re.match(r'^"""(.*?)"""', policy_src, re.S)
    docstring = (doc_match.group(1).strip() if doc_match else "").replace("\n", "\n")

    # Strip the module docstring from the inlined code.
    policy_code = policy_src
    if doc_match:
        policy_code = policy_src[doc_match.end():].lstrip()

    code = HEADER.format(
        policy_class=args.class_name,
        docstring=docstring,
        policy_code=policy_code.rstrip(),
    )

    nb = {
        "cells": [
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": code.splitlines(keepends=True),
            }
        ],
        "metadata": {
            "kernelspec": {"language": "python", "display_name": "Python 3", "name": "python3"},
            "language_info": {"name": "python"},
            "kaggle": {
                "accelerator": "none",
                "dataSources": [],
                "dockerImageVersionId": 28755,
                "isInternetEnabled": False,
                "language": "python",
                "sourceType": "notebook",
                "isGpuEnabled": False,
            },
        },
        "nbformat": 4,
        "nbformat_minor": 4,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(nb, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
