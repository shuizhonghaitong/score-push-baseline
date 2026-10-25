#!/usr/bin/env python3
"""One submission attempt, direct receipt, no account-history lookup.

Internal helper called only after the parent wrapper's validation/write gate.
Network deadlines bound an individual operation, not the experiment budget.
"""
import argparse
import contextlib
import io
import json
import os
import sys


def submit_once(api, slug, file_name, message, notebook=None, version=None):
    if notebook:
        result = api.competition_submit_code(
            file_name, message, slug, notebook, int(version), quiet=True)
    else:
        result = api.competition_submit(file_name, message, slug, quiet=True)
    ref = getattr(result, "ref", None)
    if not isinstance(ref, int) or ref <= 0:
        raise RuntimeError("No usable receipt; outcome is unknown, do not resubmit")
    return {"ref": ref}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("competition")
    p.add_argument("file")
    p.add_argument("--message", required=True)
    p.add_argument("--notebook")
    p.add_argument("--version", type=int)
    p.add_argument("--authorized", action="store_true")
    args = p.parse_args()
    if not args.authorized or os.environ.get("KAGGLE_SKILL_READ_ONLY") == "1":
        p.error("Requires the parent wrapper's write authorization")
    for key in ("VERBOSE", "VERBOSE_OUTPUT", "KAGGLE_API_ENVIRONMENT"):
        os.environ.pop(key, None)
    try:
        # Suppress SDK output: exceptions can contain signed upload URLs.
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            import requests
            from urllib3.util.retry import Retry
            original_send = requests.Session.send

            def bounded_send(session, request, **kwargs):
                kwargs["timeout"] = (15, 60)
                session.get_adapter(request.url).max_retries = Retry(total=0)
                return original_send(session, request, **kwargs)

            requests.Session.send = bounded_send
            from kaggle.api.kaggle_api_extended import KaggleApi
            api = KaggleApi()
            api.with_retry = lambda function, *a, **kw: function
            api.authenticate()
            receipt = submit_once(api, args.competition, args.file, args.message,
                                  args.notebook, args.version)
        print(json.dumps(receipt))
        return 0
    except Exception as exc:
        print(json.dumps({"outcome": "unknown", "error_type": type(exc).__name__,
                          "action": "Do not resubmit; reconcile this attempt first"}))
        return 1


if __name__ == "__main__":
    sys.exit(main())
