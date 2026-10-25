"""Submit a competition file, routing the GCS upload host that Kaggle's API
hands back (www.googleapis.com, unreachable from this network) through the
reachable storage.googleapis.com alias. Bounded timeouts, no retries that
hang. Prints a JSON receipt with the submission ref, or a failure record.

This performs the real submission (consumes one daily slot).
"""
import contextlib
import io
import json
import sys

import requests

_orig_send = requests.Session.send


def _patched_send(session, request, **kwargs):
    host = "storage.googleapis.com"
    bad = "www.googleapis.com"
    if bad in request.url:
        request.url = request.url.replace(bad, host)
    kwargs["timeout"] = (20, 180)
    session.get_adapter(request.url).max_retries = 0
    return _orig_send(session, request, **kwargs)


requests.Session.send = _patched_send


def main():
    file_name, message, slug = sys.argv[1], sys.argv[2], sys.argv[3]
    try:
        with contextlib.redirect_stdout(io.StringIO()) as so, contextlib.redirect_stderr(io.StringIO()) as se:
            from kaggle.api.kaggle_api_extended import KaggleApi
            api = KaggleApi()
            api.with_retry = lambda function, *a, **kw: function
            api.authenticate()
            result = api.competition_submit(file_name, message, slug, quiet=True)
        ref = getattr(result, "ref", None)
        if isinstance(ref, int) and ref > 0:
            print(json.dumps({"outcome": "ok", "ref": ref}))
            return 0
        detail = (getattr(result, "message", None) or so.getvalue() or se.getvalue())[:400]
        print(json.dumps({"outcome": "failed", "ref": None, "detail": detail}))
        return 1
    except Exception as exc:
        print(json.dumps({"outcome": "failed", "ref": None,
                          "error_type": type(exc).__name__, "detail": str(exc)[:400]}))
        return 1


if __name__ == "__main__":
    sys.exit(main())
