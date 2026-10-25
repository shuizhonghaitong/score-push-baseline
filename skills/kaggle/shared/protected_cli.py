"""Client commands for the operator-owned run guard."""
import argparse
import json
import os
import time
import uuid

from shared.run_guard import client, GuardError


def main(command, argv):
    if command == "account":
        return account_main(argv)
    parser = argparse.ArgumentParser(prog=f"kaggle {command}", allow_abbrev=False)
    if command == "ledger":
        parser.parse_args(argv)
        print(json.dumps(client({"operation": "events"}), indent=2))
        return 0
    parser.add_argument("competition")
    if command == "submit":
        parser.add_argument("file", help="Local project-relative path; with --notebook use bare output filename only")
        parser.add_argument("-m", "--message", default="Run-scoped baseline")
        parser.add_argument("--notebook")
        parser.add_argument("--version", type=int)
        parser.add_argument("--yes", action="store_true")
        parser.add_argument("--request-id")
        parser.add_argument("--sample")  # local validation remains an explicit separate operation
        args = parser.parse_args(argv)
        if args.notebook:
            artifact = {"kind": "notebook", "notebook": args.notebook,
                        "version": args.version, "file": args.file}
        else:
            artifact = {"kind": "file", "path": args.file}
        if not args.yes:
            print(json.dumps(client({"operation": "submit-preview", "competition": args.competition,
                                     "artifact": artifact, "request_id": args.request_id,
                                     "message": args.message})))
            return 0
        if os.environ.get("KAGGLE_SKILL_READ_ONLY") == "1":
            raise GuardError("Read-only mode")
        request_id = args.request_id or uuid.uuid4().hex
        print(json.dumps({"request_id": request_id}), flush=True)
        result = client({"operation": "submit", "request_id": request_id,
                         "competition": args.competition, "artifact": artifact,
                         "message": args.message})
        print(json.dumps(result))
        return 0 if result.get("event") == "submitted" else 1
    if command == "watch":
        parser.add_argument("--ref", type=int, required=True)
        parser.add_argument("--interval", type=int, default=30)
        parser.add_argument("--timeout", type=int, default=300)
        args = parser.parse_args(argv)
        if args.interval < 1 or args.timeout < 0:
            parser.error("interval must be positive and timeout nonnegative")
        deadline = time.monotonic() + args.timeout
        while True:
            row = client({"operation": "status", "ref": args.ref})
            print(json.dumps(row), flush=True)
            if row["status"] == "COMPLETE":
                return 0
            if row["status"] == "ERROR":
                return 1
            if time.monotonic() >= deadline:
                return 124
            time.sleep(min(args.interval, max(0, deadline - time.monotonic())))
    raise GuardError("Unsupported protected command")


def account_main(argv):
    from shared.guard_account import OPERATIONS
    parser = argparse.ArgumentParser(prog="kaggle account", allow_abbrev=False)
    parser.add_argument("operation", choices=sorted(OPERATIONS))
    parser.add_argument("--competition")
    parser.add_argument("--notebook")
    parser.add_argument("--code", help="Project-relative .py/.ipynb")
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--internet", action="store_true")
    parser.add_argument("--yes", action="store_true")
    parser.add_argument("--request-id")
    args = parser.parse_args(argv)
    request = vars(args).copy()
    if args.operation == "notebook-push":
        if not args.yes:
            print(json.dumps({"dry_run": True, "notebook": args.notebook, "code": args.code}))
            return 0
        request['request_id'] = args.request_id or uuid.uuid4().hex
        print(json.dumps({"request_id": request['request_id']}), flush=True)
    result = client(request)
    print(json.dumps(result, indent=2))
    return 1 if result.get('event') == 'notebook-unknown' else 0
