"""Run-scoped operation authority. Run outside the agent's writable namespace.

The socket is the only agent-visible interface. State and credential directories
must not be mounted into that namespace. No endpoint lists account history,
edits events, or declares an uncertain submission absent.
"""
from __future__ import annotations

import hashlib
import fcntl
import json
import os
from pathlib import Path
import shutil
import socket
import socketserver
import tempfile
import threading
import time
import uuid


def error_fields(error):
    response = getattr(error, "response", None)
    code = getattr(response, "status_code", None)
    fields = {"error_type": type(error).__name__, "http_status": code if type(code) is int else None}
    detail = getattr(error, 'safe_transport', None) or getattr(response, 'safe_transport', None)
    if isinstance(detail, dict):
        fields['transport'] = {k: detail[k] for k in ('host', 'method', 'elapsed_seconds', 'exception_chain') if k in detail}
    return fields


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class GuardError(Exception):
    pass


class EventStore:
    def __init__(self, directory):
        self.root = Path(directory).resolve()
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.lockfile = open(self.root / "service.lock", "a")
        try:
            fcntl.flock(self.lockfile, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.file = self.root / "events.jsonl"
            self.rows = []
            if self.file.exists():
                for line in self.file.read_text().splitlines():
                    row = json.loads(line)
                    expected_previous = self.rows[-1]["hash"] if self.rows else ""
                    body = {k: v for k, v in row.items() if k != "hash"}
                    if (body.get("seq") != len(self.rows) or body.get("previous") != expected_previous
                            or digest(body) != row.get("hash")):
                        raise GuardError("Damaged authoritative journal; refusing all writes")
                    self.rows.append(row)
        except BaseException:
            self.lockfile.close()
            raise

    def append(self, event, **fields):
        body = {"seq": len(self.rows), "time": time.time(), "event": event,
                "previous": self.rows[-1]["hash"] if self.rows else "", **fields}
        row = {**body, "hash": digest(body)}
        fd = os.open(self.file, os.O_WRONLY | os.O_CREAT | os.O_APPEND | os.O_NOFOLLOW, 0o600)
        with os.fdopen(fd, "a") as out:
            out.write(json.dumps(row, sort_keys=True) + "\n")
            out.flush()
            os.fsync(out.fileno())
        self.rows.append(row)
        return row


def snapshot_file(project, relative, destination):
    """Walk relative path with openat/O_NOFOLLOW; never trust an agent symlink."""
    if not isinstance(relative, str) or not relative or "\x00" in relative:
        raise GuardError("Artifact must be a project-relative path")
    parts = Path(relative).parts
    if not parts or Path(relative).is_absolute() or any(p in (".", "..") for p in parts):
        raise GuardError("Artifact must be a project-relative path")
    directory = os.open(project, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for part in parts[:-1]:
            next_fd = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=directory)
            os.close(directory)
            directory = next_fd
        fd = os.open(parts[-1], os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=directory)
        with os.fdopen(fd, "rb") as source:
            import stat
            if not stat.S_ISREG(os.fstat(source.fileno()).st_mode):
                raise GuardError("Artifact is not a regular file")
            if destination is None:
                return None  # Read-only preflight: validate traversal/type, do not copy.
            with open(destination, "xb") as target:
                shutil.copyfileobj(source, target)
    finally:
        os.close(directory)
    h = hashlib.sha256()
    with open(destination, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


class Guard:
    def __init__(self, state, project, competition, transport, max_submissions=1, notebooks=()):
        self.store = EventStore(state)
        self.project = Path(project).resolve()
        if self.store.root == self.project or self.project in self.store.root.parents:
            self.store.lockfile.close()
            raise GuardError("Authoritative state must be outside the project")
        self.competition = competition
        self.transport = transport
        self.max_submissions = max_submissions
        self.notebooks = sorted(set(notebooks))
        self.lock = threading.Lock()
        if not self.store.rows:
            self.store.append("run", competition=competition, max_submissions=max_submissions,
                              notebooks=self.notebooks)
        elif self.store.rows[0].get("competition") != competition:
            self.store.lockfile.close()
            raise GuardError("State belongs to a different competition")
        elif self.store.rows[0].get("max_submissions") != max_submissions:
            self.store.lockfile.close()
            raise GuardError("Do not change the frozen submission allowance")
        elif self.store.rows[0].get("notebooks") != self.notebooks:
            self.store.lockfile.close()
            raise GuardError("Do not change the frozen notebook allowlist")

    def call(self, request):
        with self.lock:
            operation = request.get("operation")
            from shared.guard_account import OPERATIONS, account_call
            if operation in OPERATIONS:
                return account_call(self, request)
            if operation == "events":
                return {"events": self.store.rows}
            if operation == "status":
                ref = request.get("ref")
                known = {r["ref"] for r in self.store.rows if r["event"] == "submitted"}
                if type(ref) is not int or ref not in known:
                    raise GuardError("Only receipts created by this run may be queried")
                result = self.transport.status(ref)
                allowed = {k: result.get(k) for k in ("ref", "status", "public_score", "error")}
                if allowed["ref"] != ref:
                    raise GuardError("Response ID mismatch")
                self.store.append("status", **allowed)
                return allowed
            if operation == "submit-preview":
                return self._submit(request, preview=True)
            if operation != "submit":
                raise GuardError("Unsupported operation; history and journal mutation are forbidden")
            return self._submit(request)

    def _submit(self, request, preview=False):
        if os.environ.get("KAGGLE_SKILL_READ_ONLY") == "1":
            raise GuardError("Service is read-only")
        request_id = request.get("request_id")
        if preview and request_id is None:
            request_id = "preview-" + uuid.uuid4().hex
        if not isinstance(request_id, str) or not 1 <= len(request_id) <= 100:
            raise GuardError("A stable request_id is required")
        if request.get("competition") != self.competition:
            raise GuardError("Competition is outside this run")
        rows = self.store.rows
        matching = [r for r in rows if r.get("request_id") == request_id]
        if matching:
            return {"dry_run": True, "would_replay": matching[-1]} if preview else {"replayed": True, **matching[-1]}
        terminal = {r["attempt"] for r in rows if r["event"] in ("submitted", "not_created")}
        if any(r["event"] == "attempt" and r["attempt"] not in terminal for r in rows):
            raise GuardError("Prior attempt is uncertain; no further submission is allowed")
        if sum(r["event"] == "submitted" for r in rows) >= self.max_submissions:
            raise GuardError("This run's authorized submission allowance is exhausted")
        artifact = request.get("artifact")
        if not isinstance(artifact, dict):
            raise GuardError("Missing artifact")
        attempt = uuid.uuid4().hex
        payload = {"competition": self.competition, "message": str(request.get("message", ""))[:1000]}
        if artifact.get("kind") == "file":
            snapshot_file(self.project, artifact.get("path"), None)
            payload.update(kind="file", filename=Path(artifact["path"]).name)
        elif artifact.get("kind") == "notebook":
            ref, version, filename = artifact.get("notebook"), artifact.get("version"), artifact.get("file")
            import re
            if not isinstance(ref, str) or not re.fullmatch(r"[\w-]+/[\w-]+", ref):
                raise GuardError("Invalid notebook handle")
            if ref not in self.notebooks:
                raise GuardError("Notebook is not on this run's operator-owned allowlist")
            if type(version) is not int or version < 1:
                raise GuardError("Pin a positive notebook version")
            if not isinstance(filename, str) or Path(filename).name != filename or filename in ("", ".", ".."):
                raise GuardError("Notebook submission requires a bare output filename, e.g. submission.csv; "
                                 "do not pass a local path such as output/submission.csv")
            known = {(r['notebook'], r['version']) for r in rows if r['event'] == 'notebook-pushed'}
            if (ref, version) not in known:
                raise GuardError("Only Notebook versions created by this run may be submitted")
            payload.update(kind="notebook", notebook=ref, version=version, file=filename)
        else:
            raise GuardError("Unsupported artifact kind")
        if preview:
            return {"dry_run": True, "eligible_now": True, "artifact": payload,
                    "cost": "one submission if created",
                    "note": "No write or remote call; format not checked, eligibility rechecked on submit"}
        if artifact["kind"] == "file":
            folder = self.store.root / attempt
            folder.mkdir(mode=0o700)
            target = folder / "submission"
            sha = snapshot_file(self.project, artifact["path"], target)
            payload.update(path=str(target), sha256=sha)
        # Only the service records outcomes; clients cannot supply a phase or a ref.
        self.store.append("attempt", attempt=attempt, request_id=request_id,
                          artifact={k: v for k, v in payload.items() if k not in ("path", "message")})
        phase = "preflight"

        def progress(value):
            nonlocal phase
            if value not in ("preflight", "upload", "upload_init", "upload_put", "create"):
                raise GuardError("Invalid transport phase")
            phase = value
            self.store.append("phase", attempt=attempt, phase=phase)

        try:
            result = self.transport.submit(payload, progress)
            ref = result.get("ref")
            if type(ref) is not int or ref < 1:
                # A completed transport without a receipt is never proof of absence.
                phase = "create"
                raise GuardError("Missing receipt")
            row = self.store.append("submitted", attempt=attempt, request_id=request_id, ref=ref)
        except Exception as error:
            # Phase is written BEFORE a create call. Process death leaves an unresolved attempt.
            outcome = "unknown" if phase == "create" else "not_created"
            row = self.store.append(outcome, attempt=attempt, request_id=request_id,
                                    **error_fields(error), phase=phase)
        return row


class Server(socketserver.UnixStreamServer):
    def __init__(self, path, guard):
        self.guard = guard
        super().__init__(str(path), Handler)
        os.chmod(path, 0o600)


class Handler(socketserver.StreamRequestHandler):
    def handle(self):
        try:
            self.request.settimeout(5)
            raw = self.rfile.readline(65537)
            if len(raw) > 65536:
                raise GuardError("Request too large")
            request = json.loads(raw)
            if not isinstance(request, dict):
                raise GuardError("Expected an object")
            result = self.server.guard.call(request)
            response = {"ok": True, "result": result}
        except Exception as exc:
            # Never include transport exception text: it can contain credentials/signed URLs.
            message = str(exc) if isinstance(exc, GuardError) else json.dumps(error_fields(exc))
            response = {"ok": False, "error": message}
        self.wfile.write((json.dumps(response) + "\n").encode())


def client(request):
    path = os.environ.get("KAGGLE_RUN_GUARD_SOCKET")
    if not path:
        raise GuardError("Protected operation service is required; no direct fallback")
    with socket.socket(socket.AF_UNIX) as sock:
        sock.settimeout(3700)
        sock.connect(path)
        sock.sendall((json.dumps(request) + "\n").encode())
        with sock.makefile("rb") as stream:
            response = json.loads(stream.readline())
    if not response["ok"]:
        raise GuardError(response["error"])
    return response["result"]
