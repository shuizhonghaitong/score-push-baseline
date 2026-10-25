#!/usr/bin/env python3
"""Offline inventory and structural checks; never imports or calls Kaggle."""
from __future__ import annotations

import argparse
import ast
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import shutil
import sys
from urllib.parse import urlsplit


def safe_endpoint(raw):
    try:
        p = urlsplit(raw)
        if not p.scheme or not p.hostname:
            return "set (unparseable; value omitted)"
        host = f"[{p.hostname}]" if ":" in p.hostname else p.hostname
        return f"{p.scheme}://{host}" + (f":{p.port}" if p.port else "")
    except ValueError:
        return "set (unparseable; value omitted)"


def source(path):
    """Protected push generates metadata; validate source without inventing a metadata file."""
    try:
        raw = path.read_bytes()
        if path.suffix == ".py":
            ast.parse(raw)
        elif path.suffix == ".ipynb":
            nb = json.loads(raw)
            if not isinstance(nb, dict) or nb.get("nbformat") != 4 or not isinstance(nb.get("cells"), list):
                raise ValueError
        else:
            raise ValueError
        return {"ok": True, "code_sha256": hashlib.sha256(raw).hexdigest(),
                "limits": "Source structure only; protected service generates metadata; no runtime or account verification"}
    except (OSError, ValueError, SyntaxError):
        return {"ok": False, "errors": ["Expected readable Python source or nbformat=4 .ipynb"]}


def inputs(root, required):
    if not root.is_dir() or any(Path(n).name != n or n in ("", ".", "..") for n in required):
        return {"ok": False, "errors": ["Use an existing root and bare required filenames"]}
    matches = sorted({str(p.parent.resolve()) for p in root.rglob(required[0])
                      if all((p.parent / n).is_file() for n in required)})
    return {"ok": len(matches) == 1, "matches": matches,
            "errors": [] if len(matches) == 1 else ["Missing or ambiguous input directory"],
            "limits": "Filename discovery only; validate schema before using the selected directory"}


def runtime() -> dict:
    packages = {}
    for name in ("kaggle", "kagglehub", "numpy", "pandas", "scikit-learn", "torch", "pyarrow"):
        try:
            dist = importlib.metadata.distribution(name)
            packages[name] = {"version": dist.version,
                              "requires_python": dist.metadata.get("Requires-Python")}
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    proxies = {}
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy",
                 "ALL_PROXY", "all_proxy"):
        raw = os.environ.get(name)
        if not raw:
            continue
        try:
            parsed = urlsplit(raw)
            host = parsed.hostname
            if not parsed.scheme or not host:
                raise ValueError
            if ":" in host:
                host = f"[{host}]"
            port = f":{parsed.port}" if parsed.port else ""
            proxies[name] = f"{parsed.scheme}://{host}{port}"
        except ValueError:
            proxies[name] = "set (unparseable; value omitted)"
    return {"mode": "offline_inventory", "python": sys.executable,
            "python_version": platform.python_version(),
            "kaggle_executable": shutil.which("kaggle"), "packages": packages,
            "proxy_endpoints": proxies,
            "no_proxy_present": bool(os.environ.get("NO_PROXY") or os.environ.get("no_proxy")),
            "tools": {name: shutil.which(name) for name in ("bash", "awk", "which", "nvidia-smi")},
            "gpu_note": "torch presence does not establish device availability; inventory is not a GPU test",
            "pip_index": safe_endpoint(os.environ.get("PIP_INDEX_URL", "https://pypi.org/simple")),
            "proxy_case_match": all(os.environ.get(upper) == os.environ.get(upper.lower())
                                    for upper in ("HTTP_PROXY", "HTTPS_PROXY")),
            "verified_network": False, "verified_credentials": False}


def notebook(folder: Path, competition: str | None, offline: bool,
             allow_public: bool) -> dict:
    folder = folder.resolve()
    errors, notes = [], []
    try:
        meta = json.loads((folder / "kernel-metadata.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"ok": False, "errors": ["metadata is missing, unreadable or invalid JSON"]}
    if not isinstance(meta, dict):
        return {"ok": False, "errors": ["metadata must be an object"]}
    ref = meta.get("id")
    valid_ref = isinstance(ref, str) and bool(re.fullmatch(r"[a-zA-Z0-9_-]+/[a-z0-9_-]+", ref))
    if not valid_ref:
        errors.append("id must be owner/slug")
    title = meta.get("title")
    if not isinstance(title, str) or not title.strip():
        errors.append("title is required")
    elif valid_ref and title.isascii():
        slug = re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-")
        if slug != ref.split("/", 1)[1]:
            errors.append("ASCII title-derived slug differs from id; align them before push")
    elif isinstance(title, str) and not title.isascii():
        notes.append("non-ASCII title slug not checked; verify canonical ref returned by push")
    kind = meta.get("kernel_type")
    if kind not in ("script", "notebook"):
        errors.append("kernel_type must be script or notebook")
    if meta.get("language") != "python":
        errors.append("this checker supports language=python only")
    for key in ("is_private", "enable_gpu", "enable_tpu", "enable_internet"):
        if type(meta.get(key)) is not bool:
            errors.append(f"{key} must be an explicit boolean")
    if not allow_public and meta.get("is_private") is False:
        errors.append("public notebook requires explicit --allow-public")
    if meta.get("enable_gpu") is True and meta.get("enable_tpu") is True:
        errors.append("GPU and TPU cannot both be enabled")
    if offline and meta.get("enable_internet") is not False:
        errors.append("offline contract requires enable_internet=false")
    for key in ("competition_sources", "dataset_sources", "kernel_sources", "model_sources"):
        value = meta.get(key, [])
        if not isinstance(value, list) or any(not isinstance(s, str) for s in value):
            errors.append(f"{key} must be a list of strings")
    sources = meta.get("competition_sources", [])
    if competition and (not isinstance(sources, list) or competition not in sources):
        errors.append("requested competition is absent from competition_sources")
    code = meta.get("code_file")
    code_hash = None
    if not isinstance(code, str) or not code:
        errors.append("code_file is required")
    else:
        path = (folder / code).resolve()
        if not path.is_relative_to(folder):
            errors.append("code_file resolves outside notebook directory")
        elif path.suffix != {"script": ".py", "notebook": ".ipynb"}.get(kind):
            errors.append("code_file extension does not match kernel_type")
        elif not path.is_file():
            errors.append("code_file does not exist")
        else:
            try:
                raw = path.read_bytes()
                if kind == "script":
                    ast.parse(raw, filename="code_file.py")
                else:
                    nb = json.loads(raw)
                    if (not isinstance(nb, dict) or nb.get("nbformat") != 4
                            or not isinstance(nb.get("cells"), list)):
                        errors.append("code_file must be a notebook with nbformat=4 and cells")
                code_hash = hashlib.sha256(raw).hexdigest()
            except (OSError, ValueError, SyntaxError):
                errors.append("code_file is unreadable, invalid JSON or invalid Python syntax")
    return {"ok": not errors, "errors": errors, "notes": notes,
            "code_sha256": code_hash,
            "limits": "structural only; no prediction, account or remote-runtime verification"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="command", required=True)
    subs.add_parser("runtime")
    code = subs.add_parser("source", help="Protected push: check source without local metadata")
    code.add_argument("file", type=Path)
    inp = subs.add_parser("inputs", help="Find exactly one directory containing required filenames")
    inp.add_argument("root", type=Path)
    inp.add_argument("required", nargs="+")
    nb = subs.add_parser("notebook")
    nb.add_argument("folder", type=Path)
    nb.add_argument("--competition")
    nb.add_argument("--offline", action="store_true")
    nb.add_argument("--allow-public", action="store_true")
    args = parser.parse_args()
    if args.command == "runtime":
        result = runtime()
    elif args.command == "source":
        result = source(args.file)
    elif args.command == "inputs":
        result = inputs(args.root, args.required)
    else:
        result = notebook(args.folder, args.competition, args.offline, args.allow_public)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("ok", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
