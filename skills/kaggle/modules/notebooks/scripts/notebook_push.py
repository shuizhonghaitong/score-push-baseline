#!/usr/bin/env python3
"""Push a notebook to Kaggle, which also runs it. A dry run unless --yes is given.

    notebook_push.py ./notebook-dir          dry run: what would be pushed
    notebook_push.py ./notebook-dir --yes    push

The folder holds the notebook and its kernel-metadata.json. Pushing creates a
new version and starts a run on Kaggle: with an accelerator switched on, that
run uses the account's weekly GPU hours. Get the user's go-ahead before --yes.

Kaggle receives the code file named in kernel-metadata.json and the settings
in that file, nothing else from the folder. The code file must be inside the
folder and must not look like a credential file (exit status 5). To wait for
the run and fetch its output, use the notebook-run or notebook-wait command.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

SKILL_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(SKILL_ROOT))

from shared import credentials, kaggle_cli, notebook, preflight, script  # noqa: E402

GPU_WORDS = ("gpu", "nvidia")
SOURCES = (
    ("competition_sources", "competition data"),
    ("dataset_sources", "datasets"),
    ("model_sources", "models"),
    ("kernel_sources", "notebooks"),
)


def plan(folder: Path) -> tuple[str, list[tuple[str, str]], str] | int:
    """What a push of ``folder`` would do: ``(notebook id, details, cost)``, or an exit code."""
    metadata = notebook.read_metadata(folder)
    if metadata is None:
        return script.fail(
            f"{folder / notebook.METADATA_FILE} is missing or unreadable. Write a template "
            f"with: kaggle kernels init -p {folder}",
            script.EXIT_USAGE,
        )
    slug = str(metadata.get("id") or "")
    if not script.is_handle(slug, 2):
        return script.fail(
            f"the id in {notebook.METADATA_FILE} is not owner/name", script.EXIT_USAGE
        )
    code = code_file(folder, metadata)
    if isinstance(code, int):
        return code

    def flag(name: str, default: bool) -> bool:
        value = metadata.get(name, default)
        return value if isinstance(value, bool) else str(value).lower() == "true"

    # kaggle 2.2.4 sends enable_gpu, enable_tpu and machine_shape; a machine
    # shape such as NvidiaTeslaT4 picks the accelerator by itself.
    shape = str(metadata.get("machine_shape") or "")
    tpu = flag("enable_tpu", False) or "tpu" in shape.lower()
    gpu = not tpu and (flag("enable_gpu", False) or any(w in shape.lower() for w in GPU_WORDS))
    accelerator = "TPU" if tpu else "GPU" if gpu else "none"
    details = [
        ("notebook", slug),
        ("code file", f"{code.relative_to(folder.resolve())} ({preflight.size_text(code)})"),
        ("visibility", "private" if flag("is_private", True) else "PUBLIC"),
        ("accelerator", f"{accelerator} (machine shape {shape})" if shape else accelerator),
        ("internet", "on" if flag("enable_internet", True) else "off"),
    ]
    for key, label in SOURCES:
        sources = metadata.get(key) or []
        if isinstance(sources, list) and sources:
            shown = ", ".join(str(source) for source in sources[:5])
            more = f" and {len(sources) - 5} more" if len(sources) > 5 else ""
            details.append((label, shown + more))
    cost = "starts a run on Kaggle"
    if tpu or gpu:
        cost += f"; uses the weekly {accelerator} hours"
    return slug, details, cost


def code_file(folder: Path, metadata: dict) -> Path | int:
    """The code file Kaggle will receive, checked: inside the folder, not a credential."""
    name = str(metadata.get("code_file") or "")
    if not name:
        return script.fail(f"{notebook.METADATA_FILE} names no code_file", script.EXIT_USAGE)
    base = folder.resolve()
    path = (folder / name).resolve()
    if base not in path.parents:
        return script.fail(
            f"the code_file in {notebook.METADATA_FILE} is outside {folder}; refusing to send it",
            script.EXIT_REFUSED,
        )
    if preflight.looks_secret(path.name) or preflight.looks_secret(Path(name).name):
        return script.fail(
            f"the code_file in {notebook.METADATA_FILE} looks like a credential file; "
            "refusing to send it",
            script.EXIT_REFUSED,
        )
    if not path.is_file():
        return script.fail(f"the code_file {name} is not in {folder}", script.EXIT_USAGE)
    return path


def push(folder: Path, receipt: dict | None = None) -> int:
    """Run ``kaggle kernels push``. The caller has confirmed."""
    if not kaggle_cli.installed():
        return script.missing_package("kaggle", "pushing a notebook")
    if credentials.resolve() is None:
        return script.no_credential("pushing a notebook")
    args = ["kernels", "push", "-p", str(folder)]
    result = kaggle_cli.run(args)
    status = kaggle_cli.print_result(result, args=args, tool="kernels.push")
    if status:
        return status
    # Kaggle may canonicalize a new kernel's id from its title. Poll only the
    # handle actually returned by this push, never the requested metadata id.
    match = re.search(
        r"^Kernel version ([1-9][0-9]*) successfully pushed\.\s+"
        r"Please check progress at https://(?:www\.)?kaggle\.com/code/"
        r"([\w-]+/[\w-]+)\s*$", result.stdout, re.MULTILINE,
    )
    if not match:
        return script.fail(
            "Push has no usable Notebook receipt. Inspect the existing attempt; "
            "do not push again to obtain a receipt.", script.EXIT_UNAVAILABLE,
        )
    found = {"notebook": match.group(2), "version": int(match.group(1))}
    metadata = notebook.read_metadata(folder) or {}
    requested = str(metadata.get("id") or "")
    if found['notebook'].split('/')[0] != requested.split('/')[0]:
        return script.fail("Notebook receipt owner differs from the requested owner; "
                           "inspect this attempt before proceeding.", script.EXIT_UNAVAILABLE)
    if receipt is not None:
        receipt.update(found)
    print(f"Confirmed Notebook receipt: {found['notebook']} version {found['version']}")
    script.write_state(script.state_dir() / 'notebook-push.json',
                       json.dumps({**found, 'requested_notebook': requested}))
    return script.EXIT_OK


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Push a notebook to Kaggle, which also runs it. A dry run unless --yes "
        "is given.",
        epilog="Starts a run on Kaggle. Needs the Kaggle CLI and a credential.",
    )
    parser.add_argument("dir", help="Folder with the notebook and its kernel-metadata.json")
    script.add_yes(parser)
    args = parser.parse_args(argv)
    folder = Path(args.dir)

    planned = plan(folder)
    if isinstance(planned, int):
        return planned
    slug, details, cost = planned
    gate = script.write_gate(
        args.yes, action="push a notebook version and run it", details=details, cost=cost
    )
    if gate is not None:
        return gate

    credentials.load_configured_env_file()
    receipt = {}
    status = push(folder, receipt)
    if status != 0:
        return status
    actual = receipt['notebook']
    print(f"Pushed {actual}. It is running on Kaggle; wait for it with: notebook-wait {actual}")
    return script.EXIT_OK


if __name__ == "__main__":
    sys.exit(main())
