import argparse
import json
from pathlib import Path
import shutil

from .common import now, sha256, write_json
from .replay import rebuild_eeg, rebuild_housing

ROOT = Path(__file__).resolve().parents[1]
RUNS = {
    "housing": "01-predicting-housing-affordability-in-the-u-s",
    "eeg": "04-low-cost-motor-imagery-decoding-for-rehab-cross-subject",
}


def verify(root=ROOT):
    manifest = json.loads((root / "reproduce/source-manifest.json").read_text())
    failures = []
    for row in manifest["source_files"]:
        path = root / row["path"]
        if not path.is_file() or path.is_symlink() or sha256(path) != row["sha256"]:
            failures.append(row["path"])
    return {"checked_at": now(), "passed": not failures,
            "checked": len(manifest["source_files"]), "failures": failures}


def restore(source, task, root=ROOT, purpose="replay"):
    """Validate every input before copying; never replace a mismatched file."""
    source = Path(source).resolve()
    items = [r for r in json.loads((root / "reproduce/required-inputs.json").read_text())["inputs"]
             if r["task"] == task and (purpose == "training" or r["usage"] != "base_model_training_only")]
    pending = []
    for row in items:
        relative = Path(row["path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Unsafe relative path in input manifest")
        src, dst = source / relative, root / relative
        if not src.is_file() or sha256(src) != row["sha256"]:
            raise ValueError(f"Missing or mismatched source input: {relative}")
        if dst.exists() and (dst.is_symlink() or sha256(dst) != row["sha256"]):
            raise ValueError(f"Refusing to replace an existing different input: {relative}")
        if not dst.resolve().is_relative_to(root.resolve()):
            raise ValueError(f"Destination escapes the snapshot: {relative}")
        pending.append((src, dst))
    for src, dst in pending:
        if not dst.exists():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(src, dst)
        if sha256(dst) != sha256(src):
            raise ValueError(f"Copy failed verification: {dst}")
    return {"passed": True, "restored_inputs": len(items), "downloaded": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("verify", help="Verify original source and retained predictions")
    command = commands.add_parser("restore", help="Restore hash-checked raw inputs from local storage")
    command.add_argument("--source-root", type=Path, required=True)
    command.add_argument("--task", choices=["housing"], required=True)
    command.add_argument("--purpose", choices=["replay", "training"], default="replay")
    command = commands.add_parser("rebuild", help="Reconstruct saved best submissions without training")
    command.add_argument("task", choices=RUNS)
    command.add_argument("--output", type=Path)
    command.add_argument("--report", type=Path)
    command = commands.add_parser("train-housing", help="Explicitly retrain the original 90 CPU base-model fits")
    command.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "verify":
        result = verify()
    elif args.command == "restore":
        result = restore(args.source_root, args.task, purpose=args.purpose)
    elif args.command == "train-housing":
        from .train_housing import train
        result = train(ROOT, args.output_root)
    else:
        task = args.task
        output = args.output or ROOT / "artifacts" / f"{task}-submission.csv"
        report = args.report or ROOT / "artifacts" / f"{task}-reconstruction.json"
        run = ROOT / "runs" / RUNS[task]
        if output.resolve().is_relative_to((ROOT / "runs").resolve()):
            raise ValueError("Write rebuilt outputs outside preserved runs/")
        # Verify all retained reference artifacts and any restored raw inputs first.
        if not verify()["passed"]:
            raise ValueError("Preserved source or predictions failed verification")
        if task == "housing":
            rows = json.loads((ROOT / "reproduce/required-inputs.json").read_text())["inputs"]
            for row in rows:
                if row["usage"] == "submission_reconstruction_and_training":
                    path = ROOT / row["path"]
                    if not path.is_file() or sha256(path) != row["sha256"]:
                        raise ValueError(f"Restore the exact official input first: {row['path']}")
        result = (rebuild_eeg if task == "eeg" else rebuild_housing)(run, output)
        write_json(report, result)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result.get("passed", True) else 1
