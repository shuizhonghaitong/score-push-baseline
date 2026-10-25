#!/usr/bin/env python3
"""Package an agent candidate directory into submission.zip and validate it."""
import hashlib
import re
import sys
import zipfile
from pathlib import Path

import yaml

yaml.SafeLoader.add_constructor('!include', lambda loader, node: loader.construct_scalar(node))

ALLOWED_EXTENSIONS = {".yaml", ".yml", ".md", ".txt", ".py", ".json", ".safetensors"}
ROOT_NAMES = {"agent.yaml", "agent.yml", "root_agent.yaml", "root_agent.yml"}
ALLOWED_MODELS = {
    "gemma-4-31b-it-qat-w4a16-ct",
    "gemma-4-12b-it-qat-w4a16-ct",
    "gemma-4-e4b-it-qat-w4a16-ct",
    "gemma-4-e2b-it-qat-w4a16-ct",
}


def validate(bundle: Path) -> None:
    files = [p for p in bundle.rglob("*") if p.is_file()]
    roots = [n for n in ROOT_NAMES if (bundle / n).is_file()]
    assert len(roots) == 1, f"need exactly one root config, found {roots}"
    assert not any(p.is_symlink() for p in bundle.rglob("*")), "symlinks rejected"
    assert all(p.suffix.lower() in ALLOWED_EXTENSIONS for p in files), [
        str(p) for p in files if p.suffix.lower() not in ALLOWED_EXTENSIONS
    ]
    total = sum(p.stat().st_size for p in files)
    assert total <= 3 * 2**30, f"unpacked size {total} over 3 GiB"
    models = set()
    for cfg in [p for p in files if p.suffix in (".yaml", ".yml")]:
        text = cfg.read_text()
        for target in re.findall(r"!include\s+(\S+)", text):
            assert (cfg.parent / target).is_file(), f"{cfg.name}: missing include {target}"
        try:
            data = yaml.safe_load(text)
        except yaml.YAMLError as e:
            raise SystemExit(f"{cfg.name}: YAML parse error: {e}")
        if isinstance(data, dict):
            if "adapter" in data:
                for part in ("adapter_config.json", "adapter_model.safetensors"):
                    assert (bundle / "adapters" / str(data["adapter"]) / part).is_file(), (
                        f"{cfg.name}: adapter {data['adapter']} incomplete"
                    )
            if "model" in data:
                models.add(str(data["model"]))
    assert len(models) == 1, f"one base model per submission, found {models}"
    assert models <= ALLOWED_MODELS, f"model {models} not allowed"
    print(f"validate OK: root={roots[0]} files={len(files)} size={total}B model={models.pop()}")


def main() -> None:
    bundle = Path(sys.argv[1]).resolve()
    out = Path(sys.argv[2]).resolve()
    validate(bundle)
    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(bundle.rglob("*")):
            if p.is_file():
                zf.write(p, p.relative_to(bundle).as_posix())
    sha = hashlib.sha256(out.read_bytes()).hexdigest()
    print(f"zip: {out} ({out.stat().st_size} bytes)")
    print(f"sha256: {sha}")


if __name__ == "__main__":
    main()
