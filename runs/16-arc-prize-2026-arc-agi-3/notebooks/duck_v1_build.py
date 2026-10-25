"""Build the private Kaggle submission notebook (duck harness, my own wrapper)."""
import json

DATASET_VERSION_IDS = {
    "jeroencottaar/taaf-kaggle-source-share": {"sourceId": 16761237, "databundleId": 10710965},
    "driessmit1/arc3-vllm-h100-wheelhouse-v3": {"sourceId": 20336997, "databundleId": 12377681},
    "driessmit1/vrfai-qwen3-6-27b-fp8-hf-snapshot": {"sourceId": 20338030, "databundleId": 12378286},
}

DATASET_SOURCES = [
    "jeroencottaar/taaf-kaggle-source-share",
    "driessmit1/arc3-vllm-h100-wheelhouse-v3",
    "driessmit1/vrfai-qwen3-6-27b-fp8-hf-snapshot",
]

cell_env = r'''import json
import os
import pickle
import subprocess
import sys
import sysconfig
import time
from datetime import datetime, timedelta
from pathlib import Path
from urllib.request import urlopen

TRUE_SUBMISSION = os.environ.get("KAGGLE_IS_COMPETITION_RERUN", "").strip().lower() in {"1", "true"}
NOTEBOOK_START_EPOCH = time.time()
os.environ["MPLBACKEND"] = "Agg"
os.environ["TAAF_RUN_AS_SUBMISSION"] = "1" if TRUE_SUBMISSION else "0"
os.environ["TAAF_MINIMAL_DIAGNOSTICS"] = "1" if TRUE_SUBMISSION else "0"
os.environ["ONLY_RESET_LEVELS"] = "true"

cuda_library_path = "/usr/local/nvidia/lib64"
os.environ["LIBRARY_PATH"] = os.pathsep.join(
    e for e in [cuda_library_path, *os.environ.get("LIBRARY_PATH", "").split(os.pathsep)] if e
)

WORKING_DIR = Path("/kaggle/working")
WORKING_DIR.mkdir(parents=True, exist_ok=True)
print(f"duck-v1: TRUE_SUBMISSION={TRUE_SUBMISSION}")
'''

cell_install = r'''subprocess.check_call(
    [
        sys.executable, "-m", "pip", "install",
        "--quiet", "--no-index", "--no-warn-conflicts", "--disable-pip-version-check",
        "--find-links",
        "/kaggle/input/competitions/arc-prize-2026-arc-agi-3/arc_agi_3_wheels",
        "arc-agi",
    ],
    stdout=subprocess.DEVNULL,
)
print("arc runtime installed")
'''

cell_bundle = r'''DATASET_SOURCES = __DATASET_SOURCES__
KERNEL_SOURCES = []
DATASET_BUNDLE_MARKER = "taaf-kaggle-bundle.json"
SETUP_ENV_PATH = WORKING_DIR / "taaf_setup_env.json"


def _find_bundle_dir() -> Path:
    for marker in Path("/kaggle/input").rglob(DATASET_BUNDLE_MARKER):
        return marker.parent
    raise RuntimeError("TAAF source bundle not found under /kaggle/input.")


def _dataset_mount_candidates(ref: str) -> list:
    owner, slug = ref.split("/", 1)
    return [Path("/kaggle/input") / slug, Path("/kaggle/input/datasets") / owner / slug]


def _first_existing(candidates) -> Path | None:
    return next((c for c in candidates if c.exists()), None)


BUNDLE_DIR = _find_bundle_dir()
print(f"duck-v1: source bundle = {BUNDLE_DIR}")

kaggle_input_paths = {}
for i, ref in enumerate(DATASET_SOURCES):
    candidates = _dataset_mount_candidates(ref)
    resolved = BUNDLE_DIR if i == 0 else _first_existing(candidates)
    kaggle_input_paths[ref] = str(resolved or candidates[0])
setup_env = {
    "TAAF_KAGGLE_INPUT_PATHS": json.dumps(kaggle_input_paths, sort_keys=True),
    "TAAF_KAGGLE_DATASET_SOURCES": json.dumps(DATASET_SOURCES),
    "TAAF_KAGGLE_KERNEL_SOURCES": json.dumps(KERNEL_SOURCES),
}
os.environ.update(setup_env)
SETUP_ENV_PATH.write_text(json.dumps(setup_env, indent=2, sort_keys=True) + "\n")
print(f"duck-v1: input paths = {setup_env['TAAF_KAGGLE_INPUT_PATHS']}")
'''
cell_bundle = cell_bundle.replace("__DATASET_SOURCES__", json.dumps(DATASET_SOURCES))

cell_setup = r'''def _source_path_entries(bundle_dir: Path) -> list:
    entries = []
    for repo in sorted((bundle_dir / "src").iterdir(), reverse=True):
        for candidate in (repo / "src", repo):
            if candidate.is_dir():
                entries.append(candidate)
    return entries


def _command_env() -> dict:
    env = os.environ.copy()
    env["PYTHON"] = sys.executable
    env["TAAF_KAGGLE_BUNDLE_DIR"] = str(BUNDLE_DIR)
    env["TAAF_KAGGLE_WORKING_DIR"] = str(WORKING_DIR)
    env["TAAF_KAGGLE_SETUP_ENV"] = str(SETUP_ENV_PATH)
    env.update({str(k): str(v) for k, v in json.loads(SETUP_ENV_PATH.read_text()).items()})
    return env


source_entries = _source_path_entries(BUNDLE_DIR)
for entry in source_entries:
    sys.path.insert(0, str(entry))
pth_path = Path(sysconfig.get_paths()["purelib"]) / "duck_v1_sources.pth"
pth_path.write_text("".join(f"{entry}\n" for entry in source_entries))
print(f"duck-v1: wrote {pth_path} ({len(source_entries)} source roots)")

env = _command_env()
for command in json.loads((BUNDLE_DIR / "setup_commands.json").read_text()):
    print(f"duck-v1: setup command:", flush=True)
    subprocess.run(command, shell=True, check=True, cwd=WORKING_DIR, env=env)
    env = _command_env()
    os.environ.update(env)

for entry in reversed([e for e in os.environ.get("PYTHONPATH", "").split(os.pathsep) if e]):
    if entry not in sys.path:
        sys.path.insert(0, entry)
'''

cell_benchmark = r'''with open(BUNDLE_DIR / "deploy_target.pkl", "rb") as file:
    target = pickle.load(file)
target.actual_run_as_submission = TRUE_SUBMISSION
target.is_competition_rerun = TRUE_SUBMISSION

with open(BUNDLE_DIR / "benchmark_initial.pkl", "rb") as file:
    bm = pickle.load(file)
bm.job_dir = WORKING_DIR

# --- customization hook (this run's configuration) ---
bm.n_passes = 1
bm.game_weights = None
bm.solver.save_request_logs = True
print("duck-v1:", bm.label, "games:", len(bm.games), "budget:", bm.solver.max_runtime_s_per_game,
      "concurrency:", bm.solver.concurrency, flush=True)
'''

cell_run = r'''def _competition_games():
    import arc_agi
    import taaf.game_api

    spec = taaf.game_api.ArcadeSpec(
        operation_mode=arc_agi.OperationMode.COMPETITION,
        arc_base_url=os.environ["ARC_BASE_URL"],
        environments_dir="",
    )
    arcade = arc_agi.Arcade(
        operation_mode=arc_agi.OperationMode.COMPETITION,
        arc_base_url=spec.arc_base_url,
        environments_dir="",
    )
    game_ids = [env_info.game_id for env_info in arcade.available_environments]
    if not game_ids:
        raise RuntimeError("Competition Arcade exposed zero environments.")
    return [taaf.game_api.GameAPI(env_name=game_id, arcade_spec=spec) for game_id in game_ids]


def _offline_games(env_dir: str):
    import arc_agi
    import taaf.game_api

    spec = taaf.game_api.ArcadeSpec(operation_mode=arc_agi.OperationMode.OFFLINE, environments_dir=env_dir)
    arcade = arc_agi.Arcade(operation_mode=arc_agi.OperationMode.OFFLINE, environments_dir=env_dir)
    game_ids = [env_info.game_id for env_info in arcade.available_environments]
    if not game_ids:
        raise RuntimeError(f"No offline environments found under {env_dir}.")
    return [taaf.game_api.GameAPI(env_name=game_id, arcade_spec=spec) for game_id in game_ids]


def _wait_for_gateway(base_url: str, timeout_s: float = 600.0) -> None:
    deadline = time.monotonic() + timeout_s
    last_error = ""
    while time.monotonic() < deadline:
        try:
            with urlopen(f"{base_url}api/games", timeout=10) as response:
                if response.status < 500:
                    return
        except Exception as exc:
            last_error = repr(exc)
        time.sleep(5)
    raise RuntimeError(f"Kaggle gateway did not become ready: {last_error}")


print((BUNDLE_DIR / "preamble.txt").read_text())
(WORKING_DIR / "git_status.txt").write_text((BUNDLE_DIR / "git_status.txt").read_text())
os.environ.setdefault("RECORDINGS_DIR", str(WORKING_DIR / "server_recording"))

if TRUE_SUBMISSION:
    os.environ.setdefault("ARC_API_KEY", "test-key-123")
    os.environ.setdefault("ARC_BASE_URL", "http://gateway:8001/")
    _wait_for_gateway(os.environ["ARC_BASE_URL"])
    bm.games = _competition_games()
else:
    competition_env_files = str(Path("/kaggle/input/competitions/arc-prize-2026-arc-agi-3/arc_agi_3_wheels").parent / "environment_files")
    bm.games = _offline_games(competition_env_files)

soft_end = None
if not TRUE_SUBMISSION:
    budget = float(getattr(target, "max_runtime_s", 0.0) or 0.0)
    if budget > 0:
        soft_end = datetime.fromtimestamp(NOTEBOOK_START_EPOCH) + timedelta(seconds=budget - min(600.0, budget / 2))

try:
    await bm.run(soft_end_time=soft_end, runtime_environment=target, minimal_diagnostics=TRUE_SUBMISSION)
    if not TRUE_SUBMISSION:
        import pandas as pd
        pd.DataFrame([["1_0", "1", True, 1]], columns=["row_id", "game_id", "end_of_game", "score"]).to_parquet(
            WORKING_DIR / "submission.parquet", index=False
        )
finally:
    for command in json.loads((BUNDLE_DIR / "teardown_commands.json").read_text()):
        print(f"duck-v1: teardown command: {command}", flush=True)
        subprocess.run(command, shell=True, check=False, cwd=WORKING_DIR, env=_command_env())
'''

cells = [
    {"cell_type": "markdown", "metadata": {}, "source": [
        "# ARC-AGI-3 submission — TAAF duck harness (run 16, candidate v1)\n",
        "\n",
        "Runs the open TAAF duck-harness solver (Tufa Labs, MIT) with a local vLLM-served "
        "Qwen3.6-27B-FP8 analyzer against the live competition arcade. All inputs are public "
        "Kaggle datasets; internet is disabled."
    ]},
    {"cell_type": "code", "execution_count": None, "metadata": {}, "source": cell_env.splitlines(keepends=True)},
    {"cell_type": "code", "execution_count": None, "metadata": {}, "source": cell_install.splitlines(keepends=True)},
    {"cell_type": "code", "execution_count": None, "metadata": {}, "source": cell_bundle.splitlines(keepends=True)},
    {"cell_type": "code", "execution_count": None, "metadata": {}, "source": cell_setup.splitlines(keepends=True)},
    {"cell_type": "code", "execution_count": None, "metadata": {}, "source": cell_benchmark.splitlines(keepends=True)},
    {"cell_type": "code", "execution_count": None, "metadata": {}, "source": cell_run.splitlines(keepends=True)},
]

nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
        "kaggle": {
            "accelerator": "nvidiaRtxPro6000",
            "dataSources": [
                {"sourceType": "competition", "sourceId": 133468},
                *[{"sourceType": "datasetVersion", "sourceId": DATASET_VERSION_IDS[ref]["sourceId"],
                   "databundleId": DATASET_VERSION_IDS[ref]["databundleId"]} for ref in DATASET_SOURCES],
            ],
            "isInternetEnabled": False,
            "language": "python",
            "sourceType": "notebook",
            "isGpuEnabled": False,
        },
    },
    "nbformat": 4,
    "nbformat_minor": 4,
}
out = "notebooks/duck_v1.ipynb"
json.dump(nb, open(out, "w"), indent=1)
print("wrote", out)
