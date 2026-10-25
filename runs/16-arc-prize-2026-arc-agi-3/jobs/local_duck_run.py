"""Run the TAAF duck harness locally (offline) against the public games.

Launches a local vLLM server for the Qwen3.6-27B-FP8 model, then runs the
pickled TAAF benchmark against the local public environment files.

Env knobs (defaults = smoke test):
  DUCK_GAMES      comma list of game prefixes (default: all)
  DUCK_PASSES     n_passes (default 1)
  DUCK_BUDGET_S   max_runtime_s_per_game (default 240)
  DUCK_TOTAL_S    overall soft end (default 1800)
  DUCK_CONCURRENCY (default 4)
  DUCK_PORT       vllm port (default 1234)
"""
import json
import os
import pickle
import shutil
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

RUN = Path(__file__).resolve().parent.parent
BUNDLE = Path(os.environ.get("DUCK_BUNDLE", str(RUN / "assets" / "taaf-share")))
SRC = BUNDLE / "src"
sys.path.insert(0, str(SRC / "tufa-arc-agi-framework" / "src"))
sys.path.insert(0, str(SRC / "ARC3-Inference"))

MODEL_DIR = RUN / "assets" / "qwen36-27b-fp8"
SERVED = "vrfai/Qwen3.6-27B-FP8"
PORT = int(os.environ.get("DUCK_PORT", "1234"))
BASE_URL = f"http://127.0.0.1:{PORT}/v1"
WORK = RUN / "jobs" / "duck_work"
WORK.mkdir(parents=True, exist_ok=True)

analyzer_env = {
    "LOCAL_ANALYZER_BASE_URL": BASE_URL,
    "OPENAI_BASE_URL": BASE_URL,
    "LOCAL_ANALYZER_PROVIDER": "vllm",
    "OPENAI_PROVIDER": "vllm",
    "LOCAL_ANALYZER_MODEL_ID": SERVED,
    "INFERENCE_ANALYZER_MODEL": SERVED,
    "LOCAL_ANALYZER_APP_NAME": "ARC3 Agent Harness",
    "LOCAL_ANALYZER_CONTEXT_WINDOW": "32768",
    "LOCAL_ANALYZER_MAX_OUTPUT": "0",
    "LOCAL_ANALYZER_TOOL_STEPS": "0",
    "LOCAL_ANALYZER_TOOL_TIMEOUT": "30",
    "LOCAL_ANALYZER_TOOL_OUTPUT_TOKENS": "1024",
    "LOCAL_ANALYZER_YIELD_SECONDS": "60",
    "LOCAL_ANALYZER_TEMPERATURE": "0.6",
    "LOCAL_ANALYZER_TOP_P": "0.95",
    "LOCAL_ANALYZER_TOP_K": "20",
    "LOCAL_ANALYZER_ENABLE_THINKING": "true",
    "MULTIMODAL_CONTEXT": "current_grid",
    "MULTIMODAL_UPSCALE": "4",
    "ONLY_RESET_LEVELS": "true",
    "TAAF_RUN_AS_SUBMISSION": "0",
    "TAAF_MINIMAL_DIAGNOSTICS": "0",
    "MPLBACKEND": "Agg",
}
os.environ.update(analyzer_env)

# keep all caches inside the run dir (offline worker allows writes only here)
OFF_CACHE = RUN / ".cache-offline"
OFF_CACHE.mkdir(exist_ok=True)
for var in ("HF_HOME", "TRANSFORMERS_CACHE", "HF_MODULES_CACHE", "VLLM_CACHE_ROOT", "XDG_CACHE_HOME"):
    os.environ.setdefault(var, str(OFF_CACHE / var.lower()))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "0")


def start_vllm() -> None:
    log_path = WORK / "vllm.log"
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", str(MODEL_DIR),
        "--served-model-name", SERVED,
        "--host", "127.0.0.1",
        "--port", str(PORT),
        "--tensor-parallel-size", "1",
        "--enable-auto-tool-choice",
        "--tool-call-parser", "qwen3_coder",
        "--generation-config", "vllm",
        "--enable-prefix-caching",
        "--default-chat-template-kwargs", '{"preserve_thinking": true}',
        "--reasoning-parser", "qwen3",
        "--max-model-len", "65536",
    ]
    with log_path.open("w") as log:
        proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STOUT, env=os.environ.copy())
    (WORK / "vllm.pid").write_text(str(proc.pid))
    deadline = time.time() + 1800
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError(f"vllm server died rc={proc.poll()}")
        try:
            import urllib.request
            with urllib.request.urlopen(f"{BASE_URL}/models", timeout=5) as r:
                if r.status < 500:
                    print("vllm ready:", r.read()[:200], flush=True)
                    return
        except Exception:
            time.sleep(5)
    raise TimeoutError("vllm not ready")


def main() -> None:
    if not (MODEL_DIR / "model.safetensors").exists():
        raise SystemExit("model weights missing: run the model download first")
    import arc_agi
    import taaf.game_api

    start_vllm()
    try:
        target = pickle.load(open(BUNDLE / "deploy_target.pkl", "rb"))
        bm = pickle.load(open(BUNDLE / "benchmark_initial.pkl", "rb"))
        target.actual_run_as_submission = False
        target.is_competition_rerun = False

        env_dir = str(RUN / "environment_files")
        spec = taaf.game_api.ArcadeSpec(
            operation_mode=arc_agi.OperationMode.OFFLINE, environments_dir=env_dir
        )
        arcade = arc_agi.Arcade(operation_mode=arc_agi.OperationMode.OFFLINE, environments_dir=env_dir)
        all_ids = [e.game_id for e in arcade.available_environments]
        wanted = os.environ.get("DUCK_GAMES", "")
        if wanted:
            prefixes = {w.split("-")[0] for w in wanted.split(",")}
            ids = [g for g in all_ids if g.split("-")[0] in prefixes]
        else:
            ids = all_ids
        bm.games = [taaf.game_api.GameAPI(env_name=gid, arcade_spec=spec) for gid in ids]

        bm.n_passes = int(os.environ.get("DUCK_PASSES", "1"))
        bm.job_dir = RUN / "results" / f"duck_local_{int(time.time())}"
        bm.solver.max_runtime_s_per_game = float(os.environ.get("DUCK_BUDGET_S", "240"))
        bm.solver.concurrency = int(os.environ.get("DUCK_CONCURRENCY", "4"))
        bm.solver.save_request_logs = True
        print(f"games: {len(bm.games)} passes={bm.n_passes} budget={bm.solver.max_runtime_s_per_game}s "
              f"concurrency={bm.solver.concurrency}", flush=True)

        import asyncio
        t0 = time.time()
        soft_end = datetime.fromtimestamp(t0) + timedelta(seconds=float(os.environ.get("DUCK_TOTAL_S", "1800")))
        asyncio.run(bm.run(soft_end_time=soft_end, runtime_environment=target, minimal_diagnostics=False))
        print(f"RUN DONE in {time.time()-t0:.0f}s -> {bm.job_dir}", flush=True)
    finally:
        pid_file = WORK / "vllm.pid"
        if pid_file.exists():
            try:
                os.kill(int(pid_file.read_text().strip()), 15)
            except Exception:
                pass


if __name__ == "__main__":
    main()
