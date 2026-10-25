#!/bin/bash
# Complete venv-llm after torch 2.10 --no-deps is installed.
set -e
cd /mnt/disk3/liuyutong_home/workspace/skill/score-push/runs/14-gemma-4-developer-agent
QWEN=/mnt/disk3/liuyutong_home/miniconda3/envs/qwen/lib/python3.11/site-packages
DEST=venv-llm/lib/python3.12/site-packages
WH=models/wheelhouse

echo "[1/4] reusing qwen nvidia + cutlass libs"
for d in "$QWEN"/nvidia_* "$QWEN"/nvidia; do
  [ -e "$d" ] || continue
  n=$(basename "$d")
  if [ ! -e "$DEST/$n" ]; then cp -r "$d" "$DEST/"; fi
done

echo "[2/4] wheelhouse wheels (--no-deps)"
venv-llm/bin/pip install -q --no-deps --force-reinstall $WH/*.whl 2>&1 | grep -vE "WARNING" | tail -2

echo "[3/4] remaining core deps from PyPI"
venv-llm/bin/pip install -q \
  regex cachetools psutil sentencepiece numpy requests tqdm blake3 py-cpuinfo \
  protobuf "fastapi[standard]" aiohttp openai pydantic prometheus_client pillow \
  tiktoken lark typing_extensions filelock pyzmq einops cloudpickle watchfiles \
  python-json-logger ninja mcp numba opencv-python-headless six setuptools \
  opentelemetry-sdk opentelemetry-api opentelemetry-exporter-otlp \
  2>&1 | grep -vE "WARNING" | tail -2

echo "[4/4] smoke test"
venv-llm/bin/python - <<'PY'
import torch
print("torch", torch.__version__)
import vllm
print("vllm", vllm.__version__)
from vllm.model_executor.layers.quantization import get_quantization_config
print("quant registry ok")
import transformers
print("transformers", transformers.__version__)
PY
