#!/bin/bash
# Install vllm==0.19.0 + torch + flashinfer offline from the Kaggle wheelhouse into vllm-venv.
set -e
cd "$(dirname "$0")/.."
export UV_CACHE_DIR=$PWD/.uvcache UV_PYTHON_INSTALL_DIR=$PWD/.uvpy
WH=assets/wheelhouse-v3
ZIP="$WH/arc3-vllm-h100-wheelhouse-v3.zip"
[ -f "$ZIP" ] || { echo "wheelhouse zip missing: $ZIP"; exit 1; }
[ -f "$WH/.extracted" ] || { echo "extracting wheelhouse..."; unzip -q -o "$ZIP" -d "$WH"; touch "$WH/.extracted"; }
echo "installing from wheelhouse (offline) into vllm-venv ..."
uv pip install --python vllm-venv/bin/python \
  --no-index --find-links "$WH" \
  --requirement "$WH/requirements.lock" \
  --disable-pip-version-check --no-warn-conflicts
echo "verifying import ..."
vllm-venv/bin/python -c "import vllm, torch; print('vllm', vllm.__version__, 'torch', torch.__version__, 'cuda', torch.cuda.is_available())"
echo "DONE"
