#!/bin/bash
export HF_HOME=/mnt/disk3/liuyutong_home/workspace/skill/score-push/runs/06-name-that-llm/hf_cache
export HF_HUB_DISABLE_XET=1
export HF_ENDPOINT=https://hf-mirror.com
export HTTP_PROXY=http://172.28.4.16:18999 HTTPS_PROXY=http://172.28.4.16:18999 http_proxy=$HTTP_PROXY https_proxy=$HTTPS_PROXY
export NO_PROXY=localhost,127.0.0.1,::1,172.16.0.0/12,10.0.0.0/8,192.168.0.0/16,100.64.0.0/10
/mnt/disk3/liuyutong_home/workspace/skill/score-push/runs/06-name-that-llm/venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download
import time
t=time.time()
p = snapshot_download("Qwen/Qwen3-4B", max_workers=1, resume_download=True)
print("DONE", p, round(time.time()-t,1))
PY
