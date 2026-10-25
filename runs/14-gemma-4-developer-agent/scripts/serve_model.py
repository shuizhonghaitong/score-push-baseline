#!/usr/bin/env python3
"""Serve a Gemma 4 competition model with vLLM (OpenAI-compatible API)."""
import argparse
import os
import sys
import time
from pathlib import Path

RUN = Path(__file__).resolve().parents[1]
MODEL = {
    'gemma-4-31b-it-qat-w4a16-ct': RUN / 'models' / 'gemma-4-31b',
    'gemma-4-12b-it-qat-w4a16-ct': RUN / 'models' / 'gemma-4-12b',
    'gemma-4-e4b-it-qat-w4a16-ct': RUN / 'models' / 'gemma-4-e4b',
    'gemma-4-e2b-it-qat-w4a16-ct': RUN / 'models' / 'gemma-4-e2b',
}

os.environ.setdefault('VLLM_NO_USAGE_STATS', '1')
os.environ.setdefault('VLLM_WORKER_MULTIPROC_METHOD', 'spawn')
os.environ.setdefault('VLLM_ENGINE_READY_TIMEOUT_S', '1800')
os.environ.setdefault('VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS', '1')
os.environ.setdefault('TRANSFORMERS_OFFLINE', '1')

from adk_submission import VllmConfig, VllmServer  # noqa: E402

MODEL_EXTRA_ARGS = {
    'gemma-4-12b-it-qat-w4a16-ct': (
        '--hf-overrides',
        '{"architectures": ["Gemma4ForCausalLM"], "model_type": "gemma4"}',
    ),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='gemma-4-31b-it-qat-w4a16-ct')
    ap.add_argument('--port', type=int, default=8000)
    ap.add_argument('--tp', type=int, default=1)
    ap.add_argument('--gpu-mem', type=float, default=0.85)
    ap.add_argument('--max-len', type=int, default=32768)
    ap.add_argument('--keep-already', action='store_true', help='exit 0 if the port already serves')
    args = ap.parse_args()

    import requests
    try:
        r = requests.get(f'http://127.0.0.1:{args.port}/v1/models', timeout=3)
        if r.ok:
            print('already serving:', r.json().get('data', [{}])[0].get('id'))
            if args.keep_already:
                return 0
            sys.exit('port busy')
    except Exception:
        pass

    model_path = MODEL[args.model]
    assert model_path.exists(), f'model missing: {model_path}'
    cfg = VllmConfig(
        model=str(model_path), port=args.port, host='127.0.0.1',
        tool_call_parser='gemma4', reasoning_parser='gemma4',
        max_model_len=args.max_len, dtype='bfloat16',
        gpu_memory_utilization=args.gpu_mem, enable_auto_tool_choice=True,
        enable_lora=False, tensor_parallel_size=args.tp, startup_timeout=60 * 30,
        extra_args=MODEL_EXTRA_ARGS.get(args.model, ()),
    )
    server = VllmServer(cfg, adapter_manifest=None)
    server.start()
    print('vLLM server started on', server.base_url, flush=True)
    while True:
        time.sleep(15)
        print('alive', flush=True)


if __name__ == '__main__':
    main()
