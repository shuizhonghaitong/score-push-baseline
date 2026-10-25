#!/usr/bin/env python3
"""GPU job entry: start vLLM server child (3.12 venv), then run local eval (3.13)."""
import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

RUN = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RUN / 'scripts'))


def wait_ready(port: int, timeout: float = 60 * 40) -> bool:
    import urllib.request
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            with urllib.request.urlopen(f'http://127.0.0.1:{port}/v1/models', timeout=3) as r:
                if r.status == 200:
                    return True
        except Exception:
            time.sleep(5)
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='gemma-4-31b-it-qat-w4a16-ct')
    ap.add_argument('--port', type=int, default=8000)
    ap.add_argument('--no-server', action='store_true', help='server already running')
    ap.add_argument('--tasks', default='all')
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--num-shards', type=int, default=1)
    ap.add_argument('--results', default=None)
    ap.add_argument('--agent-dir', default=str(RUN / 'agent'))
    ap.add_argument('--gold', action='store_true')
    ap.add_argument('--gpu-mem', type=float, default=0.85)
    ap.add_argument('--max-len', type=int, default=32768)
    args = ap.parse_args()

    server_proc = None
    if not args.no_server:
        logdir = Path(args.results) if args.results else RUN / 'results'
        logdir.mkdir(parents=True, exist_ok=True)
        logf = open(logdir / 'vllm_server.log', 'a')
        server_proc = subprocess.Popen(
            [str(RUN / 'venv-llm' / 'bin' / 'python'), str(RUN / 'scripts' / 'serve_model.py'),
             '--model', args.model, '--port', str(args.port), '--gpu-mem', str(args.gpu_mem),
             '--max-len', str(args.max_len)],
            stdout=logf, stderr=subprocess.STDOUT, cwd=RUN)
        print('server proc', server_proc.pid, flush=True)
        if not wait_ready(args.port):
            print('SERVER FAILED TO START', flush=True)
            server_proc.terminate()
            sys.exit(1)
        print('server ready', flush=True)

    env = dict(os.environ)
    env['MODEL_URL'] = f'http://127.0.0.1:{args.port}/v1'
    env['MODEL_NAME'] = args.model
    try:
        if args.gold:
            import gold_check as GC
            sys.argv = ['gold_check.py', '--shard', str(args.shard),
                        '--num-shards', str(args.num_shards)]
            if args.results:
                sys.argv += ['--results', args.results]
            GC.main()
        else:
            import eval_local as EL
            sys.argv = ['eval_local.py', '--tasks', args.tasks, '--shard', str(args.shard),
                        '--num-shards', str(args.num_shards), '--agent-dir', args.agent_dir]
            if args.results:
                sys.argv += ['--results', args.results]
            EL.main()
    finally:
        if server_proc is not None:
            server_proc.terminate()
            try:
                server_proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                server_proc.kill()


if __name__ == '__main__':
    main()
