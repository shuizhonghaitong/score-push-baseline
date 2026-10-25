#!/usr/bin/env python3
"""Local scorer-like evaluation of a submission agent on the public tasks.

Usage:
  venv-eval/bin/python scripts/eval_local.py --tasks 0,1,2 --shard 0 --num-shards 1
Env:
  MODEL_URL  vLLM OpenAI base url (default http://127.0.0.1:8000/v1)
  MODEL_NAME (default gemma-4-31b-it-qat-w4a16-ct)
  AGENT_DIR  (default agent)
Runs the agent phase (Phase 1) + verification (Phase 2) inside subprocess
sandboxes whose venv/python and installed wheels mirror the scorer image.
"""
import argparse
import asyncio
import json
import os
import re
import shutil
import subprocess
import sys
import tarfile
import time
import tomllib
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

RUN = Path(__file__).resolve().parents[1]
COMP = RUN / "data"
WHEELHOUSE = RUN / "models" / "wheelhouse"
EXTRA = RUN / "models" / "local-eval" / "wheels"
PYDS = RUN / "models" / "local-eval" / "pythons"

import importlib
import logging
import re
import tomllib
import zipfile
from pathlib import Path

from packaging.requirements import Requirement
from packaging.version import Version
from swegemma.config import EvalConfig, build_submission_limits
from swegemma.evaluate import Evaluator
from swegemma.models import load_tasks
from swegemma.sandbox.subprocess import SubprocessManager
import swegemma.harness.verification as verification
import swegemma.harness.agent_runner as agent_runner
from adk_submission import ModelRegistry
from google.adk.models.lite_llm import LiteLlm
from google.adk.agents.context_cache_config import ContextCacheConfig
from google.adk.apps._configs import EventsCompactionConfig

logging.getLogger().setLevel(logging.ERROR)

norm = lambda n: re.sub(r'[-_.]+', '_', n).lower()


def fits(w: str, tag: str) -> bool:
    m = re.fullmatch(r'(.+)-([^-]+)-([^-]+)-(.+)\.whl', Path(w).name)
    if not m:
        return True
    py, abi, plat = m.group(2), m.group(3), m.group(4)
    if plat not in ('any',) and 'linux' not in plat:
        return False
    if abi == 'abi3':
        return int(re.sub(r'\D', '', py) or 0) <= int(tag[2:])
    return any(t.startswith('py3') for t in py.split('.')) or tag in py.split('.')


WHEELS = {}
for w in [*(COMP / 'wheels').glob('*.whl'), *EXTRA.glob('*.whl'), *PYDS.glob('*.whl')]:
    name, ver = w.name.split('-')[:2]
    WHEELS.setdefault(norm(name), []).append((Version(ver), w))

OWN = {'fastapi/fastapi': 'fastapi', 'Textualize/rich': 'rich',
       'psf/requests': 'requests', 'encode/httpx': 'httpx'}
ONLY_FOR = {'pytest_httpbin': {'psf/requests'}, 'httpbin': {'psf/requests'},
            'flasgger': {'psf/requests'}, 'brotlicffi': {'psf/requests', 'encode/httpx'},
            'brotli': {'encode/httpx'}, 'zstandard': {'encode/httpx'}}


def python_for(tag: str) -> Path:
    if tag == 'cp313':
        return PYDS / 'py313' / 'python' / 'bin' / 'python3.13'
    return PYDS / 'py312' / 'python' / 'bin' / 'python3.12'


CALIBRATION = {
    'rich_3064': {'pygments': '2.15.1'}, 'rich_3130': {'pygments': '2.15.1'},
    'rich_3296': {'pygments': '2.16.1'}, 'rich_3469': {'pygments': '2.18.0'},
    'rich_3782': {'pygments': '2.19.2'}, 'rich_4070': {'pygments': '2.19.2'},
    'rich_4079': {'pygments': '2.20.0'},
    'requests_6589': {'werkzeug': '2.3.8', 'flask': '2.3.3'},
    'requests_6592': {'werkzeug': '2.3.8', 'flask': '2.3.3'},
    'requests_6629': {'werkzeug': '2.3.8', 'flask': '2.3.3'},
}
TASK_PYTHON = {'requests_6757': 'cp312', 'rich_3472': 'cp312'}
ASYNC_SHIM = {'fastapi_15745', 'fastapi_15763', 'fastapi_15785'}
NO_CERT_TIME = {'requests_6757'}

BASE = {}
SHIM = RUN / 'results' / 'shims'


def base_for(repo: str, tag: str) -> Path:
    if (repo, tag) not in BASE:
        sel = {}
        for name, opts in WHEELS.items():
            opts = [(v, w) for v, w in opts if fits(str(w), tag)]
            if not opts or name == OWN[repo] or (name in ONLY_FOR and repo not in ONLY_FOR[name]):
                continue
            comp = [(v, w) for v, w in opts if w.parent == COMP / 'wheels']
            if name in ('pytest', 'pytest_timeout'):
                sel[name] = next(w for v, w in opts if str(v) in ('8.4.2', '2.1.0'))
            elif comp:
                sel[name] = max(comp)[1]
            elif name != 'pygments':
                sel[name] = max(opts)[1]
        d = RUN / 'results' / f'scorer_site_{OWN[repo]}_{tag}'
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True, exist_ok=True)
        subprocess.run([str(python_for(tag)), '-m', 'pip', 'install', '-q', '--no-index',
                        '--no-deps', '--ignore-requires-python', '--target', str(d),
                        *map(str, sel.values())], check=True)
        for shim in ('imp.py', 'telnetlib.py'):
            shutil.copy(COMP / 'docker' / shim, d / shim)
        BASE[(repo, tag)] = d
    return BASE[(repo, tag)]


def poetry_to_pep440(spec):
    spec = spec if isinstance(spec, str) else spec.get('version', '')
    m = re.fullmatch(r'\^(\d+)(\.\d+)?(\.\d+)?', spec.strip())
    return f'>={spec.strip()[1:]},<{int(m.group(1)) + 1}' if m else ('' if spec.strip() == '*' else spec)


CURRENT = {'id': None}


def setup_env(docker, sandbox_id, repo='', **kwargs):
    tid = CURRENT['id']
    tag = TASK_PYTHON.get(tid, 'cp313')
    site = Path(docker.exec(sandbox_id, 'python -c "import site; print(site.getsitepackages()[0])"').stdout.strip())
    if tag != 'cp313':
        venv_dir = site.parents[2]
        subprocess.run([str(python_for(tag)), '-m', 'venv', '--clear', '--without-pip', str(venv_dir)], check=True)
        site = next(venv_dir.glob('lib/python3*/site-packages'))
    ws = docker.exec(sandbox_id, 'pwd').stdout.strip()
    base = base_for(repo, tag)
    (site / '00_src.pth').write_text(f'{ws}/src\n')
    (site / '01_scorer_base.pth').write_text(f'{base}\n')
    (site / 'zz_workspace_paths.pth').write_text(
        '\n'.join(str(p) for p in Path(ws).iterdir() if p.is_dir() and not p.name.startswith('.')) + '\n')
    if tid in ASYNC_SHIM:
        (SHIM).mkdir(parents=True, exist_ok=True)
        _write_async_shim(SHIM)
        (site / '02_async_shim.pth').write_text(f'{SHIM}\n')
    if tid in NO_CERT_TIME:
        (site / '03_no_cert_time.pth').write_text(f'import sys; sys.path.insert(0, {str(SHIM)!r}); import swe_no_cert_time\n')
    try:
        data = tomllib.loads(Path(ws, 'pyproject.toml').read_text())
    except Exception:
        data = {}
    reqs = list(data.get('project', {}).get('dependencies', []))
    reqs += [n + poetry_to_pep440(s) for n, s in
             data.get('tool', {}).get('poetry', {}).get('dependencies', {}).items() if n != 'python']
    picks, queue, seen = {}, list(reqs), set()
    while queue:
        try:
            req = Requirement(queue.pop(0))
        except Exception:
            continue
        key = norm(req.name)
        if key in seen or key == OWN.get(repo) or (req.marker and not req.marker.evaluate({'extra': ''})):
            continue
        seen.add(key)
        options = [(v, w) for v, w in WHEELS.get(key, [])
                   if fits(str(w), tag) and req.specifier.contains(v, prereleases=True)
                   and not (key in ONLY_FOR and repo not in ONLY_FOR[key])]
        if options:
            best = max(options)[1]
            picks[key] = best
            with zipfile.ZipFile(best) as z:
                meta = next(n for n in z.namelist() if n.endswith('.dist-info/METADATA'))
                queue += [l.split(':', 1)[1].strip() for l in z.read(meta).decode().splitlines()
                          if l.startswith('Requires-Dist:')]
    for name, ver in CALIBRATION.get(tid, {}).items():
        picks[name] = next(w for v, w in WHEELS[name] if str(v) == ver and fits(str(w), tag))
    if picks:
        subprocess.run([str(python_for(tag)), '-m', 'pip', 'install', '-q', '--no-index', '--no-deps',
                        '--upgrade', '--ignore-requires-python', '--target', str(site),
                        *map(str, picks.values())])
    return picks


def _write_async_shim(shim_dir: Path):
    di = shim_dir / 'swe_async_shim-0.1.dist-info'
    di.mkdir(parents=True, exist_ok=True)
    (di / 'METADATA').write_text('Metadata-Version: 2.1\nName: swe-async-shim\nVersion: 0.1\n')
    (di / 'entry_points.txt').write_text('[pytest11]\nswe_async_shim = swe_async_shim\n')
    (shim_dir / 'swe_async_shim.py').write_text('''import asyncio, inspect
import pytest


@pytest.hookimpl(tryfirst=True)
def pytest_pyfunc_call(pyfuncitem):
    if inspect.iscoroutinefunction(pyfuncitem.obj):
        args = {a: pyfuncitem.funcargs[a] for a in pyfuncitem._fixtureinfo.argnames}
        asyncio.run(pyfuncitem.obj(**args))
        return True
    return None
''')
    (shim_dir / 'swe_no_cert_time.py').write_text('''import ssl
ssl._ssl._test_decode_cert = lambda *a, **k: None
''')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--tasks', default='all')
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--num-shards', type=int, default=1)
    ap.add_argument('--results', default=None)
    ap.add_argument('--model-url', default=os.environ.get('MODEL_URL', 'http://127.0.0.1:8000/v1'))
    ap.add_argument('--model-name', default=os.environ.get('MODEL_NAME', 'gemma-4-31b-it-qat-w4a16-ct'))
    ap.add_argument('--agent-dir', default=str(RUN / 'agent'))
    args = ap.parse_args()

    tasks = load_tasks(COMP / 'tasks.jsonl')
    if args.tasks != 'all':
        keep = set(args.tasks.split(','))
        tasks = [t for t in tasks if t.instance_id in keep]
    tasks = [t for i, t in enumerate(tasks) if i % args.num_shards == args.shard]
    print(f'{len(tasks)} tasks', flush=True)

    results_dir = Path(args.results or RUN / 'results' / f'run_shard{args.shard}of{args.num_shards}')
    results_dir.mkdir(parents=True, exist_ok=True)

    models = ModelRegistry()
    models.register(args.model_name, LiteLlm(
        model=f"openai/{args.model_name}", api_base=args.model_url, api_key="EMPTY"))
    limits, gen_constraints = build_submission_limits()

    cfg = EvalConfig(
        tasks_path=COMP / 'tasks.jsonl', snapshots_dir=COMP / 'snapshots',
        results_dir=results_dir, submission_dir=Path(args.agent_dir), models=models,
        sandbox='subprocess', timeout_seconds=240, verification_timeout_seconds=300,
        max_time_minutes=7, max_tool_calls=28, max_turns=80, limits=limits,
        generation_constraints=gen_constraints, adapter_manifest=None,
        context_cache_config=ContextCacheConfig(min_tokens=2048, ttl_seconds=1800, cache_intervals=10),
        events_compaction_config=EventsCompactionConfig(
            compaction_interval=15, overlap_size=2, token_threshold=14336, event_retention_size=5),
        graph_dir=str(COMP / 'graphs'), embeddings_dir=str(COMP / 'embeddings'),
        wheels_dir=COMP / 'wheels', verbose=False, display_mode='quiet')
    ev = Evaluator(cfg)
    ev.sandbox = ev.docker = SubprocessManager(timeout_seconds=240, system_site_packages=False)
    verification.install_test_dependencies = setup_env
    agent_runner.install_test_dependencies = setup_env

    pool = ThreadPoolExecutor(max_workers=1)
    for i, task in enumerate(tasks, 1):
        CURRENT['id'] = task.instance_id
        t0 = time.time()
        try:
            res = pool.submit(asyncio.run, ev.evaluate_task(task=task, task_index=i, total_tasks=len(tasks))).result()
            row = dict(id=task.instance_id, repo=task.repo, resolved=bool(res.resolved),
                       exit_code=res.test_exit_code, tool_calls=res.tool_calls,
                       seconds=round(time.time() - t0, 1), error=(res.error or '')[:200])
        except Exception as e:  # noqa: BLE001
            row = dict(id=task.instance_id, repo=task.repo, resolved=False, exit_code=-1,
                       tool_calls=0, seconds=round(time.time() - t0, 1), error=repr(e)[:200])
        with open(results_dir / 'progress.jsonl', 'a') as fh:
            fh.write(json.dumps(row) + '\n')
        print(json.dumps(row), flush=True)
    print('EVAL DONE', flush=True)


if __name__ == '__main__':
    main()
