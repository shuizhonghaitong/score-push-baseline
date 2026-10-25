#!/usr/bin/env python3
"""Build the private Kaggle A/B notebook (v1 vs v2 on the devset)."""
import json
import sys
from pathlib import Path

RUN = Path(__file__).resolve().parents[1]
OUT = RUN / 'kaggle_ab'
OUT.mkdir(exist_ok=True)

# ---- 1. patch eval_local.py for Kaggle paths
src = (RUN / 'scripts' / 'eval_local.py').read_text()
repls = [
    ("import argparse\nimport asyncio", "import argparse\nimport asyncio\nimport glob"),
    ('RUN = Path(__file__).resolve().parents[1]', "RUN = Path('/kaggle/working')"),
    ('COMP = RUN / "data"', "COMP = Path('/kaggle/input/competitions/gemma-4-developer-agent')"),
    ('WHEELHOUSE = RUN / "models" / "wheelhouse"',
     "WHEELHOUSE = Path('/kaggle/input/datasets/metric/gemma-4-developer-agent-wheelhouse')"),
    ('EXTRA = RUN / "models" / "local-eval" / "wheels"',
     "EXTRA = Path(glob.glob('/kaggle/input/**/inline_snapshot-*.whl', recursive=True)[0]).parent"),
    ('PYDS = RUN / "models" / "local-eval" / "pythons"',
     "PYDS = Path(glob.glob('/kaggle/input/**/cpython-3.12-linux.tgz.bin', recursive=True)[0]).parent"),
    ("""def python_for(tag: str) -> Path:
    if tag == 'cp313':
        return PYDS / 'py313' / 'python' / 'bin' / 'python3.13'
    return PYDS / 'py312' / 'python' / 'bin' / 'python3.12'""",
     """def python_for(tag: str) -> Path:
    if tag == 'cp313':
        return Path(sys.executable)
    dest = RUN / f'py{tag}'
    ver = f'{tag[2]}.{tag[3:]}'
    py = dest / 'python' / 'bin' / f'python{ver}'
    if not py.exists():
        dest.mkdir(parents=True, exist_ok=True)
        with tarfile.open(PYDS / f'cpython-{ver}-linux.tgz.bin', 'r:gz') as t:
            t.extractall(dest)
    return py"""),
]
for old, new in repls:
    assert old in src, f'missing: {old[:60]}'
    src = src.replace(old, new)
eval_src = src

# ---- 2. agent bundles
def bundle(rel: Path) -> dict:
    base = RUN / rel
    files = {}
    for p in sorted(base.rglob('*')):
        if p.is_file():
            files[str(p.relative_to(base))] = p.read_text()
    return files

V1 = bundle('agent')
V2 = bundle('agent-v2')

# ---- 3. devset
devset_file = RUN / 'results' / 'devset_kaggle.txt'
devset = devset_file.read_text().strip() if devset_file.exists() else 'FILL_DEVSET_BEFORE_PUSH'
assert devset != 'FILL_DEVSET_BEFORE_PUSH', 'devset_kaggle.txt missing'

CELLS = []

def code(s):
    CELLS.append({'cell_type': 'code', 'execution_count': None, 'metadata': {}, 'outputs': [], 'source': s.splitlines(keepends=True)})

def md(s):
    CELLS.append({'cell_type': 'markdown', 'metadata': {}, 'source': s.splitlines(keepends=True)})

md("# run14 A/B: v1 vs v2 on the Kaggle A100\n\n"
   "Same devset, same model instance, sequential. Writes results/results_v{1,2}/progress.jsonl and an ab_summary.")

code("""import os
os.environ.update({
    'TRANSFORMERS_NO_TF': '1', 'TRANSFORMERS_NO_FLAX': '1', 'TRANSFORMERS_OFFLINE': '1',
    'VLLM_NO_USAGE_STATS': '1', 'VLLM_WORKER_MULTIPROC_METHOD': 'spawn',
    'VLLM_ENGINE_READY_TIMEOUT_S': '1800', 'VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS': '1',
    'NCCL_DEBUG': 'WARN', 'LITELLM_LOCAL_MODEL_COST_MAP': 'True', 'LITELLM_TELEMETRY': 'False',
    'LITELLM_LOG': 'WARNING', 'OPENAI_API_KEY': 'EMPTY', 'LOCAL_API_KEY': 'EMPTY',
    'PYTORCH_CUDA_ALLOC_CONF': 'expandable_segments:True',
    'OTEL_SDK_DISABLED': 'true', 'OTEL_PYTHON_DISABLED': 'true',
})
import logging
for name in ('LiteLLM', 'litellm', 'httpx', 'httpcore', 'openai', 'urllib3', 'opentelemetry'):
    lg = logging.getLogger(name)
    lg.setLevel(logging.WARNING)
    if 'litellm' in name.lower():
        lg.propagate = False
        lg.handlers.clear()
import torch
print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), 'gpus', torch.cuda.device_count())
""")

code("""import glob, os, shutil, subprocess, sys
from pathlib import Path

for pth_pattern in ('/usr/local/lib/python*/dist-packages/*cutlass*.pth',
                    '/usr/local/lib/python*/site-packages/*cutlass*.pth'):
    for pth in glob.glob(pth_pattern):
        try:
            os.unlink(pth)
        except OSError:
            pass

WH = Path('/kaggle/input/datasets/metric/gemma-4-developer-agent-wheelhouse')
tmp = Path('/tmp/wheelhouse'); tmp.mkdir(exist_ok=True)
n = 0
for w in sorted(WH.glob('*.whl')):
    if 'cutlass' in w.name.lower():
        continue
    target_name = w.name.replace('cu128', '+cu128') if ('cu128' in w.name and '+' not in w.name) else w.name
    target = tmp / target_name
    if not target.exists():
        try:
            os.symlink(w, target)
        except OSError:
            shutil.copyfile(w, target)
    n += 1
print('installing', n, 'wheels')
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '--no-deps', '--force-reinstall',
                *sorted(str(w) for w in tmp.glob('*.whl'))], check=True)
import importlib
importlib.invalidate_caches()
import vllm, transformers
print('vllm', vllm.__version__, 'transformers', transformers.__version__)
""")

import base64
eval_b64 = base64.b64encode(eval_src.encode()).decode()
agents_b64 = base64.b64encode(json.dumps({'v1': V1, 'v2': V2}).encode()).decode()

code(f"""import base64, glob
from pathlib import Path
COMP = Path(glob.glob('/kaggle/input/**/tasks.jsonl', recursive=True)[0]).parent
assert (COMP / 'docker' / 'imp.py').exists()
Path('/kaggle/working').mkdir(parents=True, exist_ok=True)
EVAL_SRC = base64.b64decode(''.join(r'''
{eval_b64}
'''.splitlines())).decode()
(Path('/kaggle/working/eval_local_kaggle.py')).write_text(EVAL_SRC)
import ast as _a; _a.parse(EVAL_SRC); print('wrote eval_local_kaggle.py', len(EVAL_SRC), 'parses OK')
""")


code(f"""import base64, json, shutil
from pathlib import Path
BUNDLES = json.loads(base64.b64decode(''.join(r'''
{agents_b64}
'''.splitlines())).decode())
for tag, files in BUNDLES.items():
    d = Path('/kaggle/working') / f'agent_{{tag}}'
    if d.exists():
        shutil.rmtree(d)
    for rel, text in files.items():
        p = d / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
    print(tag, '->', sorted(files))
""")


code("""import re
import litellm
litellm.drop_params = True
from adk_submission import VllmConfig, VllmServer, discover_adapters
from adk_submission.discovery import discover_declared_models
from swegemma.config import ALLOWED_ADAPTER_EXTENSIONS

MODEL_NAME = 'gemma-4-31b-it-qat-w4a16-ct'
MODEL_PATH = Path(f'/kaggle/input/models/google/gemma-4/other/{MODEL_NAME}/2')
assert MODEL_PATH.exists(), MODEL_PATH
AGENT_DIR = Path('/kaggle/working/agent_v1')
declared = discover_declared_models(str(AGENT_DIR))
assert list(declared) == [MODEL_NAME], declared
adapters = discover_adapters(str(AGENT_DIR), adapter_extensions=ALLOWED_ADAPTER_EXTENSIONS)
gpu_count = torch.cuda.device_count()
tp_size = 1 if gpu_count == 1 else (2 if gpu_count == 2 else 4)
vllm_cfg = VllmConfig(
    model=str(MODEL_PATH), port=8000, host='127.0.0.1',
    tool_call_parser='gemma4', reasoning_parser='gemma4',
    max_model_len=32768, dtype='bfloat16', gpu_memory_utilization=0.90,
    enable_auto_tool_choice=True, enable_lora=False,
    scheduler_reserve_full_isl=False, tensor_parallel_size=tp_size,
    startup_timeout=60 * 30, extra_args=())
server_instance = VllmServer(vllm_cfg, adapter_manifest=adapters)
server_instance.start()
print('server at', server_instance.base_url, 'tp', tp_size)
""")

code(f"""import os, sys, time
from pathlib import Path
base = str(server_instance.base_url).rstrip('/')
os.environ['MODEL_URL'] = base if base.endswith('/v1') else base + '/v1'
os.environ['MODEL_NAME'] = MODEL_NAME
print('MODEL_URL', os.environ['MODEL_URL'])
sys.path.insert(0, '/kaggle/working')
import eval_local_kaggle as EL
DEVSET = r'''
{devset}
'''.strip()
print('devset:', len(DEVSET.split(',')), 'tasks')
for tag in ('v2', 'v1'):
    results = Path('/kaggle/working') / f'results_{{tag}}'
    results.mkdir(parents=True, exist_ok=True)
    (results / 'progress.jsonl').write_text('')
    sys.argv = ['eval_local.py', '--tasks', DEVSET, '--results', str(results),
                '--agent-dir', str(Path('/kaggle/working') / f'agent_{{tag}}')]
    t0 = time.time()
    EL.main()
    print(tag, 'done in', round(time.time() - t0), 's')
""")

code("""import json
from pathlib import Path
rows = {}
for tag in ('v1', 'v2'):
    rows[tag] = [json.loads(l) for l in (Path(f'/kaggle/working/results_{tag}/progress.jsonl')).read_text().splitlines() if l.strip()]
def summ(rows):
    r = [x for x in rows if x.get('resolved')]
    return len(r), len(rows)
a, na = summ(rows['v1']); b, nb = summ(rows['v2'])
w1 = sum(1 for x in rows['v1'] if x['resolved'] and not next(y for y in rows['v2'] if y['id'] == x['id'])['resolved'])
w2 = sum(1 for x in rows['v2'] if x['resolved'] and not next(y for y in rows['v1'] if y['id'] == x['id'])['resolved'])
print(f'v1: {a}/{na} resolved   v2: {b}/{nb} resolved   v1-only wins {w1}  v2-only wins {w2}')
comp = []
for x in rows['v1']:
    y = next((q for q in rows['v2'] if q['id'] == x['id']), None)
    comp.append({'id': x['id'], 'repo': x['repo'], 'v1': int(x['resolved']),
                 'v2': int(y['resolved']) if y else -1,
                 'v1_calls': x.get('tool_calls'), 'v2_calls': y.get('tool_calls') if y else None,
                 'v1_s': x.get('seconds'), 'v2_s': y.get('seconds') if y else None})
import csv, io
buf = io.StringIO()
w = csv.DictWriter(buf, fieldnames=list(comp[0].keys()))
w.writeheader(); w.writerows(comp)
out = Path('/kaggle/working/ab_summary.csv')
out.write_text(buf.getvalue())
print(out.read_text())
""")

nb = {
    'nbformat': 4,
    'nbformat_minor': 4,
    'metadata': {
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python', 'version': '3.13'},
        'kaggle': {
            'accelerator': 'gpu_a100_80gm',
            'dataSources': [
                {'sourceType': 'competition', 'sourceId': 149921},
                {'sourceType': 'dataset', 'sourceId': 'metric/gemma-4-developer-agent-wheelhouse'},
                {'sourceType': 'model', 'sourceId': 'google/gemma-4'},
            ],
            'isInternetEnabled': False,
            'isGpuEnabled': True,
            'language': 'python',
            'sourceType': 'notebook',
        },
    },
    'cells': CELLS,
}
(OUT / 'run14_ab.ipynb').write_text(json.dumps(nb, indent=1))
meta = {
    'id': 'lilac1996/run14-gemma4-ab-v1v2',
    'title': 'run14 gemma4 ab v1 vs v2 (private)',
    'code_file': 'run14_ab.ipynb',
    'language': 'python',
    'kernel_type': 'notebook',
    'is_private': True,
    'enable_gpu': True,
    'enable_tpu': False,
    'enable_internet': False,
    'keywords': ['automated'],
    'dataset_sources': ['metric/gemma-4-developer-agent-wheelhouse'],
    'kernel_sources': [],
    'competition_sources': ['gemma-4-developer-agent'],
    'model_sources': ['google/gemma-4'],
}
(OUT / 'kernel-metadata.json').write_text(json.dumps(meta, indent=1))
print('built', OUT / 'run14_ab.ipynb', len(json.dumps(nb)), 'bytes')
print('cells:', len(CELLS))
