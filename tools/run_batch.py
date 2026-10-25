#!/usr/bin/env python3
"""Run the authorized Kaggle tasks with the existing local Codex and Qwen services."""
import argparse
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path('/mnt/disk3/liuyutong_home/miniconda3/envs/kaggle/bin/python')
CODEX = Path('/mnt/disk3/liuyutong_home/.codex/packages/standalone/releases/0.162.1-x86_64-unknown-linux-musl/bin/codex')
AUTH = ROOT.parent / 'reruns/20261021-r6-comparison-abc/01-baseline/a/runtime/codex/auth.json'
TOKEN = ROOT / '.private/kaggle-access-token'
SKILL = ROOT / 'skills/kaggle'


def now():
    return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat()


def write_json(path, value):
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temp.replace(path)


def environment(profile, training_gpu='0'):
    env = os.environ.copy()
    for name in list(env):
        if name.startswith(('CODEX_', 'KAGGLE_')):
            env.pop(name)
    env['CODEX_HOME'] = str(profile)
    env['KAGGLE_API_TOKEN'] = str(TOKEN)
    env['PATH'] = str(PYTHON.parent) + ':' + str(CODEX.parent) + ':' + env.get('PATH', '')
    for name in ['HTTP_PROXY', 'HTTPS_PROXY', 'http_proxy', 'https_proxy']:
        env[name] = 'http://172.28.4.16:18999'
    exclusions = env.get('NO_PROXY', env.get('no_proxy', ''))
    env['NO_PROXY'] = env['no_proxy'] = 'localhost,127.0.0.1,::1,172.28.4.16,' + exclusions
    env['HF_ENDPOINT'] = 'https://hf-mirror.com'
    env['TZ'] = 'Asia/Shanghai'
    env['PYTHONDONTWRITEBYTECODE'] = '1'
    # Inference endpoints and local training allocations are independent.
    env['CUDA_VISIBLE_DEVICES'] = training_gpu
    env['OMP_NUM_THREADS'] = env['MKL_NUM_THREADS'] = '8'
    return env


def prepare(label, competition, port, smoke=False, training_gpu='0', recovery_source=None):
    run = ROOT / 'runs' / label
    profile = ROOT / '.local' / label / 'codex'
    run.mkdir(parents=True, exist_ok=True)
    (profile / 'skills').mkdir(parents=True, exist_ok=True)
    skill_link = profile / 'skills/kaggle'
    if not skill_link.exists():
        skill_link.symlink_to(SKILL, target_is_directory=True)
    if not (profile / 'auth.json').exists():
        shutil.copy2(AUTH, profile / 'auth.json')
        (profile / 'auth.json').chmod(0o600)
    config = f'''model = "qwen3.8-27b-fp8"
model_provider = "custom"
model_reasoning_effort = "medium"
approval_policy = "never"
default_permissions = "kaggle-run"
cli_auth_credentials_store = "file"
web_search = "disabled"

[model_providers.custom]
name = "Qwen3.8-27B-FP8 GPU {port - 12800}"
base_url = "http://127.0.0.1:{port}/v1"
wire_api = "responses"
requires_openai_auth = true

[permissions.kaggle-run.filesystem]
":root" = "read"
"{run}" = "write"
"{profile}" = "write"
"{ROOT / '.private'}" = "write"
"/tmp" = "write"
[permissions.kaggle-run.network]
enabled = true

[features]
multi_agent = false
'''
    (profile / 'config.toml').write_text(config)
    pool_path = ROOT / 'training-resources.json'
    pool = json.loads(pool_path.read_text()) if pool_path.exists() else {}
    training_note = (
        f'The exposed local training device is host GPU{training_gpu}, shared with existing jobs: '
        'inspect available memory/utilization before training and avoid disrupting them.'
        if training_gpu else
        'No local training GPU is allocated to this run. Use local CPU or authorized '
        'private Kaggle remote compute; preserve the empty CUDA_VISIBLE_DEVICES setting.'
    )
    if pool.get('mode') == 'local_gpu_pool':
        training_note = f'''Prefer local GPU training and feature extraction using the shared pool on
host GPUs {pool['candidate_gpus']}. The ordinary Codex tool sandbox hides NVIDIA
devices, so CUDA_VISIBLE_DEVICES alone does not enable local training. Read
{ROOT / 'LOCAL_TRAINING.md'} and use {ROOT / 'tools/local_training.py'} to queue
Python training scripts from this run. The host worker supplies explicit GPU
devices, restricts writes to this run, and returns logs/checkpoints here.
Reuse the existing CUDA-enabled Python at {pool['cuda_python']}; a run-local venv
with --system-site-packages can add dependencies. Do not replace it with CPU-only
PyTorch. Prepare data/weights in the normal tool environment first; training jobs
run offline. Respect existing jobs and wait for an available pool GPU. Use Kaggle
remote GPU when local resources cannot support the workload or the competition
requires remote execution. Record the actual training host/GPU in the report.'''
    (run / 'AGENTS.md').write_text(f'''# Run environment
Use the available $kaggle skill. This is one unattended competition run; no human
will provide advice or operate the tools during it. Resolve errors autonomously;
if an unavoidable login, agreement or permission is missing, record a blocked
RESULT.json and finish. Authorized: modelling, training, downloading allowed data,
private competition Notebooks, and multiple useful official submissions within
the live Kaggle quota. Inspect previews then proceed without asking again.

The competition is {competition}. Keep code, evaluated attempts and RESULT.json
in this directory. Reuse authorized past code and data; preserve originals. The
official raw-data caches include
/mnt/disk3/liuyutong_home/workspace/kaggle and
{ROOT.parent / 'reruns'}. No old history exclusions or one-submit cap apply.
Name That LLM historical preflight evidence is saved at
{ROOT / ".local/name-that-llm-history-preflight.json"}. The old B02 protected
attempt remains unknown in its preserved ledger; do not restore, modify or retry
that old job/artifact. This new autonomous run is separately authorized: use
distinct candidate code and a private Notebook, check live account submissions
and quota, and reconcile any uncertain writes created by this run before continuing.

Python and Kaggle CLI are from {PYTHON.parent}. The shared environment is read-only;
install additional packages in a project-local venv with --system-site-packages.
Use exec_command with login=false when available: host login startup scripts can
replace proxy variables and PATH. Preserve the configured environment in child
shells and verify its effective route before transfers. Reuse packages and caches.
Kaggle authentication is already selected by KAGGLE_API_TOKEN pointing to a private
file; do not print its contents or credentials. If it expires, preview then run
the skill's auth-session command with --output {TOKEN} and --yes.

Both cases of HTTP(S)_PROXY explicitly use the unmetered 172.28.4.16:18999 route.
Preserve localhost/private NO_PROXY. Probe destinations before large transfers;
do not fall back to 17890 or unconfirmed proxies. HF_ENDPOINT=https://hf-mirror.com
is set before imports; prefer local caches. Stop large transfers if the unmetered
route is unavailable. Public research through bounded HTTP requests is available;
the hosted web-search tool is not configured for this local model.

Qwen inference runs on GPUs 4-7; do not stop or modify their services.
The provider accepts at most four image inputs per request, including images
already present in conversation history. Use a contact sheet for visual previews
and text/numeric inspection for subsequent bulk analysis; avoid accumulating
more than four images in the model context.
{training_note}
CPU and authorized Kaggle private remote compute are available. Other runs may
share the account's remote accelerator quota and concurrency; inspect active jobs
and wait instead of cancelling another run. Limit CPU thread oversubscription.
No fixed experiment-round or score threshold is imposed. Finish after useful
improvement stalls, preserving the best valid artifacts and scored submission.
Distinguish this run's best public score from account historical best/rank and
save timestamped rank evidence; unverified values stay null.
''')
    if recovery_source is not None:
        source = Path(recovery_source).resolve()
        if not source.is_relative_to(ROOT / 'runs') or source == run.resolve() or not source.is_dir():
            raise ValueError('Recovery source must be a separate existing run')
        with (run / 'AGENTS.md').open('a') as instructions:
            instructions.write(f'''\n# Recovery attempt\nThis is a separately recorded recovery of {source}. Preserve that directory and
its failed execution evidence. Reuse its cached data, extracted features and code
by reading or copying them into this new run. Report this recovery separately.
The previous attempt ended because accumulated image inputs exceeded the provider
limit. For this recovery use numeric inspection and local image feature extraction;
do not send image inputs into the Qwen conversation. The training pipeline can
still process image files normally.\n''')
    template = (ROOT / 'prompt.template.txt').read_text()
    prompt = template.replace('{{competition}}', competition)
    if smoke:
        prompt = ('使用 $kaggle 在 Titanic 做一次最小端到端验证：复用缓存、简单训练验证，完成一次文件提交并确认成绩和排名；'
                  '再运行一个私有 CPU Notebook，确认其成功和输出下载，无需第二次正式提交。不做冲分。'
                  '全程自主完成，把提交 ID、分数、排名证据、Notebook 名称及版本和核验结果写入 RESULT.json。')
    (run / 'prompt.txt').write_text(prompt)
    return run, profile, prompt


def execute(label, competition, port, smoke=False, training_gpu='0', recovery_source=None):
    if (ROOT / '.local/EXPERIMENTS_PAUSED.json').exists():
        raise RuntimeError('Experiments paused by user; review the saved checkpoint before starting')
    pool_path = ROOT / 'training-resources.json'
    pool = json.loads(pool_path.read_text()) if pool_path.exists() else {}
    if pool.get('mode') == 'local_gpu_pool':
        training_gpu = ''
    status_path = ROOT / 'runs' / label / 'status.json'
    if status_path.exists():
        raise RuntimeError(f'{label} already has execution state; inspect before resuming')
    run, profile, prompt = prepare(label, competition, port, smoke, training_gpu, recovery_source)
    status_path = run / 'status.json'
    if status_path.exists():
        raise RuntimeError(f'{label} already has execution state; inspect before resuming')
    env = environment(profile, training_gpu)
    # Renew/select the existing authorized login before every independent run.
    with (run / 'auth-check.log').open('w') as log:
        auth = subprocess.run([str(PYTHON), str(SKILL / 'scripts/kaggle_skill.py'),
                               'auth-session', '--output', str(TOKEN), '--yes'],
                              env=env, stdout=log, stderr=log, timeout=90)
    if auth.returncode:
        write_json(status_path, {'status': 'auth_blocked', 'time': now(), 'exit_code': auth.returncode})
        return False
    state = {'status': 'running', 'competition': competition, 'model': 'qwen3.8-27b-fp8',
             'port': port, 'training_gpu': training_gpu or None,
             'started_at': now(), 'run': str(run), 'profile': str(profile)}
    if pool.get('mode') == 'local_gpu_pool':
        state.update(local_training_mode='gpu_pool', local_training_gpu_pool=pool['candidate_gpus'])
    write_json(status_path, state)
    args = [str(CODEX), '--no-daemon', 'exec', '--skip-git-repo-check', '--json',
            '--color', 'never', '-C', str(run), '-o', str(run / 'final.txt'), '-']
    with (run / 'events.jsonl').open('w') as out, (run / 'stderr.log').open('w') as err:
        child = subprocess.Popen(args, stdin=subprocess.PIPE, stdout=out, stderr=err, env=env)
        state['pid'] = child.pid
        write_json(status_path, state)
        child.communicate(prompt.encode())
    state.update(status='finished' if child.returncode == 0 else 'execution_failed',
                 exit_code=child.returncode, finished_at=now(), result_exists=(run / 'RESULT.json').exists())
    write_json(status_path, state)
    return child.returncode == 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['smoke', 'six'])
    a = parser.parse_args()
    (ROOT / '.local').mkdir(exist_ok=True)
    with (ROOT / '.local/batch.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if a.stage == 'smoke':
            return 0 if execute('00-titanic-smoke-v2', 'titanic', 12804, smoke=True) else 1
        approval = ROOT / '.local/smoke-verified.json'
        if not approval.exists() or not json.loads(approval.read_text()).get('passed'):
            raise RuntimeError('First verify the smoke submission and Notebook evidence')
        for existing in (ROOT / 'runs').glob('[0-9][0-9]-*/status.json'):
            if not existing.parent.name.startswith('00-'):
                raise RuntimeError('Existing six-task execution state; inspect instead of starting duplicates')
        plan_path = ROOT / 'plan.json'
        plan = json.loads(plan_path.read_text())
        plan['status'] = 'six_runs_in_progress'
        write_json(plan_path, plan)
        tasks = plan['tasks']
        for i, task in enumerate(tasks):
            label = f"{task['order']:02d}-{task['competition']}"
            try:
                execute(label, task['competition'], 12804 + i % 4)
            except Exception as exc:
                # One task's environment failure must not prevent the remaining
                # selected competitions from being attempted. Preserve its logs.
                state_path = ROOT / 'runs' / label / 'status.json'
                state = json.loads(state_path.read_text()) if state_path.exists() else {}
                state.update(status='execution_failed', finished_at=now(),
                             runner_error_type=type(exc).__name__)
                write_json(state_path, state)
        plan['status'] = 'six_runs_finished_review_results'
        write_json(plan_path, plan)
        return 0


if __name__ == '__main__':
    raise SystemExit(main())
