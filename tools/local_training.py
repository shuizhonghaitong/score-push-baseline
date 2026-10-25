#!/usr/bin/env python3
"""Queue local GPU Python jobs with explicit devices and a restricted filesystem."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
import uuid

from run_batch import ROOT, now, write_json


def safe_json(path, value):
    fd, temporary = tempfile.mkstemp(prefix='.status-', dir=path.parent)
    with os.fdopen(fd, 'w') as stream:
        stream.write(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    os.replace(temporary, path)


def policy():
    return json.loads((ROOT / 'training-resources.json').read_text())


def validate_run(path):
    run = Path(path).resolve()
    if run.parent != ROOT / 'runs' or not run.is_dir():
        raise ValueError('Run must be an existing direct child of the project runs directory')
    return run


def validate_request(run, request):
    script = Path(request['script']).resolve()
    python = Path(request['python']).resolve()
    if not script.is_relative_to(run) or not script.is_file():
        raise ValueError('Training script must be inside this run')
    if not python.is_file() or not os.access(python, os.X_OK):
        raise ValueError('Python interpreter is unavailable')
    if request['memory_gib'] <= 0:
        raise ValueError('Requested memory must be positive')
    return script, python


def inventory():
    result = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid,memory.free,utilization.gpu',
                             '--format=csv,noheader,nounits'], capture_output=True, text=True,
                            timeout=10, check=True)
    rows = []
    for line in result.stdout.splitlines():
        index, identity, free, utilization = map(str.strip, line.split(','))
        rows.append({'index': int(index), 'uuid': identity, 'free_mib': int(free),
                     'utilization': int(utilization)})
    return rows


def select_gpu(rows, settings, occupied, memory_gib):
    by_index = {row['index']: row for row in rows}
    for index in settings['candidate_gpus']:
        row = by_index.get(index)
        if row and index not in occupied and row['utilization'] <= settings['maximum_start_utilization_percent']:
            if row['free_mib'] >= memory_gib * 1024 + settings['reserve_memory_mib']:
                return row
    return None


def command(run, request, gpu):
    script, python = validate_request(run, request)
    args = ['/usr/bin/bwrap', '--die-with-parent', '--new-session', '--ro-bind', '/', '/',
            '--dev', '/dev', '--proc', '/proc', '--unshare-user', '--unshare-pid',
            '--unshare-ipc', '--unshare-uts', '--unshare-net', '--tmpfs', '/tmp',
            '--bind', str(run), str(run)]
    for name in (f"nvidia{gpu['index']}", 'nvidiactl', 'nvidia-uvm', 'nvidia-uvm-tools'):
        path = Path('/dev') / name
        if path.exists():
            args += ['--dev-bind', str(path), str(path)]
    args += ['--chdir', str(run), '--', str(python), str(script), *request.get('args', [])]
    return args


def job_environment(gpu):
    env = {k: v for k, v in os.environ.items() if not k.startswith(('CODEX_', 'KAGGLE_'))}
    env.update(CUDA_VISIBLE_DEVICES=gpu['uuid'], PYTHONDONTWRITEBYTECODE='1',
               OMP_NUM_THREADS='8', MKL_NUM_THREADS='8', HF_ENDPOINT='https://hf-mirror.com',
               HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1')
    return env


def service():
    if (ROOT / '.local/EXPERIMENTS_PAUSED.json').exists():
        raise SystemExit('Experiment training paused by user; review the saved checkpoint before restarting')
    folder = ROOT / '.local/local-training'
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / 'service.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        active = {}
        while True:
            for job, record in list(active.items()):
                code = record['child'].poll()
                if code is None:
                    continue
                state = json.loads((job / 'status.json').read_text())
                state.update(status='finished' if code == 0 else 'failed', exit_code=code,
                             finished_at=now())
                safe_json(job / 'status.json', state)
                del active[job]
            settings = policy()
            try:
                devices = inventory()
            except (OSError, ValueError, subprocess.SubprocessError):
                devices = []
            for path in sorted((ROOT / 'runs').glob('*/gpu-jobs/*/request.json')):
                job = path.parent
                status = job / 'status.json'
                if path.is_symlink() or job.resolve() != job or path.stat().st_uid != os.getuid():
                    continue
                try:
                    run = validate_run(job.parent.parent)
                    if status.exists() and json.loads(status.read_text())['status'] != 'queued':
                        continue
                    request = json.loads(path.read_text())
                    validate_request(run, request)
                    occupied = {record['gpu']['index'] for record in active.values()}
                    gpu = select_gpu(devices, settings, occupied, request['memory_gib'])
                    if gpu is None:
                        continue
                    with os.fdopen(os.open(job / 'stdout.log', os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600), 'w') as out, \
                         os.fdopen(os.open(job / 'stderr.log', os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o600), 'w') as err:
                        child = subprocess.Popen(command(run, request, gpu), env=job_environment(gpu),
                                                 stdout=out, stderr=err)
                    state = {'status': 'running', 'pid': child.pid, 'gpu': gpu['index'],
                             'gpu_uuid': gpu['uuid'], 'started_at': now(),
                             'requested_memory_gib': request['memory_gib']}
                    safe_json(status, state)
                    active[job] = {'child': child, 'gpu': gpu}
                except (OSError, ValueError, KeyError, TypeError) as error:
                    safe_json(status, {'status': 'failed', 'finished_at': now(),
                                        'error_type': type(error).__name__, 'error': str(error)[:300]})
            write_json(folder / 'status.json', {'updated_at': now(), 'pid': os.getpid(),
                       'candidate_gpus': settings['candidate_gpus'],
                       'active': [{'job': str(job), 'gpu': rec['gpu']['index']}
                                  for job, rec in active.items()]})
            time.sleep(3)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    sub.add_parser('service')
    submit = sub.add_parser('submit')
    submit.add_argument('--run-dir', type=Path, default=Path.cwd())
    submit.add_argument('--python', type=Path)
    submit.add_argument('--memory-gib', type=float, default=16)
    submit.add_argument('script', type=Path)
    submit.add_argument('args', nargs=argparse.REMAINDER)
    status = sub.add_parser('status')
    status.add_argument('job', type=Path)
    args = parser.parse_args()
    if args.action == 'service':
        service()
    elif args.action == 'submit':
        run = validate_run(args.run_dir)
        if (run / 'PAUSE_STATE.json').exists():
            raise SystemExit('This run is paused by user; review PAUSE_STATE.json before queuing work')
        request = {'script': str(args.script.resolve()),
                   'python': str((args.python or Path(policy()['cuda_python'])).resolve()),
                   'memory_gib': args.memory_gib, 'args': args.args[1:] if args.args[:1] == ['--'] else args.args,
                   'created_at': now()}
        validate_request(run, request)
        job = run / 'gpu-jobs' / (str(time.time_ns()) + '-' + uuid.uuid4().hex[:8])
        job.mkdir(parents=True, mode=0o700)
        write_json(job / 'status.json', {'status': 'queued', 'created_at': now()})
        write_json(job / 'request.json', request)
        print(job)
    else:
        job = args.job.resolve()
        run = validate_run(job.parent.parent)
        if job.parent != run / 'gpu-jobs':
            raise ValueError('Invalid job location')
        print((job / 'status.json').read_text())
        for name in ('stdout.log', 'stderr.log'):
            if (job / name).exists():
                print(name + ':\n' + (job / name).read_text()[-4000:])


if __name__ == '__main__':
    main()
