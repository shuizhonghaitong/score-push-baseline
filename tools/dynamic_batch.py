#!/usr/bin/env python3
"""Fill free endpoints in difficulty-batch order without completion barriers."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import os
from pathlib import Path
import re
import signal
import time
import urllib.request

from parallel_batch import process, same_process
import run_batch as runner

ROOT = runner.ROOT
PORTS = [12804, 12805, 12806, 12807]
TASKS_PER_ENDPOINT = 3
MAX_CONCURRENT_RUNS = len(PORTS) * TASKS_PER_ENDPOINT
TERMINAL = {'finished', 'execution_failed', 'auth_blocked'}


def state_path(task):
    return ROOT / 'runs' / f"{task['order']:02d}-{task['competition']}" / 'status.json'


def states(tasks):
    return {t['order']: json.loads(state_path(t).read_text()) if state_path(t).exists()
            else {'status': 'queued'} for t in tasks}


def current_batch(tasks, current):
    return min((t.get('batch', 1) for t in tasks
                if current[t['order']]['status'] not in TERMINAL), default=None)


def candidates(initial, followup, current, reserved=()):
    tasks = initial + followup
    for task in sorted(tasks, key=lambda t: (t.get('batch', 1), t['order'])):
        if task['order'] in reserved:
            continue
        if current[task['order']]['status'] != 'queued':
            continue
        yield task


def read_followup(path, initial, previous):
    """Accept appended tasks while preserving every existing task identity."""
    document = json.loads(path.read_text())
    later = document['tasks']
    tasks = initial + later
    by_order = {t['order']: t for t in tasks}
    if len(by_order) != len(tasks):
        raise ValueError('Duplicate task order in queue')
    for task in previous:
        if by_order.get(task['order'], {}).get('competition') != task['competition']:
            raise ValueError('Existing task removed or reassigned')
    seen = {}
    for task in tasks:
        prior = seen.get(task['competition'])
        if prior and task.get('retry_of_order') != prior['order']:
            raise ValueError('Duplicate competition without an explicit recovery task')
        if prior is None:
            seen[task['competition']] = task
    return document, later


def endpoint_available(port):
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(f'http://127.0.0.1:{port}/metrics', timeout=3) as response:
            body = response.read().decode()
        def gauge(name):
            values = re.findall(r'^vllm:' + name + r'(?:\{[^\n]*\})?\s+([\d.eE+-]+)\s*$', body, re.MULTILINE)
            return sum(float(x) for x in values) if values else None
        running, waiting = gauge('num_requests_running'), gauge('num_requests_waiting')
        cache = gauge('kv_cache_usage_perc')
        if cache is None:
            cache = gauge('gpu_cache_usage_perc')
        return (running is not None and running < TASKS_PER_ENDPOINT and waiting == 0
                and cache is not None and cache < .70)
    except (OSError, ValueError):
        return False


def free_slot(active, port):
    occupied = {job['slot'] for job in active.values() if job['port'] == port}
    return next((n for n in range(1, TASKS_PER_ENDPOINT + 1) if n not in occupied), None)


def adopt_active(transition):
    active = {}
    for job in transition['adopted']:
        if job['done']:
            continue
        order, port = job['task']['order'], job['port']
        slot = free_slot(active, port)
        if order in active or slot is None:
            raise RuntimeError('Adopted jobs exceed task identity or endpoint capacity')
        active[order] = {'adopted': job, 'task': job['task'], 'port': port, 'slot': slot,
                         'training_gpu': job.get('training_gpu')}
    return active


def hold_old_queues(initial):
    """Freeze queue owners, never their existing Codex jobs."""
    old = json.loads((ROOT / '.local/parallel-transition.json').read_text())
    console = json.loads((ROOT / '.local/console-transition.json').read_text())
    original_plan = json.loads((ROOT / 'plan.json').read_text())
    server = console['server']
    if not same_process(server['pid'], server['start_ticks']):
        raise RuntimeError('Original console identity changed')
    # The previous helper would resume this console after Housing alone; the
    # new owner waits for all adopted jobs before resuming it.
    helper_path = str(ROOT / 'tools/retire_serial_console.py').encode()
    helpers = []
    for entry in Path('/proc').iterdir():
        if not entry.name.isdigit():
            continue
        try:
            args = (entry / 'cmdline').read_bytes().split(b'\0')
        except OSError:
            continue
        if len(args) > 1 and args[1] == helper_path:
            helpers.append(int(entry.name))
    for pid in helpers:
        os.kill(pid, signal.SIGKILL)
    os.kill(server['pid'], signal.SIGSTOP)
    controllers = []
    for pid in {old['dispatcher']['pid'], original_plan['scheduler_pid']}:
        item = process(pid)
        if not item or item['state'] == 'Z':
            continue
        args = Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
        allowed = [str(ROOT / 'tools/run_batch.py').encode(),
                   str(ROOT / 'tools/parallel_batch.py').encode()]
        if not any(path in args for path in allowed):
            raise RuntimeError('Unexpected old queue owner')
        os.kill(pid, signal.SIGSTOP)
        controllers.append(item)
    time.sleep(.1)
    for item in controllers:
        current = same_process(item['pid'], item['start_ticks'])
        if current and current['state'] not in ['T', 't']:
            raise RuntimeError('Old queue did not remain stopped')
    adopted = []
    for task in initial:
        path = state_path(task)
        if not path.exists():
            continue
        state = json.loads(path.read_text())
        if state['status'] != 'running':
            continue
        worker = process(state['pid'])
        if not worker or worker['ppid'] not in {x['pid'] for x in controllers}:
            raise RuntimeError('Existing job has an unexpected owner')
        adopted.append({'task': task, 'worker': worker, 'port': state['port'],
                        'training_gpu': state.get('training_gpu', '0' if task['order'] == 1 else None),
                        'done': False})
    transition = {'started_at': runner.now(), 'server': server,
                  'controllers': controllers, 'adopted': adopted,
                  'note': 'Existing Codex jobs are adopted unchanged; obsolete queue owners stay stopped.'}
    runner.write_json(ROOT / '.local/dynamic-transition.json', transition)
    return transition


def update_adopted(transition):
    changed = False
    for job in transition['adopted']:
        if job['done']:
            continue
        worker = same_process(job['worker']['pid'], job['worker']['start_ticks'])
        if not worker:
            raise RuntimeError('Adopted job disappeared before its exit was observed')
        if worker['state'] != 'Z':
            continue
        code = os.waitstatus_to_exitcode(worker['wait_status'])
        path = state_path(job['task'])
        state = json.loads(path.read_text())
        state.update(status='finished' if code == 0 else 'execution_failed', exit_code=code,
                     finished_at=runner.now(), result_exists=(path.parent / 'RESULT.json').exists(),
                     dispatcher_adopted=True)
        runner.write_json(path, state)
        job.update(done=True, exit_code=code, finished_at=runner.now())
        changed = True
    for controller in transition['controllers']:
        if controller.get('retired'):
            continue
        owned = [j for j in transition['adopted'] if j['worker']['ppid'] == controller['pid']]
        if not all(j['done'] for j in owned):
            continue
        old = same_process(controller['pid'], controller['start_ticks'])
        if old and old['state'] != 'Z':
            if old['state'] not in ['T', 't']:
                raise RuntimeError('Obsolete queue unexpectedly resumed')
            os.kill(controller['pid'], signal.SIGKILL)
        controller['retired'] = True
        changed = True
    if all(c.get('retired') for c in transition['controllers']) and not transition.get('console_resumed'):
        for server in transition.get('servers', [transition['server']]):
            if same_process(server['pid'], server['start_ticks']):
                os.kill(server['pid'], signal.SIGCONT)
        transition.update(console_resumed=True, console_resumed_at=runner.now())
        changed = True
    if changed:
        runner.write_json(ROOT / '.local/dynamic-transition.json', transition)


def launch(task, port, gpu):
    label = f"{task['order']:02d}-{task['competition']}"
    try:
        options = {'training_gpu': gpu}
        if task.get('retry_of_order') is not None:
            options['recovery_source'] = ROOT / 'runs' / f"{task['retry_of_order']:02d}-{task['competition']}"
        return runner.execute(label, task['competition'], port, **options)
    except Exception as exc:
        path = state_path(task)
        state = json.loads(path.read_text()) if path.exists() else {}
        pid = state.get('pid')
        live = process(pid) if pid else None
        if live and live['state'] != 'Z':
            raise RuntimeError('Keep this endpoint occupied: its task child is still alive') from exc
        path.parent.mkdir(parents=True, exist_ok=True)
        state.update(status='execution_failed', finished_at=runner.now(),
                     runner_error_type=type(exc).__name__)
        runner.write_json(path, state)
        return False


def gpu_allocation(task, active):
    pool = ROOT / 'training-resources.json'
    if pool.exists() and json.loads(pool.read_text()).get('mode') == 'local_gpu_pool':
        return ''  # The training worker leases GPUs per job, rather than per conversation.
    if any(job.get('training_gpu') == '0' for job in active.values()):
        return ''
    name = task['competition'].lower()
    return '0' if any(word in name for word in ['soil', 'misconception', 'ultrasound',
                                               'filament', 'low-light', 'enveda', 'buildings']) else ''


def main():
    if (ROOT / '.local/EXPERIMENTS_PAUSED.json').exists():
        raise SystemExit('Experiments paused by user; review the saved checkpoint before restarting')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--resume-transition', type=Path)
    args = parser.parse_args()
    lock_name = (f'dynamic-{args.resume_transition.stem}.lock'
                 if args.resume_transition else 'dynamic.lock')
    with (ROOT / '.local' / lock_name).open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan_path = ROOT / 'plan.json'
        plan = json.loads(plan_path.read_text())
        followup_path = ROOT / 'followup-plan.json'
        followup = json.loads(followup_path.read_text())
        initial, later = plan['tasks'], followup['tasks']
        tasks = initial + later
        transition = (json.loads(args.resume_transition.read_text())
                      if args.resume_transition else hold_old_queues(initial))
        runner.write_json(ROOT / '.local/dynamic-transition.json', transition)
        active = adopt_active(transition)
        plan.update(status='initial_six_in_progress', scheduling='available_qwen_endpoints',
                    max_concurrent_runs=MAX_CONCURRENT_RUNS, tasks_per_endpoint=TASKS_PER_ENDPOINT,
                    scheduler_pid=os.getpid(), routes=None,
                    inference_gpu_by_task={str(job['task']['order']): job['port'] - 12800
                                           for job in active.values()})
        followup.update(status='queued_with_initial_tasks', scheduling='available_qwen_endpoints',
                        start_after='An available task slot on a Qwen endpoint; select by batch then order without completion barriers',
                        max_concurrent_runs=MAX_CONCURRENT_RUNS,
                        tasks_per_endpoint=TASKS_PER_ENDPOINT, routes=None)
        for task in later:
            task['qwen_inference_gpu'] = None  # Assigned from live endpoint availability.
        runner.write_json(plan_path, plan)
        runner.write_json(followup_path, followup)
        with ThreadPoolExecutor(max_workers=MAX_CONCURRENT_RUNS) as pool:
            while True:
                update_adopted(transition)
                followup, later = read_followup(followup_path, initial, later)
                tasks = initial + later
                for order, job in list(active.items()):
                    if 'adopted' in job:
                        if job['adopted']['done']:
                            del active[order]
                    elif job['future'].done():
                        job['future'].result()
                        del active[order]
                current = states(tasks)
                initial_done = all(current[t['order']]['status'] in TERMINAL for t in initial)
                if initial_done and plan['status'] != 'initial_six_finished':
                    plan['status'] = 'initial_six_finished'
                    runner.write_json(plan_path, plan)
                if not active and all(current[t['order']]['status'] in TERMINAL for t in tasks):
                    followup.update(status='finished_review_results', finished_at=runner.now())
                    runner.write_json(followup_path, followup)
                    return
                active_orders = {job['task']['order'] for job in active.values()}
                waiting = list(candidates(initial, later if followup.get('enabled') else [], current,
                                          reserved=active_orders))
                for port in PORTS:
                    slot = free_slot(active, port)
                    if not waiting or slot is None or not endpoint_available(port):
                        continue
                    task = waiting.pop(0)
                    if task in later and followup['status'] != 'running':
                        followup.update(status='running', started_at=runner.now())
                        runner.write_json(followup_path, followup)
                    gpu = gpu_allocation(task, active)
                    future = pool.submit(launch, task, port, gpu)
                    active[task['order']] = {'future': future, 'task': task, 'port': port,
                                             'slot': slot, 'training_gpu': gpu or None}
                    plan['inference_gpu_by_task'][str(task['order'])] = port - 12800
                    runner.write_json(plan_path, plan)
                summary = {'updated_at': runner.now(), 'scheduler_pid': os.getpid(),
                           'mode': 'available_qwen_endpoints', 'initial_six_ended': initial_done,
                           'batch_barrier': False, 'eeg_dependency_barrier': False,
                           'earliest_unfinished_batch': current_batch(tasks, current),
                           'queue_priority': ['batch', 'order'],
                           'max_concurrent_runs': MAX_CONCURRENT_RUNS,
                           'tasks_per_endpoint': TASKS_PER_ENDPOINT,
                           'task_count': len(tasks),
                           'queued_orders': [t['order'] for t in candidates(initial, later, current,
                                                                            reserved=set(active))],
                           'active': [{'order': job['task']['order'], 'gpu': job['port'] - 12800,
                                       'slot': job['slot'], 'training_gpu': job.get('training_gpu')}
                                      for job in active.values()]}
                runner.write_json(ROOT / '.local/dynamic-status.json', summary)
                time.sleep(5)


if __name__ == '__main__':
    main()
