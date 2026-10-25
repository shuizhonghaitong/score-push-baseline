#!/usr/bin/env python3
"""Run two authorized task routes, adopting the existing Housing job unchanged."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import fcntl
import json
import os
from pathlib import Path
import signal
import time

import run_batch as runner

ROOT = runner.ROOT
ROUTES = {'A': [(1, 12804), (3, 12805), (4, 12804)],
          'B': [(2, 12806), (5, 12807), (6, 12806)]}


def process(pid):
    """PID start time prevents accidentally signalling a reused process id."""
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        return {'pid': pid, 'state': fields[0], 'ppid': int(fields[1]),
                'start_ticks': fields[19], 'wait_status': int(fields[49])}
    except FileNotFoundError:
        return None


def same_process(pid, start_ticks):
    item = process(pid)
    return item if item and item['start_ticks'] == start_ticks else None


def stop_dispatcher(dispatcher, worker):
    """Stop only the obsolete Python queue; keep its Codex child running."""
    parent = process(dispatcher)
    child = process(worker)
    if not parent or not child or child['ppid'] != dispatcher:
        raise RuntimeError('Dispatcher/worker relationship changed; no process was stopped')
    command = Path(f'/proc/{dispatcher}/cmdline').read_bytes().split(b'\0')
    if str(ROOT / 'tools/run_batch.py').encode() not in command or b'six' not in command:
        raise RuntimeError('Not the expected old serial dispatcher')
    # No other formal task may already have been started by the old loop.
    for path in (ROOT / 'runs').glob('[0-9][0-9]-*/status.json'):
        if path.parent.name[:2] not in ['00', '01']:
            raise RuntimeError('Another task already has state; inspect before transition')
    os.kill(dispatcher, signal.SIGSTOP)
    try:
        for _ in range(100):
            stopped = same_process(dispatcher, parent['start_ticks'])
            if stopped and stopped['state'] in ['T', 't']:
                break
            time.sleep(.01)
        else:
            raise RuntimeError('Old dispatcher did not stop')
        child = same_process(worker, child['start_ticks'])
        if not child or child['ppid'] != dispatcher:
            raise RuntimeError('Worker identity changed during transition')
        for path in (ROOT / 'runs').glob('[0-9][0-9]-*/status.json'):
            if path.parent.name[:2] not in ['00', '01']:
                raise RuntimeError('Old queue advanced during transition')
    except Exception:
        if same_process(dispatcher, parent['start_ticks']):
            os.kill(dispatcher, signal.SIGCONT)
        raise
    return {'dispatcher': parent, 'worker': child, 'stopped_at': runner.now(),
            'note': 'Only the obsolete queue is stopped; its existing Codex job continues unchanged.'}


def adopt_housing(transition):
    parent, child = transition['dispatcher'], transition['worker']
    while True:
        active = same_process(child['pid'], child['start_ticks'])
        if not active:
            raise RuntimeError('Adopted worker disappeared before its exit could be observed')
        if active['state'] == 'Z':
            # Its stopped parent cannot reap it yet, so the kernel preserves the
            # real wait status. No model process is interrupted or restarted.
            code = os.waitstatus_to_exitcode(active['wait_status'])
            path = next((ROOT / 'runs').glob('01-*/status.json'))
            state = json.loads(path.read_text())
            state.update(status='finished' if code == 0 else 'execution_failed',
                         exit_code=code, finished_at=runner.now(),
                         result_exists=(path.parent / 'RESULT.json').exists(),
                         dispatcher_adopted=True)
            runner.write_json(path, state)
            old = same_process(parent['pid'], parent['start_ticks'])
            if old:
                if old['state'] not in ['T', 't']:
                    raise RuntimeError('Old dispatcher unexpectedly resumed')
                # Retire it only after its sole Codex job has finished. It can
                # never advance its obsolete serial loop and launch duplicates.
                os.kill(parent['pid'], signal.SIGKILL)
            transition.update(completed_at=runner.now(), worker_exit_code=code,
                              old_dispatcher_retired=True)
            runner.write_json(ROOT / '.local/parallel-transition.json', transition)
            return
        time.sleep(2)


def run_route(name, tasks, transition, execute=runner.execute):
    for order, port in ROUTES[name]:
        task = tasks[order]
        label = f"{order:02d}-{task['competition']}"
        if order == 1:
            adopt_housing(transition)
            continue
        state_path = ROOT / 'runs' / label / 'status.json'
        if state_path.exists():
            raise RuntimeError('Refusing duplicate task: ' + label)
        # Route A is CPU/remote after Housing. Route B uses CPU/remote while
        # Housing might still use GPU0, and may use GPU0 for its later tasks.
        gpu = ''
        housing = next((ROOT / 'runs').glob('01-*/status.json'))
        if name == 'B' and json.loads(housing.read_text()).get('status') != 'running':
            gpu = '0'
        try:
            execute(label, task['competition'], port, training_gpu=gpu)
        except Exception as exc:
            state = json.loads(state_path.read_text()) if state_path.exists() else {}
            state.update(status='execution_failed', finished_at=runner.now(),
                         runner_error_type=type(exc).__name__)
            runner.write_json(state_path, state)


def run_routes(tasks, transition, route=run_route):
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs = [pool.submit(route, name, tasks, transition) for name in ['A', 'B']]
        for job in jobs:
            job.result()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--adopt-dispatcher', type=int, required=True)
    p.add_argument('--adopt-worker', type=int, required=True)
    a = p.parse_args()
    with (ROOT / '.local/parallel.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        transition = stop_dispatcher(a.adopt_dispatcher, a.adopt_worker)
        runner.write_json(ROOT / '.local/parallel-transition.json', transition)
        plan_path = ROOT / 'plan.json'
        plan = json.loads(plan_path.read_text())
        plan.update(status='six_runs_in_progress', max_concurrent_runs=2,
                    routes={name: [order for order, _ in jobs] for name, jobs in ROUTES.items()},
                    scheduler_pid=os.getpid(),
                    inference_gpu_by_task={str(order): port - 12800
                                           for route in ROUTES.values() for order, port in route})
        runner.write_json(plan_path, plan)
        tasks = {task['order']: task for task in plan['tasks']}
        run_routes(tasks, transition)
        plan['status'] = 'six_runs_finished_review_results'
        runner.write_json(plan_path, plan)


if __name__ == '__main__':
    main()
