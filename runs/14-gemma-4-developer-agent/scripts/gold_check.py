#!/usr/bin/env python3
"""Verify the local grader ceiling: reference patches must pass (~127/129)."""
import asyncio
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

RUN = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RUN / 'scripts'))
import eval_local as EL  # noqa: E402

from swegemma.config import EvalConfig  # noqa: E402
from swegemma.evaluate import Evaluator  # noqa: E402
from swegemma.models import load_tasks  # noqa: E402
from swegemma.sandbox.subprocess import SubprocessManager  # noqa: E402
from adk_eval_core.tracing import SessionTrace  # noqa: E402


class GoldEvaluator(Evaluator):
    async def _run_agent_sandbox(self, task, snapshot_path, task_index, total_tasks, **kwargs):
        return task.patch, None, SessionTrace()


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--num-shards', type=int, default=1)
    ap.add_argument('--results', default=None)
    ap.add_argument('--tasks', default='all')
    args = ap.parse_args()

    tasks = load_tasks(EL.COMP / 'tasks.jsonl')
    if args.tasks != 'all':
        keep = set(args.tasks.split(','))
        tasks = [t for t in tasks if t.instance_id in keep]
    else:
        tasks = [t for i, t in enumerate(tasks) if i % args.num_shards == args.shard]
    results_dir = Path(args.results or RUN / 'results' / f'gold_shard{args.shard}of{args.num_shards}')
    results_dir.mkdir(parents=True, exist_ok=True)

    cfg = EvalConfig(tasks_path=EL.COMP / 'tasks.jsonl', snapshots_dir=EL.COMP / 'snapshots',
                     results_dir=results_dir, submission_dir=RUN / 'agent', models=None,
                     sandbox='subprocess', timeout_seconds=300, verification_timeout_seconds=300,
                     display_mode='quiet')
    ev = GoldEvaluator(cfg)
    ev.sandbox = ev.docker = SubprocessManager(timeout_seconds=300, system_site_packages=False)
    EL.verification.install_test_dependencies = EL.setup_env
    EL.agent_runner.install_test_dependencies = EL.setup_env

    pool = ThreadPoolExecutor(max_workers=1)
    for i, task in enumerate(tasks, 1):
        EL.CURRENT['id'] = task.instance_id
        t0 = time.time()
        try:
            res = pool.submit(asyncio.run, ev.evaluate_task(task=task)).result()
            row = dict(id=task.instance_id, repo=task.repo, resolved=bool(res.resolved),
                       exit_code=res.test_exit_code, seconds=round(time.time() - t0, 1),
                       error=(res.error or '')[:200])
        except Exception as e:  # noqa: BLE001
            row = dict(id=task.instance_id, repo=task.repo, resolved=False, exit_code=-1,
                       seconds=round(time.time() - t0, 1), error=repr(e)[:200])
        with open(results_dir / 'progress.jsonl', 'a') as fh:
            fh.write(json.dumps(row) + '\n')
        print(json.dumps(row), flush=True)
    print('GOLD CHECK DONE', flush=True)


if __name__ == '__main__':
    main()
