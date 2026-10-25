#!/usr/bin/env python3
"""Incremental gold check: process each task as soon as its snapshot exists."""
import asyncio
import json
import sys
import time
from pathlib import Path

RUN = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RUN / 'scripts'))
import eval_local as EL  # noqa: E402
from swegemma.config import EvalConfig  # noqa: E402
from swegemma.models import load_tasks  # noqa: E402
from swegemma.sandbox.subprocess import SubprocessManager  # noqa: E402
import gold_check as GC  # noqa: E402


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--results', default=str(RUN / 'results' / 'gold_full'))
    ap.add_argument('--timeout', type=float, default=6 * 3600)
    args = ap.parse_args()

    tasks = load_tasks(EL.COMP / 'tasks.jsonl')
    results_dir = Path(args.results)
    results_dir.mkdir(parents=True, exist_ok=True)
    pf = results_dir / 'progress.jsonl'
    done = set()
    if pf.exists():
        for line in pf.read_text().splitlines():
            try:
                done.add(json.loads(line)['id'])
            except Exception:
                pass
    cfg = EvalConfig(tasks_path=EL.COMP / 'tasks.jsonl', snapshots_dir=EL.COMP / 'snapshots',
                     results_dir=results_dir, submission_dir=RUN / 'agent', models=None,
                     sandbox='subprocess', timeout_seconds=300, verification_timeout_seconds=300,
                     display_mode='quiet')
    ev = GC.GoldEvaluator(cfg)
    ev.sandbox = ev.docker = SubprocessManager(timeout_seconds=300, system_site_packages=False)
    EL.verification.install_test_dependencies = EL.setup_env
    EL.agent_runner.install_test_dependencies = EL.setup_env
    pending = [t for t in tasks if t.instance_id not in done]
    t0 = time.time()
    while pending and time.time() - t0 < args.timeout:
        ready = [t for t in pending if (EL.COMP / 'snapshots' / (t.instance_id + '.tgz')).exists()]
        if not ready:
            print('waiting for snapshots, pending=%d' % len(pending), flush=True)
            time.sleep(60)
            continue
        for task in ready:
            if task.instance_id in done or task.instance_id not in [p.instance_id for p in pending]:
                continue
            EL.CURRENT['id'] = task.instance_id
            st = time.time()
            try:
                res = asyncio.run(ev.evaluate_task(task=task))
                row = dict(id=task.instance_id, repo=task.repo, resolved=bool(res.resolved),
                           exit_code=res.test_exit_code, seconds=round(time.time() - st, 1),
                           error=(res.error or '')[:200])
            except Exception as e:  # noqa: BLE001
                row = dict(id=task.instance_id, repo=task.repo, resolved=False, exit_code=-1,
                           seconds=round(time.time() - st, 1), error=repr(e)[:200])
            done.add(task.instance_id)
            with open(pf, 'a') as fh:
                fh.write(json.dumps(row) + '\n')
            print(json.dumps(row), flush=True)
        pending = [t for t in pending if t.instance_id not in done]
    print('GOLD FILL DONE remaining=%d' % len(pending), flush=True)


if __name__ == '__main__':
    main()
