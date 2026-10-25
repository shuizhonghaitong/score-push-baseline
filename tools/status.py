#!/usr/bin/env python3
"""Show task execution and the recorded results without contacting Kaggle."""
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def gpu_training_observation():
    """Observe host GPU processes; an inference endpoint is not a child of a run."""
    try:
        inventory = subprocess.run(['nvidia-smi', '--query-gpu=index,uuid', '--format=csv,noheader,nounits'],
                                   capture_output=True, text=True, timeout=5, check=True)
        indices = {line.split(',')[1].strip(): int(line.split(',')[0])
                   for line in inventory.stdout.splitlines() if ',' in line}
        apps = subprocess.run(['nvidia-smi', '--query-compute-apps=gpu_uuid,pid,used_memory',
                               '--format=csv,noheader,nounits'], capture_output=True,
                              text=True, timeout=5, check=True)
        result = []
        for line in apps.stdout.splitlines():
            parts = [part.strip() for part in line.split(',')]
            if len(parts) != 3 or parts[0] not in indices:
                continue
            pid = int(parts[1])
            ancestors, current = set(), pid
            while current > 1 and current not in ancestors:
                ancestors.add(current)
                try:
                    fields = Path(f'/proc/{current}/stat').read_text().rsplit(')', 1)[1].split()
                    current = int(fields[1])
                except (OSError, ValueError):
                    break
            result.append({'gpu': indices[parts[0]], 'pid': pid,
                           'memory_mib': parts[2], 'ancestors': ancestors})
        return result
    except (OSError, ValueError, subprocess.SubprocessError):
        return None


def snapshot():
    plan = json.loads((ROOT / 'plan.json').read_text())
    followup_path = ROOT / 'followup-plan.json'
    followup = json.loads(followup_path.read_text()) if followup_path.exists() else {}
    tasks = plan['tasks'] + followup.get('tasks', [])
    planned_gpus = dict(plan.get('inference_gpu_by_task', {}))
    planned_gpus.update({str(task['order']): task.get('qwen_inference_gpu')
                         for task in followup.get('tasks', [])})
    runs = []
    observed = gpu_training_observation()
    labels = [('00-titanic-smoke-v2', 'Titanic preflight')]
    labels += [(f"{t['order']:02d}-{t['competition']}", t['title']) for t in tasks]
    for label, title in labels:
        directory = ROOT / 'runs' / label
        state = {'status': 'queued'}
        if (directory / 'status.json').exists():
            state = json.loads((directory / 'status.json').read_text())
        row = {'title': title, 'directory': str(directory), 'execution': state}
        order = directory.name[:2]
        planned_gpu = planned_gpus.get(str(int(order)))
        port = state.get('port')
        training_jobs = []
        for path in directory.glob('gpu-jobs/*/status.json'):
            try:
                job = json.loads(path.read_text())
            except (OSError, ValueError):
                continue
            if job.get('status') == 'running':
                training_jobs.append(job)
        training_pids = {job.get('pid') for job in training_jobs}
        row['resources'] = {
            'qwen_inference_gpu': port - 12800 if port else planned_gpu,
            'qwen_endpoint': f'http://127.0.0.1:{port}/v1' if port else None,
            'local_training_gpu_allocation': state.get('training_gpu', '0' if order == '01' else None),
            'local_training_gpu_pool': state.get('local_training_gpu_pool'),
            'local_training_jobs': training_jobs,
            'local_gpu_observation_available': observed is not None,
            'observed_local_gpu_jobs': [
                {k: job[k] for k in ['gpu', 'pid', 'memory_mib']}
                for job in (observed or []) if state.get('pid') in job['ancestors']
                or training_pids.intersection(job['ancestors'])
            ],
        }
        result_path = directory / 'RESULT.json'
        if result_path.exists():
            try:
                row['result'] = json.loads(result_path.read_text())
            except ValueError:
                row['result_read_error'] = 'incomplete JSON'
        if state['status'] == 'finished' and (not isinstance(row.get('result'), dict)
                                              or not row['result']):
            # A successful CLI exit only proves that the agent turn ended.
            # Preserve the raw run state while reporting missing deliverables.
            state['process_status'] = state['status']
            state['status'] = 'execution_incomplete'
            row['failure_summary'] = '模型进程已退出，但没有有效 RESULT.json；任务尚未完成。'
        events = directory / 'events.jsonl'
        if events.exists():
            with events.open('rb') as stream:
                size = stream.seek(0, 2)
                stream.seek(max(0, size - 131072))
                tail = stream.read().decode('utf-8', errors='replace')
            for line in reversed(tail.splitlines()):
                try:
                    event = json.loads(line)
                except ValueError:
                    continue
                if event.get('type') == 'turn.failed':
                    row['failure_summary'] = (event.get('error') or {}).get('message')
                item = event.get('item', {})
                if item.get('type') == 'agent_message':
                    row['latest_message'] = item.get('text', '').strip()[:1000]
                    break
        runs.append(row)
    return {'runs': runs, 'competition_count': len({t['competition'] for t in tasks}),
            'attempt_count': len(tasks)}


if __name__ == '__main__':
    print(json.dumps(snapshot(), ensure_ascii=False, indent=2))
