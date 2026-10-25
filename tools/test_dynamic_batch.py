"""Check unrestricted dispatch, endpoint load and uninterrupted adoption."""
import json
from pathlib import Path
import signal
import tempfile
import unittest
from unittest.mock import patch

import dynamic_batch as batch


class DynamicBatchTests(unittest.TestCase):
    def test_batch_priority_never_blocks_later_queued_tasks(self):
        tasks = [{'order': 1, 'batch': 1}, {'order': 2, 'batch': 1},
                 {'order': 3, 'batch': 2}, {'order': 4, 'batch': 2},
                 {'order': 5, 'batch': 3}, {'order': 6, 'batch': 4},
                 {'order': 19, 'batch': 1}]
        current = {t['order']: {'status': 'queued'} for t in tasks}
        current[1]['status'] = 'finished'
        current[2]['status'] = 'execution_failed'
        current[3]['status'] = 'running'  # An existing later-batch task stays live.
        self.assertEqual(list(batch.candidates(tasks, [], current)), [tasks[-1], tasks[3], tasks[4], tasks[5]])
        self.assertEqual(list(batch.candidates(tasks, [], current, reserved={19})), [tasks[3], tasks[4], tasks[5]])
        current[19]['status'] = 'finished'
        self.assertEqual(list(batch.candidates(tasks, [], current)), [tasks[3], tasks[4], tasks[5]])
        current[3]['status'] = current[4]['status'] = 'finished'
        self.assertEqual(list(batch.candidates(tasks, [], current)), [tasks[4], tasks[5]])
        current[5]['status'] = 'auth_blocked'
        self.assertEqual(list(batch.candidates(tasks, [], current)), [tasks[5]])

    def test_queue_extension_is_reloaded_and_existing_identity_preserved(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'followup.json'
            initial = [{'order': 1, 'competition': 'soil'}]
            previous = [{'order': 7, 'competition': 'traffic'}]
            added = {'order': 14, 'competition': 'arc'}
            path.write_text(json.dumps({'tasks': previous + [added]}))
            _, later = batch.read_followup(path, initial, previous)
            self.assertEqual(later[-1], added)
            path.write_text(json.dumps({'tasks': [added]}))
            with self.assertRaisesRegex(ValueError, 'removed or reassigned'):
                batch.read_followup(path, initial, previous)

    def test_duplicate_competition_requires_explicit_recovery_link(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'followup.json'
            initial = [{'order': 1, 'competition': 'soil'}]
            retry = {'order': 19, 'competition': 'soil'}
            path.write_text(json.dumps({'tasks': [retry]}))
            with self.assertRaisesRegex(ValueError, 'explicit recovery'):
                batch.read_followup(path, initial, [])
            retry['retry_of_order'] = 1
            path.write_text(json.dumps({'tasks': [retry]}))
            self.assertEqual(batch.read_followup(path, initial, [])[1], [retry])

    def test_all_queued_tasks_eligible_without_phase_or_eeg_barrier(self):
        initial = [{'order': n} for n in range(1, 7)]
        later = [{'order': n} for n in range(7, 14)]
        states = {n: {'status': 'queued'} for n in range(1, 14)}
        states[1]['status'] = states[2]['status'] = 'running'
        self.assertEqual([t['order'] for t in batch.candidates(initial, later, states)], list(range(3, 14)))
        # Auth/preflight can be in flight before its status file says running.
        self.assertEqual([t['order'] for t in batch.candidates(initial, later, states, reserved={3})], list(range(4, 14)))
        states[3]['status'] = 'finished'
        self.assertIn(4, [t['order'] for t in batch.candidates(initial, later, states)])
        for n in range(1, 7):
            states[n]['status'] = 'finished'
        self.assertEqual([t['order'] for t in batch.candidates(initial, later, states)], list(range(7, 14)))

    def test_disabled_followup_is_not_eligible(self):
        current = {3: {'status': 'running'}, 4: {'status': 'queued'}}
        self.assertEqual(list(batch.candidates([{'order': 3}, {'order': 4}], [], current)), [{'order': 4}])

    def test_endpoint_admission_uses_requests_queue_and_cache_headroom(self):
        class Response:
            def __init__(self, body): self.body = body
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def read(self): return self.body.encode()
        samples = [(0, 0, .0, True), (1, 0, .3, True), (2, 0, .3, True), (3, 0, .3, False),
                   (0, 2, .0, False), (0, 0, .85, False)]
        for running, waiting, cache, available in samples:
            body = (f'vllm:num_requests_running{{engine="0"}} {running}.0\n'
                    f'vllm:num_requests_waiting{{engine="0"}} {waiting}.0\n'
                    f'vllm:kv_cache_usage_perc{{engine="0"}} {cache}\n')
            with patch.object(batch.urllib.request, 'build_opener') as opener:
                opener.return_value.open.return_value = Response(body)
                self.assertEqual(batch.endpoint_available(12805), available)

    def test_three_tasks_share_endpoint_without_overwriting_each_other(self):
        transition = {'adopted': [
            {'task': {'order': n}, 'port': 12804, 'done': False} for n in (3, 19, 7)]}
        active = batch.adopt_active(transition)
        self.assertEqual(set(active), {3, 19, 7})
        self.assertEqual({j['slot'] for j in active.values()}, {1, 2, 3})
        self.assertIsNone(batch.free_slot(active, 12804))
        self.assertEqual(batch.free_slot(active, 12805), 1)
        del active[3]
        self.assertEqual(batch.free_slot(active, 12804), 1)
        transition['adopted'].append({'task': {'order': 8}, 'port': 12804, 'done': False})
        with self.assertRaisesRegex(RuntimeError, 'capacity'):
            batch.adopt_active(transition)

    def test_training_gpu_allocation_is_separate_and_exclusive(self):
        task = {'competition': 'umud-ultrasound'}
        with tempfile.TemporaryDirectory() as directory, patch.object(batch, 'ROOT', Path(directory)):
            self.assertEqual(batch.gpu_allocation(task, {}), '0')
            self.assertEqual(batch.gpu_allocation(task, {12804: {'training_gpu': '0'}}), '')
            self.assertEqual(batch.gpu_allocation({'competition': 'housing'}, {}), '')
            (Path(directory) / 'training-resources.json').write_text('{"mode":"local_gpu_pool"}')
            self.assertEqual(batch.gpu_allocation(task, {}), '')

    def test_console_resumes_only_after_both_old_queues_are_retired(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / '.local').mkdir()
            tasks = [{'order': 1, 'competition': 'housing'}, {'order': 2, 'competition': 'soil'}]
            for task in tasks:
                p = root / 'runs' / f"{task['order']:02d}-{task['competition']}"
                p.mkdir(parents=True)
                (p / 'status.json').write_text('{"status":"running"}')
            def meta(pid, ppid=1):
                return {'pid': pid, 'ppid': ppid, 'start_ticks': str(pid)}
            transition = {'server': meta(10), 'controllers': [meta(20), meta(30)],
                          'servers': [meta(10), meta(11)],
                          'adopted': [{'task': tasks[0], 'worker': meta(21, 20), 'done': False},
                                      {'task': tasks[1], 'worker': meta(31, 30), 'done': False}]}
            process_states = {10: 'T', 11: 'T', 20: 'T', 30: 'T', 21: 'Z', 31: 'S'}
            def same(pid, ticks):
                return {'state': process_states[pid], 'wait_status': 0}
            with patch.object(batch, 'ROOT', root), patch.object(batch, 'same_process', side_effect=same), \
                 patch.object(batch.os, 'kill') as kill:
                batch.update_adopted(transition)
                kill.assert_called_once_with(20, signal.SIGKILL)
                self.assertFalse(transition['adopted'][1]['done'])
                process_states[31] = 'Z'
                batch.update_adopted(transition)
                self.assertIn((30, signal.SIGKILL), [call.args for call in kill.call_args_list])
                self.assertIn((10, signal.SIGCONT), [call.args for call in kill.call_args_list])
                self.assertEqual(kill.call_args.args, (11, signal.SIGCONT))
                self.assertFalse(any(call.args[0] in [21, 31] for call in kill.call_args_list))


if __name__ == '__main__':
    unittest.main()
