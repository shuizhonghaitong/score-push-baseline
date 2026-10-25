"""Check two-route limits and replacement of a dispatcher without stopping its job."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

import parallel_batch as batch


class ParallelBatchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / '.local').mkdir()
        (self.root / 'runs/01-housing').mkdir(parents=True)
        (self.root / 'runs/01-housing/status.json').write_text(json.dumps({'status': 'finished'}))

    def test_two_routes_preserve_order_and_do_not_exceed_two_jobs(self):
        lock = threading.Lock()
        count, peak, started = 0, 0, []
        tasks = {n: {'competition': f'task{n}'} for n in range(1, 7)}

        def execute(label, competition, port, training_gpu):
            nonlocal count, peak
            with lock:
                count += 1
                peak = max(peak, count)
                started.append(int(label[:2]))
            time.sleep(.03)
            with lock:
                count -= 1

        def route(name, tasks, transition):
            batch.run_route(name, tasks, transition, execute=execute)

        with patch.object(batch, 'ROOT', self.root), patch.object(batch, 'adopt_housing') as adopt:
            batch.run_routes(tasks, {}, route=route)
        self.assertEqual(peak, 2)
        self.assertEqual(sorted(started), [2, 3, 4, 5, 6])
        self.assertLess(started.index(3), started.index(4))
        self.assertLess(started.index(2), started.index(5))
        self.assertLess(started.index(5), started.index(6))
        adopt.assert_called_once()

    def test_existing_task_is_not_started_or_overwritten(self):
        path = self.root / 'runs/02-task2'
        path.mkdir()
        (path / 'status.json').write_text('{"status":"running","preserved":true}')
        tasks = {n: {'competition': f'task{n}'} for n in range(1, 7)}
        with patch.object(batch, 'ROOT', self.root), patch.object(batch.runner, 'execute') as execute:
            with self.assertRaises(RuntimeError):
                batch.run_route('B', tasks, {}, execute=execute)
        execute.assert_not_called()
        self.assertTrue(json.loads((path / 'status.json').read_text())['preserved'])

    def test_dispatcher_transfer_keeps_job_alive_and_prevents_old_queue_advancing(self):
        tools = self.root / 'tools'
        tools.mkdir()
        script = tools / 'run_batch.py'
        ready = self.root / 'ready.json'
        finished = self.root / 'worker-finished'
        duplicate = self.root / 'old-queue-advanced'
        script.write_text('''import subprocess,sys,json
from pathlib import Path
child=subprocess.Popen([sys.executable,'-c',
    "import time,sys;from pathlib import Path;time.sleep(.7);Path(sys.argv[1]).write_text('done');sys.exit(7)",sys.argv[2]])
Path(sys.argv[3]).write_text(json.dumps({'child':child.pid}))
child.wait()
Path(sys.argv[4]).write_text('must not happen')
''')
        parent = subprocess.Popen([sys.executable, str(script), 'six', str(finished), str(ready), str(duplicate)])
        child = None
        try:
            for _ in range(200):
                if ready.exists():
                    child = json.loads(ready.read_text())['child']
                    break
                time.sleep(.005)
            self.assertIsNotNone(child)
            path = self.root / 'runs/01-housing/status.json'
            path.write_text(json.dumps({'status': 'running', 'pid': child}))
            with patch.object(batch, 'ROOT', self.root):
                transition = batch.stop_dispatcher(parent.pid, child)
                self.assertIn(batch.process(parent.pid)['state'], ['T', 't'])
                batch.adopt_housing(transition)
            parent.wait(timeout=5)
            self.assertTrue(finished.exists(), 'The worker must continue while its old queue is stopped')
            self.assertFalse(duplicate.exists())
            state = json.loads(path.read_text())
            self.assertEqual(state['exit_code'], 7)
            self.assertEqual(state['status'], 'execution_failed')
            self.assertTrue(state['dispatcher_adopted'])
        finally:
            if parent.poll() is None:
                parent.kill()
                parent.wait()


if __name__ == '__main__':
    unittest.main()
