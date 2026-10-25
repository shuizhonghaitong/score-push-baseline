import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import local_training as training


class LocalTrainingTests(unittest.TestCase):
    def test_selects_only_pool_devices_with_memory_headroom_and_no_active_lease(self):
        settings = {'candidate_gpus': [2, 3], 'maximum_start_utilization_percent': 20,
                    'reserve_memory_mib': 8192}
        rows = [{'index': n, 'uuid': str(n), 'free_mib': 80000, 'utilization': 0}
                for n in range(8)]
        self.assertEqual(training.select_gpu(rows, settings, set(), 16)['index'], 2)
        self.assertEqual(training.select_gpu(rows, settings, {2}, 16)['index'], 3)
        self.assertIsNone(training.select_gpu(rows, settings, {2, 3}, 16))
        rows[2]['free_mib'] = 20000
        rows[3]['utilization'] = 90
        self.assertIsNone(training.select_gpu(rows, settings, set(), 16))

    def test_script_cannot_escape_run_and_job_is_network_isolated(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            run = root / 'runs/test'
            run.mkdir(parents=True)
            script = run / 'train.py'
            script.write_text('pass')
            request = {'script': str(script), 'python': sys.executable,
                       'memory_gib': 16, 'args': ['--epochs', '2']}
            with patch.object(training, 'ROOT', root):
                self.assertEqual(training.validate_run(run), run)
                command = training.command(run, request, {'index': 2})
                self.assertIn('--unshare-net', command)
                self.assertEqual(command[-2:], ['--epochs', '2'])
                self.assertEqual(command[command.index('--bind') + 1:command.index('--bind') + 3], [str(run), str(run)])
                outside = root / 'outside.py'
                outside.write_text('pass')
                request['script'] = str(outside)
                with self.assertRaises(ValueError):
                    training.validate_request(run, request)

    def test_training_environment_uses_selected_uuid_and_drops_agent_credentials(self):
        with patch.dict(os.environ, {'KAGGLE_API_TOKEN': 'private', 'CODEX_HOME': '/private'}):
            env = training.job_environment({'uuid': 'GPU-test'})
        self.assertEqual(env['CUDA_VISIBLE_DEVICES'], 'GPU-test')
        self.assertNotIn('KAGGLE_API_TOKEN', env)
        self.assertNotIn('CODEX_HOME', env)
        self.assertEqual(env['HF_HUB_OFFLINE'], '1')


if __name__ == '__main__':
    unittest.main()
