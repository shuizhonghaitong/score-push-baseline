import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import status


class CompletionReportingTests(unittest.TestCase):
    def test_zero_exit_without_report_is_incomplete_and_raw_evidence_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'plan.json').write_text(json.dumps({'tasks': [
                {'order': 15, 'competition': 'arc', 'title': 'ARC'}]}))
            run = root / 'runs/15-arc'
            run.mkdir(parents=True)
            original = {'status': 'finished', 'exit_code': 0}
            (run / 'status.json').write_text(json.dumps(original))
            with patch.object(status, 'ROOT', root), patch.object(status, 'gpu_training_observation', return_value=None):
                row = status.snapshot()['runs'][1]
                self.assertEqual(row['execution']['status'], 'execution_incomplete')
                self.assertEqual(json.loads((run / 'status.json').read_text()), original)
                (run / 'RESULT.json').write_text('{"status":"completed"}')
                self.assertEqual(status.snapshot()['runs'][1]['execution']['status'], 'finished')
                (run / 'RESULT.json').unlink()
                (run / 'status.json').write_text('{"status":"running"}')
                self.assertEqual(status.snapshot()['runs'][1]['execution']['status'], 'running')


if __name__ == '__main__':
    unittest.main()
