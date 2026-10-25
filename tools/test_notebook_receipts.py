"""Regression checks for the canonical Notebook handle observed in live preflight."""
import contextlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'skills/kaggle/modules/notebooks/scripts'))
sys.path.insert(0, str(ROOT / 'skills/kaggle'))
import notebook_push
import notebook_run


class NotebookReceiptTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.folder = Path(self.temp.name)
        (self.folder / 'kernel-metadata.json').write_text(json.dumps({'id': 'owner/requested'}))

    def push(self, output):
        receipt = {}
        result = subprocess.CompletedProcess([], 0, output, '')
        with patch.object(notebook_push.kaggle_cli, 'installed', return_value=True), \
             patch.object(notebook_push.credentials, 'resolve', return_value=object()), \
             patch.object(notebook_push.kaggle_cli, 'run', return_value=result) as run, \
             patch.object(notebook_push.script, 'state_dir', return_value=self.folder), \
             contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            status = notebook_push.push(self.folder, receipt)
        self.assertEqual(run.call_count, 1)
        return status, receipt

    def test_canonical_handle_and_version_are_saved(self):
        status, receipt = self.push(
            'Your kernel title does not resolve to the specified id.\n'
            'Kernel version 1 successfully pushed.  Please check progress at '
            'https://www.kaggle.com/code/owner/canonical-title\n')
        self.assertEqual(status, 0)
        self.assertEqual(receipt, {'notebook': 'owner/canonical-title', 'version': 1})
        self.assertEqual(json.loads((self.folder / 'notebook-push.json').read_text())['notebook'],
                         'owner/canonical-title')

    def test_missing_or_untrusted_receipt_does_not_claim_success(self):
        for output in ['Push finished',
                       'Kernel version 1 successfully pushed. Please check progress at '
                       'https://kaggle.com.evil.example/code/owner/name']:
            with self.subTest(output=output):
                status, receipt = self.push(output)
                self.assertNotEqual(status, 0)
                self.assertEqual(receipt, {})

    def test_unexpected_owner_does_not_advance(self):
        status, receipt = self.push('Kernel version 2 successfully pushed. Please check progress at '
                                    'https://www.kaggle.com/code/other/canonical-title')
        self.assertNotEqual(status, 0)
        self.assertEqual(receipt, {})

    def test_run_waits_on_actual_receipt_handle(self):
        def push(folder, receipt):
            receipt.update(notebook='owner/canonical-title', version=3)
            return 0
        with patch.object(notebook_run.notebook_push, 'plan', return_value=('owner/requested', [], '')), \
             patch.object(notebook_run.script, 'write_gate', return_value=None), \
             patch.object(notebook_run.credentials, 'load_configured_env_file'), \
             patch.object(notebook_run.notebook_push, 'push', side_effect=push), \
             patch.object(notebook_run.notebook_wait, 'wait_and_fetch', return_value=0) as wait:
            self.assertEqual(notebook_run.main([str(self.folder), '--yes']), 0)
            self.assertEqual(wait.call_args.args[0], 'owner/canonical-title')

    def test_failed_push_never_starts_polling(self):
        with patch.object(notebook_run.notebook_push, 'plan', return_value=('owner/requested', [], '')), \
             patch.object(notebook_run.script, 'write_gate', return_value=None), \
             patch.object(notebook_run.credentials, 'load_configured_env_file'), \
             patch.object(notebook_run.notebook_push, 'push', return_value=4), \
             patch.object(notebook_run.notebook_wait, 'wait_and_fetch') as wait:
            self.assertEqual(notebook_run.main([str(self.folder), '--yes']), 4)
            wait.assert_not_called()


if __name__ == '__main__':
    unittest.main()
