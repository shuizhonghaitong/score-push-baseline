import json
from pathlib import Path
import tempfile
import unittest

from reproduce.cli import restore, verify
from reproduce.common import sha256, write_json


class PreservationTests(unittest.TestCase):
    def test_verification_detects_source_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "original.py"
            source.write_text("print('original')\n")
            write_json(root / "reproduce/source-manifest.json", {"source_files": [
                {"path": "original.py", "sha256": sha256(source)}]})
            self.assertTrue(verify(root)["passed"])
            source.write_text("changed")
            self.assertEqual(verify(root)["failures"], ["original.py"])

    def test_restore_validates_all_inputs_before_any_copy(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root, source = base / "snapshot", base / "storage"
            source.mkdir()
            (source / "one.csv").write_text("expected")
            write_json(root / "reproduce/required-inputs.json", {"inputs": [
                {"task": "housing", "path": "one.csv", "sha256": sha256(source / "one.csv"), "usage": "replay"},
                {"task": "housing", "path": "missing.csv", "sha256": "0" * 64, "usage": "replay"}]})
            with self.assertRaisesRegex(ValueError, "Missing or mismatched"):
                restore(source, "housing", root)
            self.assertFalse((root / "one.csv").exists())

    def test_restore_rejects_path_traversal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            write_json(root / "reproduce/required-inputs.json", {"inputs": [
                {"task": "housing", "path": "../escape.csv", "sha256": "0" * 64, "usage": "replay"}]})
            with self.assertRaisesRegex(ValueError, "Unsafe relative"):
                restore(root, "housing", root)

    def test_restore_never_overwrites_a_different_existing_file(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root, source = base / "snapshot", base / "storage"
            source.mkdir()
            (source / "train.csv").write_text("reference")
            write_json(root / "reproduce/required-inputs.json", {"inputs": [
                {"task": "housing", "path": "train.csv", "sha256": sha256(source / "train.csv"), "usage": "replay"}]})
            (root / "train.csv").write_text("my-data")
            with self.assertRaisesRegex(ValueError, "Refusing to replace"):
                restore(source, "housing", root)
            self.assertEqual((root / "train.csv").read_text(), "my-data")

    def test_replay_does_not_require_the_training_only_dataset(self):
        with tempfile.TemporaryDirectory() as directory:
            base = Path(directory)
            root, source = base / "snapshot", base / "storage"
            source.mkdir()
            (source / "train.csv").write_text("reference")
            write_json(root / "reproduce/required-inputs.json", {"inputs": [
                {"task": "housing", "path": "train.csv", "sha256": sha256(source / "train.csv"), "usage": "replay"},
                {"task": "housing", "path": "external.csv", "sha256": "0" * 64, "usage": "base_model_training_only"}]})
            self.assertEqual(restore(source, "housing", root)["restored_inputs"], 1)
            self.assertTrue((root / "train.csv").is_file())


if __name__ == "__main__":
    unittest.main()
