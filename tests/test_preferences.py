import json
import os
import subprocess
import sys
import tempfile
import unittest
from hashlib import sha256
from pathlib import Path
from unittest.mock import patch

from pyml_workbench import DatasetConfig, ExperimentConfig, SplitConfig
from pyml_workbench.preferences import (
    MAX_PREFERENCE_BYTES,
    MAX_RECENT_FILES,
    PreferenceError,
    PreferenceStore,
)


def _canonical_config_hash(config: ExperimentConfig) -> str:
    encoded = json.dumps(
        config.to_dict(),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return sha256(encoded.encode("utf-8")).hexdigest()


class PreferenceStoreTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        self.preference_path = self.root / "state" / "preferences.json"
        self.store = PreferenceStore(self.preference_path)

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_missing_file_returns_defaults_without_creating_state(self):
        result = self.store.load()

        self.assertIsNone(result.log_directory)
        self.assertEqual(result.recent_files, ())
        self.assertFalse(self.preference_path.exists())
        self.assertFalse(self.preference_path.parent.exists())

    def test_log_directory_round_trips_as_normalized_absolute_path(self):
        log_directory = self.root / "app logs" / ".." / "logs"

        result = self.store.set_log_directory(log_directory)

        self.assertEqual(result.log_directory, str((self.root / "logs").resolve()))
        self.assertFalse((self.root / "logs").exists())
        self.assertIsNone(self.store.set_log_directory(None).log_directory)

    def test_recent_files_are_normalized_deduplicated_newest_first_and_limited(self):
        files = []
        for index in range(MAX_RECENT_FILES + 2):
            path = self.root / f"source_{index}.csv"
            path.write_text("x\n1\n", encoding="utf-8")
            files.append(path)
            self.store.record_recent_file(path)

        result = self.store.record_recent_file(
            files[-1].parent / "." / files[-1].name
        )

        self.assertEqual(len(result.recent_files), MAX_RECENT_FILES)
        self.assertEqual(result.recent_files[0].path, str(files[-1].resolve()))
        self.assertEqual(
            [item.path for item in result.recent_files],
            [str(path.resolve()) for path in reversed(files[-MAX_RECENT_FILES:])],
        )
        stored = json.loads(self.preference_path.read_text(encoding="utf-8"))
        self.assertEqual(stored["schema_version"], 1)
        self.assertEqual(len(stored["recent_files"]), MAX_RECENT_FILES)

    def test_paths_are_deduplicated_using_platform_normcase(self):
        source = self.root / "MiXeD.csv"
        source.write_text("x\n1\n", encoding="utf-8")
        self.store.record_recent_file(source)
        self.store.record_recent_file(str(source).swapcase())

        result = self.store.load()

        self.assertEqual(
            sum(
                os.path.normcase(item.path) == os.path.normcase(str(source.resolve()))
                for item in result.recent_files
            ),
            1,
        )
        if os.name == "nt":
            self.assertEqual(len(result.recent_files), 1)

    def test_missing_recent_file_is_retained_and_reported(self):
        source = self.root / "temporary.csv"
        source.write_text("x\n1\n", encoding="utf-8")
        self.store.record_recent_file(source)
        source.unlink()

        result = self.store.load()

        self.assertEqual(len(result.recent_files), 1)
        self.assertEqual(result.recent_files[0].path, str(source.resolve()))
        self.assertFalse(result.recent_files[0].exists)
        self.assertEqual(
            json.loads(self.preference_path.read_text(encoding="utf-8"))["recent_files"],
            [str(source.resolve())],
        )

    def test_clearing_history_does_not_delete_source_file_or_log_directory(self):
        source = self.root / "keep.csv"
        source.write_text("x\n1\n", encoding="utf-8")
        log_directory = self.root / "logs"
        self.store.set_log_directory(log_directory)
        self.store.record_recent_file(source)

        result = self.store.clear_recent_files()

        self.assertTrue(source.is_file())
        self.assertEqual(result.recent_files, ())
        self.assertEqual(result.log_directory, str(log_directory.resolve()))

    def test_preference_writes_do_not_change_experiment_config_or_hash(self):
        config = ExperimentConfig(
            dataset=DatasetConfig(
                source_path="input.csv",
                target_column="label",
                feature_columns=("value",),
            ),
            task="classification",
            model_id="C01",
            parameters={"max_iter": 25},
            split=SplitConfig(seed=17),
            output_dir="result",
        )
        before_payload = config.to_dict()
        before_hash = _canonical_config_hash(config)
        source = self.root / "recent.csv"
        source.write_text("x\n1\n", encoding="utf-8")

        self.store.set_log_directory(self.root / "logs")
        self.store.record_recent_file(source)
        self.store.clear_recent_files()

        self.assertEqual(config.to_dict(), before_payload)
        self.assertEqual(_canonical_config_hash(config), before_hash)
        self.assertNotIn("log_directory", config.to_dict())
        self.assertNotIn("recent_files", config.to_dict())

    def test_malformed_unknown_wrong_typed_duplicate_and_nonfinite_json_are_rejected(self):
        invalid_documents = (
            b"{broken",
            b'{"schema_version":1,"log_directory":null,"recent_files":[],"other":true}',
            b'{"schema_version":true,"log_directory":null,"recent_files":[]}',
            b'{"schema_version":1,"log_directory":null,"recent_files":"not-an-array"}',
            b'{"schema_version":1,"log_directory":NaN,"recent_files":[]}',
            b'{"schema_version":1,"log_directory":null,"recent_files":[],"recent_files":[]}',
        )
        for content in invalid_documents:
            with self.subTest(content=content):
                self.preference_path.parent.mkdir(parents=True, exist_ok=True)
                self.preference_path.write_bytes(content)
                before = self.preference_path.read_bytes()

                with self.assertRaises(PreferenceError):
                    self.store.load()

                self.assertEqual(self.preference_path.read_bytes(), before)

    def test_oversized_file_is_rejected_without_rewriting_it(self):
        self.preference_path.parent.mkdir(parents=True, exist_ok=True)
        content = b" " * (MAX_PREFERENCE_BYTES + 1)
        self.preference_path.write_bytes(content)

        with self.assertRaises(PreferenceError):
            self.store.load()

        self.assertEqual(self.preference_path.read_bytes(), content)

    def test_failed_replace_preserves_previous_file_and_cleans_temporary_file(self):
        self.store.set_log_directory(self.root / "old-logs")
        original = self.preference_path.read_bytes()

        with patch("pyml_workbench.preferences.os.replace", side_effect=OSError("denied")):
            with self.assertRaisesRegex(PreferenceError, "Could not write"):
                self.store.set_log_directory(self.root / "new-logs")

        self.assertEqual(self.preference_path.read_bytes(), original)
        self.assertEqual(list(self.preference_path.parent.glob("*.tmp")), [])
        self.assertEqual(
            self.store.load().log_directory, str((self.root / "old-logs").resolve())
        )

    def test_module_import_has_no_optional_imports_or_filesystem_side_effects(self):
        source_root = Path(__file__).resolve().parents[1] / "src"
        isolated_working_directory = self.root / "isolated"
        isolated_working_directory.mkdir()
        script = (
            "import pathlib, sys; "
            "import pyml_workbench.preferences; "
            "blocked = ('PySide6', 'matplotlib', 'torch'); "
            "loaded = [name for name in sys.modules "
            "if any(name == item or name.startswith(item + '.') for item in blocked)]; "
            "assert not loaded, loaded; "
            "assert list(pathlib.Path.cwd().iterdir()) == []"
        )
        environment = os.environ.copy()
        existing_pythonpath = environment.get("PYTHONPATH")
        environment["PYTHONPATH"] = (
            str(source_root)
            if not existing_pythonpath
            else str(source_root) + os.pathsep + existing_pythonpath
        )
        completed = subprocess.run(
            [sys.executable, "-X", "utf8", "-B", "-c", script],
            cwd=isolated_working_directory,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(
            completed.returncode,
            0,
            msg=completed.stderr or completed.stdout,
        )


if __name__ == "__main__":
    unittest.main()
