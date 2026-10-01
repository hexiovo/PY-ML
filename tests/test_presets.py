from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path
import threading
import unittest
import warnings
from unittest import mock

from pyml_workbench.config import DatasetConfig, ExperimentConfig, SplitConfig
import pyml_workbench.presets as presets
from pyml_workbench.presets import PresetError, load_preset, save_preset


def _config(*, source: str = "missing-input.csv", output_dir: str | None = None) -> ExperimentConfig:
    return ExperimentConfig(
        dataset=DatasetConfig(
            source_path=source,
            target_column="目标",
            feature_columns=("温度", "pressure"),
        ),
        task="regression",
        model_id="R01",
        parameters={"max_iter": 20, "nested": {"enabled": True, "values": [1, 2.5, None]}},
        split=SplitConfig(seed=13),
        output_dir=output_dir,
    )


def _digest(config: dict[str, object]) -> str:
    encoded = json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _write_wrapper(path: Path, config: dict[str, object], **extra: object) -> None:
    wrapper: dict[str, object] = {"schema": 1, "config": config, "sha256": _digest(config)}
    wrapper.update(extra)
    path.write_text(json.dumps(wrapper, ensure_ascii=False, allow_nan=False), encoding="utf-8")


class PresetTests(unittest.TestCase):
    def test_public_import_has_no_optional_gui_model_or_side_effects(self):
        with tempfile.TemporaryDirectory() as temp:
            env = os.environ.copy()
            env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
            script = r"""
import contextlib, io, sys, threading, warnings
from pathlib import Path

hooks_before = (sys.excepthook, threading.excepthook, warnings.showwarning)
captured = io.StringIO()
with contextlib.redirect_stdout(captured):
    import pyml_workbench.presets
hooks_after = (sys.excepthook, threading.excepthook, warnings.showwarning)
optional_prefixes = ("torch", "matplotlib", "PySide6", "PyQt5", "PyQt6")
loaded_optional = [name for name in sys.modules if name.startswith(optional_prefixes)]
assert all(before is after for before, after in zip(hooks_before, hooks_after)), "import changed a global hook"
assert not loaded_optional, f"import loaded optional GUI/model dependencies: {loaded_optional}"
assert "pyml_workbench.gui" not in sys.modules, "import loaded the GUI module"
assert captured.getvalue() == "", f"import wrote to stdout: {captured.getvalue()!r}"
assert not list(Path.cwd().iterdir()), "import wrote files to the working directory"
"""
            result = subprocess.run(
                [sys.executable, "-B", "-c", script],
                check=False,
                capture_output=True,
                text=True,
                encoding="utf-8",
                cwd=temp,
                env=env,
                timeout=30,
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout, "")

    def test_roundtrip_preserves_legacy_canonical_shape_and_hash(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "experiment.json"
            config = _config()
            expected = config.to_dict()
            self.assertNotIn("sequence", expected)
            digest = save_preset(path, config, name="实验一")
            saved = json.loads(path.read_text(encoding="utf-8"))
            restored = load_preset(path)

        self.assertEqual(saved["schema"], 1)
        self.assertEqual(saved["name"], "实验一")
        self.assertEqual(saved["config"], expected)
        self.assertNotIn("sequence", saved["config"])
        self.assertEqual(digest, _digest(expected))
        self.assertEqual(saved["sha256"], digest)
        self.assertEqual(restored, config)
        self.assertEqual(_digest(restored.to_dict()), digest)

    def test_load_does_not_require_or_read_dataset_and_never_runs_experiment(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "missing-data-config.json"
            config = _config(source="does-not-exist.csv")
            save_preset(path, config)
            self.assertFalse(Path("does-not-exist.csv").exists())
            with mock.patch("pyml_workbench.presets._config_type", wraps=presets._config_type):
                restored = load_preset(path)
        self.assertEqual(restored.dataset.source_path, "does-not-exist.csv")

    def test_rejects_duplicate_unknown_nonfinite_and_wrongly_typed_fields(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "bad.json"
            config = _config().to_dict()

            path.write_text('{"schema":1,"schema":1,"config":{},"sha256":"' + "0" * 64 + '"}', encoding="utf-8")
            with self.assertRaisesRegex(PresetError, "duplicate"):
                load_preset(path)

            unknown_wrapper = dict(config=config)
            _write_wrapper(path, config, unexpected=True)
            with self.assertRaisesRegex(PresetError, "unknown"):
                load_preset(path)

            unknown_config = dict(config)
            unknown_config["extra"] = "not allowed"
            _write_wrapper(path, unknown_config)
            with self.assertRaisesRegex(PresetError, "unknown"):
                load_preset(path)

            wrong_type = dict(config)
            wrong_type["model_id"] = 12
            _write_wrapper(path, wrong_type)
            with self.assertRaisesRegex(PresetError, "strings"):
                load_preset(path)

            raw = json.dumps({"schema": 1, "config": config, "sha256": "0" * 64}).replace(
                '"parameters":', '"parameters":{"bad":NaN},"unused_parameters":'
            )
            path.write_text(raw, encoding="utf-8")
            with self.assertRaisesRegex(PresetError, "non-finite"):
                load_preset(path)

    def test_rejects_tampered_hash_and_noncanonical_config(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "tampered.json"
            config = _config().to_dict()
            _write_wrapper(path, config)
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["sha256"] = "f" * 64
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(PresetError, "SHA-256"):
                load_preset(path)

            noncanonical = dict(config)
            noncanonical["sequence"] = None
            _write_wrapper(path, noncanonical)
            with self.assertRaisesRegex(PresetError, "canonical"):
                load_preset(path)

    def test_rejects_oversized_file_and_oversized_save(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "oversized.json"
            path.write_bytes(b" " * (presets.MAX_PRESET_BYTES + 1))
            with self.assertRaisesRegex(PresetError, "1 MiB"):
                load_preset(path)

            too_large = _config()
            too_large.parameters["large"] = "x" * presets.MAX_PRESET_BYTES
            with self.assertRaisesRegex(PresetError, "1 MiB"):
                save_preset(Path(temp) / "too-large.json", too_large)
            self.assertFalse((Path(temp) / "too-large.json").exists())

    def test_atomic_replace_failure_preserves_previous_file(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "preset.json"
            save_preset(path, _config())
            original = path.read_bytes()
            changed = _config()
            changed.parameters["max_iter"] = 99
            with mock.patch.object(presets.os, "replace", side_effect=OSError("simulated replacement failure")):
                with self.assertRaisesRegex(PresetError, "atomically write"):
                    save_preset(path, changed)
            self.assertEqual(path.read_bytes(), original)
            self.assertFalse(list(Path(temp).glob(".preset.json.*.tmp")))

    def test_save_refuses_dataset_and_model_output_collisions(self):
        with tempfile.TemporaryDirectory() as temp:
            source = Path(temp) / "source.csv"
            source.write_bytes(b"keep dataset")
            with self.assertRaisesRegex(PresetError, "dataset source"):
                save_preset(source, _config(source=str(source)))
            self.assertEqual(source.read_bytes(), b"keep dataset")

            output = Path(temp) / "model-output"
            output.mkdir()
            artifact = output / "model.json"
            artifact.write_text("keep model artifact", encoding="utf-8")
            with self.assertRaisesRegex(PresetError, "output directory"):
                save_preset(artifact, _config(output_dir=str(output)))
            self.assertEqual(artifact.read_text(encoding="utf-8"), "keep model artifact")


if __name__ == "__main__":
    unittest.main()
