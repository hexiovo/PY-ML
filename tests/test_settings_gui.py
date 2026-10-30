"""Qt settings file workflows with real preset validation and atomic writes."""
import importlib.util
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from pyml_workbench.config import DatasetConfig, ExperimentConfig
from pyml_workbench.presets import load_preset, save_preset


@unittest.skipUnless(importlib.util.find_spec("PySide6"), "desktop extra required")
class SettingsGuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from pyml_workbench.gui import QApplication, WorkbenchWindow

        cls.app = QApplication.instance() or QApplication([])
        cls.window_type = WorkbenchWindow

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="pyml-settings-")
        self.root = Path(self.temp.name)
        data = self.root / "合成数据.csv"
        data.write_text("x,target\n1,2\n2,4\n3,6\n4,8\n5,10\n", encoding="utf-8")
        self.config = ExperimentConfig(
            dataset=DatasetConfig(str(data), target_column="target", feature_columns=("x",)),
            task="regression", model_id="R02", output_dir=str(self.root / "results"),
        )
        self.source = self.root / "设置.json"
        save_preset(self.source, self.config)
        self.window = self.window_type(
            preferences_path=self.root / "preferences.json",
            checkpoint_directory=self.root / "checkpoints",
        )

    def tearDown(self):
        self.window.close()
        self.window.deleteLater()
        self.app.processEvents()
        self.temp.cleanup()

    def test_import_edit_save_and_export_copy(self):
        self.assertTrue(self.window.load_configuration_json(self.source))
        self.assertEqual(self.window._configuration_path, self.source)
        self.window.seed_spin.setValue(123)
        self.assertTrue(self.window.save_configuration_changes())
        self.assertEqual(load_preset(self.source).split.seed, 123)
        self.assertEqual(self.window.process.state().name, "NotRunning")
        self.window.seed_spin.setValue(456)
        copy = self.root / "副本.json"
        self.assertTrue(self.window.save_configuration_json(copy))
        self.assertEqual(load_preset(copy).split.seed, 456)
        self.assertEqual(load_preset(self.source).split.seed, 123)
        self.assertEqual(self.window._configuration_path, self.source)
        self.assertTrue(self.window.save_configuration_changes())
        self.assertEqual(load_preset(self.source).split.seed, 456)

    def test_first_save_asks_once_and_appends_extension(self):
        self.assertTrue(self.window.apply_experiment_config(self.config))
        with patch("pyml_workbench.gui.QFileDialog.getSaveFileName",
                   return_value=(str(self.root / "新设置"), "")) as dialog:
            self.window.save_changes_action.trigger()
            dialog.assert_called_once()
        destination = self.root / "新设置.json"
        self.assertEqual(self.window._configuration_path, destination)
        self.window.seed_spin.setValue(77)
        with patch("pyml_workbench.gui.QFileDialog.getSaveFileName") as dialog:
            self.assertTrue(self.window.save_configuration_changes())
            dialog.assert_not_called()
        self.assertEqual(load_preset(destination).split.seed, 77)

    def test_cancel_does_not_create_or_bind_file(self):
        self.assertTrue(self.window.apply_experiment_config(self.config))
        with patch("pyml_workbench.gui.QFileDialog.getSaveFileName", return_value=("", "")):
            self.assertFalse(self.window.save_configuration_changes())
        self.assertIsNone(self.window._configuration_path)

    def test_invalid_import_keeps_current_file_and_form(self):
        self.assertTrue(self.window.load_configuration_json(self.source))
        self.window.seed_spin.setValue(99)
        invalid = self.root / "invalid.json"
        invalid.write_text('{"schema": 999}', encoding="utf-8")
        self.assertFalse(self.window.load_configuration_json(invalid))
        self.assertEqual(self.window._configuration_path, self.source)
        self.assertEqual(self.window.seed_spin.value(), 99)

    def test_failed_save_keeps_original_and_binding(self):
        self.assertTrue(self.window.load_configuration_json(self.source))
        original = self.source.read_bytes()
        self.window.seed_spin.setValue(99)
        with patch("pyml_workbench.presets.os.replace", side_effect=OSError("read only")):
            self.assertFalse(self.window.save_configuration_changes())
        self.assertEqual(self.source.read_bytes(), original)
        self.assertEqual(self.window._configuration_path, self.source)

    def test_actions_follow_existing_editability_rules(self):
        self.assertFalse(self.window.save_changes_action.isEnabled())
        self.assertTrue(self.window.load_config_action.isEnabled())
        self.assertTrue(self.window.load_configuration_json(self.source))
        self.assertTrue(self.window.save_changes_action.isEnabled())
        self.assertTrue(self.window.save_config_action.isEnabled())
        self.window.data_group.setEnabled(False)
        self.window._update_worker_controls()
        self.assertFalse(self.window.save_changes_action.isEnabled())
        self.assertFalse(self.window.load_config_action.isEnabled())


if __name__ == "__main__":
    unittest.main()
