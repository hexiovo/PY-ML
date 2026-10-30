"""Source-aware plugin detection, registry migration, and menu integration."""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

# Import the app package first so pandas/dateutil are initialized before
# PySide6 installs its import hook for six.moves.
from pyml_workbench.gui import WorkbenchWindow

from PySide6.QtWidgets import QApplication, QDialog

from pyml_workbench.plugin_dialog import PluginSettingsDialog
from pyml_workbench.plugins import (
    HYPEREEG_SOURCE,
    PYFNIRS_SOURCE,
    PluginError,
    PluginSettings,
    PluginSettingsStore,
    detect_hypereeg_data_conversion,
    detect_pyfnirs_data_conversion,
)


def _write_plugin(
    directory: Path,
    *,
    source_project: str | None = None,
    include_source: bool = True,
    entry_path: str = "DataConversion.exe",
    create_entry: bool = True,
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    entry = directory / entry_path
    if create_entry:
        entry.parent.mkdir(parents=True, exist_ok=True)
        entry.write_bytes(b"synthetic placeholder; detector must not execute it")
    manifest = {
        "schema_version": 1,
        "id": "data-conversion",
        "name": "Synthetic converter",
        "version": "0.0-test",
        "entrypoint": {"kind": "executable", "path": entry_path},
    }
    if include_source:
        manifest["source_project"] = source_project
    (directory / "pyml-plugin.json").write_text(
        json.dumps(manifest, ensure_ascii=False), encoding="utf-8"
    )
    return entry


def _write_legacy_settings(path: Path, directory: Path | None) -> None:
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "directory": str(directory) if directory else None,
                "python_executable": None,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )


class PluginDetectionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="plugin-detect-")
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_pyfnirs_detector_requires_exact_source_and_existing_in_tree_entry(self) -> None:
        legacy = self.root / "legacy"
        _write_plugin(legacy, include_source=False)
        with self.assertRaises(PluginError):
            detect_pyfnirs_data_conversion(PluginSettings(str(legacy)))

        wrong_source = self.root / "wrong-source"
        _write_plugin(wrong_source, source_project=HYPEREEG_SOURCE)
        with self.assertRaises(PluginError):
            detect_pyfnirs_data_conversion(PluginSettings(str(wrong_source)))

        wrong_case = self.root / "wrong-case"
        _write_plugin(wrong_case, source_project="pyfnirs")
        with self.assertRaises(PluginError):
            detect_pyfnirs_data_conversion(PluginSettings(str(wrong_case)))

        missing_entry = self.root / "missing-entry"
        _write_plugin(
            missing_entry, source_project=PYFNIRS_SOURCE, create_entry=False
        )
        with self.assertRaises(PluginError):
            detect_pyfnirs_data_conversion(PluginSettings(str(missing_entry)))

        outside = self.root / "outside.exe"
        outside.write_bytes(b"synthetic")
        escaped_entry = self.root / "escaped-entry"
        _write_plugin(
            escaped_entry,
            source_project=PYFNIRS_SOURCE,
            entry_path="../outside.exe",
            create_entry=False,
        )
        with self.assertRaises(PluginError):
            detect_pyfnirs_data_conversion(PluginSettings(str(escaped_entry)))

        valid = self.root / "valid"
        valid_entry = _write_plugin(valid, source_project=PYFNIRS_SOURCE)
        detected = detect_pyfnirs_data_conversion(PluginSettings(str(valid)))
        self.assertEqual(Path(detected.program), valid_entry.resolve())
        self.assertEqual(detected.arguments, ())

    def test_hypereeg_accepts_legacy_manifest_but_rejects_other_explicit_source(self) -> None:
        legacy = self.root / "legacy-hypereeg"
        legacy_entry = _write_plugin(legacy, include_source=False)
        detected = detect_hypereeg_data_conversion(PluginSettings(str(legacy)))
        self.assertEqual(Path(detected.program), legacy_entry.resolve())

        wrong_source = self.root / "wrong-hypereeg-source"
        _write_plugin(wrong_source, source_project=PYFNIRS_SOURCE)
        with self.assertRaises(PluginError):
            detect_hypereeg_data_conversion(PluginSettings(str(wrong_source)))


class PluginSettingsStoreTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="plugin-registry-")
        self.root = Path(self.temporary.name)
        self.registry_path = self.root / "plugins.json"
        self.legacy_pyfnirs_path = self.root / "pyfnirs_plugins.json"
        self.store = PluginSettingsStore(
            self.registry_path,
            legacy_paths={PYFNIRS_SOURCE: self.legacy_pyfnirs_path},
        )

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_migrates_both_legacy_files_without_losing_either_source(self) -> None:
        hypereeg_dir = self.root / "old-hypereeg"
        pyfnirs_dir = self.root / "old-pyfnirs"
        _write_legacy_settings(self.registry_path, hypereeg_dir)
        _write_legacy_settings(self.legacy_pyfnirs_path, pyfnirs_dir)

        registry = self.store.load_all()

        self.assertEqual(registry[HYPEREEG_SOURCE].directory, str(hypereeg_dir.resolve()))
        self.assertEqual(registry[PYFNIRS_SOURCE].directory, str(pyfnirs_dir.resolve()))
        stored = json.loads(self.registry_path.read_text(encoding="utf-8"))
        self.assertEqual(stored["schema_version"], 2)
        self.assertEqual(set(stored), {"schema_version", "plugins"})
        self.assertEqual(set(stored["plugins"]), {HYPEREEG_SOURCE, PYFNIRS_SOURCE})
        self.assertTrue(self.legacy_pyfnirs_path.exists())

        reloaded = PluginSettingsStore(
            self.registry_path,
            legacy_paths={PYFNIRS_SOURCE: self.legacy_pyfnirs_path},
        ).load_all()
        self.assertEqual(reloaded, registry)

    def test_saving_one_source_preserves_other_entries(self) -> None:
        hypereeg_dir = self.root / "hypereeg"
        pyfnirs_dir = self.root / "pyfnirs"
        self.store.save(PluginSettings(str(hypereeg_dir)), HYPEREEG_SOURCE)
        self.store.save(PluginSettings(str(pyfnirs_dir)), PYFNIRS_SOURCE)
        self.store.save(
            PluginSettings(str(self.root / "hypereeg-updated")), HYPEREEG_SOURCE
        )

        registry = self.store.load_all()

        self.assertEqual(
            registry[HYPEREEG_SOURCE].directory,
            str((self.root / "hypereeg-updated").resolve()),
        )
        self.assertEqual(
            registry[PYFNIRS_SOURCE].directory,
            str(pyfnirs_dir.resolve()),
        )
        self.store.remove(HYPEREEG_SOURCE)
        self.assertNotIn(HYPEREEG_SOURCE, self.store.load_all())
        self.assertIn(PYFNIRS_SOURCE, self.store.load_all())


class PluginMenuIntegrationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.application = QApplication.instance() or QApplication(["plugin-integration"])

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="plugin-menu-")
        self.root = Path(self.temporary.name)
        self.preferences_path = self.root / "preferences.json"
        self.window = WorkbenchWindow(
            preferences_path=self.preferences_path,
            checkpoint_directory=self.root / "checkpoints",
        )

    def tearDown(self) -> None:
        self.window.close()
        self.application.processEvents()
        self.temporary.cleanup()

    def test_plugin_conversion_is_a_submenu_sharing_both_source_actions(self) -> None:
        menu = next(action.menu() for action in self.window.menuBar().actions()
                    if action.text() == "插件")
        conversion = menu.actions()[0]
        self.assertEqual(conversion.text(), "数据转换")
        self.assertIsNotNone(conversion.menu())
        self.assertTrue(conversion.isEnabled())
        self.assertEqual(conversion.menu().actions(), [
            self.window.hypereeg_conversion_action,
            self.window.pyfnirs_conversion_action,
        ])
        self.assertIs(self.window.data_conversion_action, conversion)

    def test_submenu_actions_launch_only_the_selected_source(self) -> None:
        hypereeg = self.root / "hypereeg"
        pyfnirs = self.root / "pyfnirs"
        _write_plugin(hypereeg, include_source=False)
        _write_plugin(pyfnirs, source_project=PYFNIRS_SOURCE)
        self.window.plugin_settings_store.save(PluginSettings(str(hypereeg)), HYPEREEG_SOURCE)
        self.window.plugin_settings_store.save(PluginSettings(str(pyfnirs)), PYFNIRS_SOURCE)
        menu = next(action.menu() for action in self.window.menuBar().actions()
                    if action.text() == "插件")
        menu.aboutToShow.emit()
        with patch("pyml_workbench.gui.QProcess.startDetached", return_value=(True, 1)) as launch:
            for action, directory in zip(menu.actions()[0].menu().actions(), (hypereeg, pyfnirs)):
                action.trigger()
                launch.assert_called_with(str(directory / "DataConversion.exe"), [], str(directory))
            self.assertEqual(launch.call_count, 2)

    def test_both_actions_start_disabled_and_unified_dialog_saves_both_immediately(self) -> None:
        self.assertEqual(self.window.format_menu.title(), "格式")
        self.assertEqual(self.window.data_conversion_menu.title(), "数据转换")
        self.assertEqual(
            self.window.plugin_settings_store.path,
            self.preferences_path.with_name("plugins.json"),
        )
        self.assertFalse(self.window.hypereeg_conversion_action.isEnabled())
        self.assertFalse(self.window.pyfnirs_conversion_action.isEnabled())
        self.assertIn("插件 → 设置", self.window.hypereeg_conversion_action.toolTip())
        self.assertIn("插件 → 设置", self.window.pyfnirs_conversion_action.toolTip())

        hypereeg_dir = self.root / "hypereeg-release"
        _write_plugin(hypereeg_dir, include_source=False)
        pyfnirs_dir = self.root / "pyfnirs-release"
        _write_plugin(pyfnirs_dir, source_project=PYFNIRS_SOURCE)

        def save_both(dialog: PluginSettingsDialog):
            dialog.directory_edits[HYPEREEG_SOURCE].setText(str(hypereeg_dir))
            dialog.directory_edits[PYFNIRS_SOURCE].setText(str(pyfnirs_dir))
            dialog.save_settings()
            return QDialog.DialogCode.Accepted

        with patch.object(PluginSettingsDialog, "exec", new=save_both):
            self.window.show_plugin_settings()

        self.assertTrue(self.window.hypereeg_conversion_action.isEnabled())
        self.assertTrue(self.window.pyfnirs_conversion_action.isEnabled())
        registry = self.window.plugin_settings_store.load_all()
        self.assertEqual(
            registry[HYPEREEG_SOURCE].directory, str(hypereeg_dir.resolve())
        )
        self.assertEqual(
            registry[PYFNIRS_SOURCE].directory, str(pyfnirs_dir.resolve())
        )
        stored = json.loads(
            self.window.plugin_settings_store.path.read_text(encoding="utf-8")
        )
        self.assertEqual(stored["schema_version"], 2)

        restarted = WorkbenchWindow(
            preferences_path=self.preferences_path,
            checkpoint_directory=self.root / "checkpoints-restarted",
        )
        try:
            self.assertTrue(restarted.hypereeg_conversion_action.isEnabled())
            self.assertTrue(restarted.pyfnirs_conversion_action.isEnabled())
            (pyfnirs_dir / "DataConversion.exe").unlink()
            restarted.refresh_pyfnirs_plugin()
            restarted.refresh_plugins()
            self.assertTrue(restarted.hypereeg_conversion_action.isEnabled())
            self.assertFalse(restarted.pyfnirs_conversion_action.isEnabled())
        finally:
            restarted.close()

    def test_legacy_configs_migrate_and_enable_only_the_matching_actions(self) -> None:
        hypereeg_dir = self.root / "legacy-hypereeg-release"
        _write_plugin(hypereeg_dir, include_source=False)
        pyfnirs_dir = self.root / "legacy-pyfnirs-release"
        _write_plugin(pyfnirs_dir, source_project=PYFNIRS_SOURCE)
        _write_legacy_settings(self.preferences_path.with_name("plugins.json"), hypereeg_dir)
        _write_legacy_settings(
            self.preferences_path.with_name("pyfnirs_plugins.json"), pyfnirs_dir
        )

        migrated = WorkbenchWindow(
            preferences_path=self.preferences_path,
            checkpoint_directory=self.root / "checkpoints-migrated",
        )
        try:
            self.assertTrue(migrated.hypereeg_conversion_action.isEnabled())
            self.assertTrue(migrated.pyfnirs_conversion_action.isEnabled())
            stored = json.loads(
                migrated.plugin_settings_store.path.read_text(encoding="utf-8")
            )
            self.assertEqual(stored["schema_version"], 2)
            self.assertEqual(
                set(stored["plugins"]), {HYPEREEG_SOURCE, PYFNIRS_SOURCE}
            )
        finally:
            migrated.close()

    def test_invalid_pyfnirs_source_stays_disabled_without_affecting_hypereeg(self) -> None:
        hypereeg_dir = self.root / "valid-hypereeg"
        _write_plugin(hypereeg_dir, include_source=False)
        wrong_source_dir = self.root / "wrong-pyfnirs"
        _write_plugin(wrong_source_dir, source_project=HYPEREEG_SOURCE)
        self.window.plugin_settings_store.save(
            PluginSettings(str(hypereeg_dir)), HYPEREEG_SOURCE
        )
        self.window.plugin_settings_store.save(
            PluginSettings(str(wrong_source_dir)), PYFNIRS_SOURCE
        )

        self.window.refresh_plugins()
        self.window.refresh_pyfnirs_plugin()

        self.assertTrue(self.window.hypereeg_conversion_action.isEnabled())
        self.assertFalse(self.window.pyfnirs_conversion_action.isEnabled())
        self.assertIn("来源标识不匹配", self.window.pyfnirs_conversion_action.toolTip())

    def test_dialog_contains_both_sources_on_one_page(self) -> None:
        dialog = PluginSettingsDialog(self.window.plugin_settings_store, self.window)
        try:
            self.assertEqual(dialog.windowTitle(), "插件设置")
            self.assertIn(HYPEREEG_SOURCE, dialog.directory_edits)
            self.assertIn(PYFNIRS_SOURCE, dialog.directory_edits)
            self.assertIn("同时管理 HyperEEG 和 PYfNIRs", dialog.explanation_label.text())
        finally:
            dialog.close()

    def test_clearing_one_source_keeps_the_other_source(self) -> None:
        hypereeg_dir = self.root / "hypereeg-to-clear"
        pyfnirs_dir = self.root / "pyfnirs-to-keep"
        self.window.plugin_settings_store.save(
            PluginSettings(str(hypereeg_dir)), HYPEREEG_SOURCE
        )
        self.window.plugin_settings_store.save(
            PluginSettings(str(pyfnirs_dir)), PYFNIRS_SOURCE
        )
        dialog = PluginSettingsDialog(self.window.plugin_settings_store, self.window)
        try:
            dialog.clear_source(HYPEREEG_SOURCE)
            dialog.save_settings()
        finally:
            dialog.close()

        registry = self.window.plugin_settings_store.load_all()
        self.assertNotIn(HYPEREEG_SOURCE, registry)
        self.assertEqual(registry[PYFNIRS_SOURCE].directory, str(pyfnirs_dir.resolve()))

    def test_cancel_does_not_persist_unsaved_settings(self) -> None:
        original_dir = self.root / "original-hypereeg"
        replacement_dir = self.root / "unsaved-hypereeg"
        self.window.plugin_settings_store.save(
            PluginSettings(str(original_dir)), HYPEREEG_SOURCE
        )
        dialog = PluginSettingsDialog(self.window.plugin_settings_store, self.window)
        try:
            dialog.directory_edits[HYPEREEG_SOURCE].setText(str(replacement_dir))
            dialog.reject()
        finally:
            dialog.close()

        self.assertEqual(
            self.window.plugin_settings_store.load(HYPEREEG_SOURCE).directory,
            str(original_dir.resolve()),
        )


if __name__ == "__main__":
    unittest.main()
