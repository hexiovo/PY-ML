"""Targeted Qt regressions for final-test binding and exact float parameters."""
import contextlib
import csv
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import threading
import time
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pyml_workbench.gui import WorkbenchWindow
from pyml_workbench.batch_gui import BatchSearchDialog
from PySide6.QtCore import QProcess, QTimer, Qt
from PySide6.QtWidgets import QApplication, QCheckBox, QLineEdit, QSpinBox
from pyml_workbench import (
    DatasetConfig,
    ExperimentConfig,
    ObjectiveSpec,
    SearchSpace,
    SearchSpec,
    SplitConfig,
    _worker as worker,
    create_search_jobs,
    run_search_jobs,
)


class GuiContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.application = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        diagnostic_root = Path(self.temp_dir.name) / "diagnostics"
        environment = patch.dict(os.environ, {"PYML_LOG_ROOT": str(diagnostic_root)})
        environment.start()
        self.addCleanup(environment.stop)
        self.window = WorkbenchWindow(
            preferences_path=Path(self.temp_dir.name) / "preferences.json"
        )
        self.addCleanup(self.window.close)

    def _diagnostic_export_state(self, controller):
        self.window.bound_result = {"preserve": "business result"}
        self.window.current_metrics = {"preserve": 0.75}
        self.window.artifact_paths = {"preserve": "artifact-path"}
        return {
            "bound_result": self.window.bound_result,
            "current_metrics": self.window.current_metrics,
            "artifact_paths": self.window.artifact_paths,
            "preferences": self.window.preferences,
            "stored_preferences": self.window.preference_store.load(),
            "log_root": controller.log_root,
            "session": controller.session,
        }

    def _assert_diagnostic_export_state_unchanged(self, controller, before):
        self.assertIs(self.window.bound_result, before["bound_result"])
        self.assertIs(self.window.current_metrics, before["current_metrics"])
        self.assertIs(self.window.artifact_paths, before["artifact_paths"])
        self.assertEqual(self.window.preferences, before["preferences"])
        self.assertEqual(
            self.window.preference_store.load(), before["stored_preferences"]
        )
        self.assertEqual(controller.log_root, before["log_root"])
        self.assertIs(controller.session, before["session"])

    def _start_diagnostic_controller(self):
        from pyml_workbench.gui import _ApplicationLogController

        log_root = Path(os.environ["PYML_LOG_ROOT"])
        self.window.preferences = self.window.preference_store.set_log_directory(log_root)
        controller = _ApplicationLogController(log_root)
        self.window.logging_controller = controller
        self.addCleanup(controller.close)
        controller.record_event("gui_diagnostics_ready")
        return controller

    def test_diagnostics_menu_exports_current_gui_session_and_preserves_results(self):
        import json
        import zipfile

        controller = self._start_diagnostic_controller()
        before = self._diagnostic_export_state(controller)
        file_menu = next(
            action.menu()
            for action in self.window.menuBar().actions()
            if action.text() == "文件"
        )
        action = self.window.export_diagnostics_action
        self.assertIn(action, file_menu.actions())
        self.assertEqual(action.objectName(), "exportDiagnosticsAction")
        self.assertEqual(action.text(), "导出诊断包…")
        self.assertIn("不会自动上传", action.toolTip())

        destination_without_suffix = Path(self.temp_dir.name) / "gui-session-export"
        expected_path = destination_without_suffix.with_suffix(".zip")
        with (
            patch(
                "pyml_workbench.gui.QInputDialog.getItem",
                return_value=("当前界面会话", True),
            ),
            patch(
                "pyml_workbench.gui.QFileDialog.getSaveFileName",
                return_value=(str(destination_without_suffix), ""),
            ),
            patch("pyml_workbench.gui.QMessageBox.warning") as warning,
        ):
            action.trigger()

        self.assertTrue(expected_path.is_file())
        self.assertGreater(expected_path.stat().st_size, 0)
        with zipfile.ZipFile(expected_path) as package:
            members = set(package.namelist())
            summary = json.loads(package.read("summary.json"))
        self.assertIn("manifest.json", members)
        self.assertIn("tracebacks.json", members)
        self.assertTrue(
            any(name.startswith(f"logs/{controller.session.session_id}/") for name in members)
        )
        self.assertEqual(summary["errors"], [])
        self.assertLessEqual(expected_path.stat().st_size, 20 * 1024 * 1024)
        self.assertIn("诊断包已导出", self.window.result_status_label.text())
        warning.assert_not_called()
        self._assert_diagnostic_export_state_unchanged(controller, before)

    def test_diagnostics_menu_exports_a_pasted_historical_error_id(self):
        import json
        import uuid
        import zipfile

        from pyml_workbench.diagnostics import LogSession

        controller = self._start_diagnostic_controller()
        before = self._diagnostic_export_state(controller)
        old_session = LogSession(
            Path(os.environ["PYML_LOG_ROOT"]),
            session_id=uuid.uuid4().hex,
            role="worker",
        )
        self.addCleanup(old_session.close)
        old_error = old_session.record_external_error(
            "RuntimeError",
            "saved worker error",
            "Traceback (most recent call last):\nRuntimeError: saved worker error",
        )
        old_session.close()
        destination = Path(self.temp_dir.name) / "historical-error.zip"

        with (
            patch(
                "pyml_workbench.gui.QInputDialog.getItem",
                return_value=("按错误编号", True),
            ),
            patch(
                "pyml_workbench.gui.QInputDialog.getText",
                return_value=(old_error.error_id, True),
            ) as get_text,
            patch(
                "pyml_workbench.gui.QFileDialog.getSaveFileName",
                return_value=(str(destination), ""),
            ),
            patch("pyml_workbench.gui.QMessageBox.warning") as warning,
        ):
            self.window.export_diagnostics_action.trigger()

        self.assertRegex(old_error.error_id, r"^[0-9a-f]{32}$")
        get_text.assert_called_once()
        self.assertEqual(get_text.call_args.args[-1], "")
        with zipfile.ZipFile(destination) as package:
            summary = json.loads(package.read("summary.json"))
            tracebacks = json.loads(package.read("tracebacks.json"))
        self.assertEqual([item["error_id"] for item in summary["errors"]], [old_error.error_id])
        self.assertEqual(
            [item["error_id"] for item in tracebacks["errors"]], [old_error.error_id]
        )
        self.assertIn("saved worker error", tracebacks["errors"][0]["traceback"])
        self.assertIn("诊断包已导出", self.window.result_status_label.text())
        warning.assert_not_called()
        self._assert_diagnostic_export_state_unchanged(controller, before)

    def test_diagnostics_menu_cancellation_does_not_create_a_zip(self):
        controller = self._start_diagnostic_controller()
        before = self._diagnostic_export_state(controller)
        destination = Path(self.temp_dir.name) / "cancelled.zip"

        with (
            patch(
                "pyml_workbench.gui.QInputDialog.getItem",
                return_value=("", False),
            ),
            patch(
                "pyml_workbench.gui.QInputDialog.getText"
            ) as get_text,
            patch(
                "pyml_workbench.gui.QFileDialog.getSaveFileName"
            ) as get_save_file,
            patch("pyml_workbench.gui.QMessageBox.warning") as warning,
        ):
            self.window.export_diagnostics_action.trigger()

        self.assertFalse(destination.exists())
        get_text.assert_not_called()
        get_save_file.assert_not_called()
        warning.assert_not_called()
        self._assert_diagnostic_export_state_unchanged(controller, before)

        with (
            patch(
                "pyml_workbench.gui.QInputDialog.getItem",
                return_value=("当前界面会话", True),
            ),
            patch("pyml_workbench.gui.QFileDialog.getSaveFileName", return_value=("", "")) as get_save_file,
            patch("pyml_workbench.gui.QMessageBox.warning") as warning,
        ):
            self.window.export_diagnostics_action.trigger()
        self.assertFalse(destination.exists())
        get_save_file.assert_called_once()
        warning.assert_not_called()
        self._assert_diagnostic_export_state_unchanged(controller, before)

    def test_diagnostics_menu_rejects_invalid_and_missing_error_ids(self):
        controller = self._start_diagnostic_controller()
        before = self._diagnostic_export_state(controller)

        invalid_destination = Path(self.temp_dir.name) / "invalid-error.zip"
        with (
            patch(
                "pyml_workbench.gui.QInputDialog.getItem",
                return_value=("按错误编号", True),
            ),
            patch(
                "pyml_workbench.gui.QInputDialog.getText",
                return_value=("not-an-error-id", True),
            ),
            patch(
                "pyml_workbench.gui.QFileDialog.getSaveFileName"
            ) as get_save_file,
            patch("pyml_workbench.gui.QMessageBox.warning") as warning,
        ):
            self.window.export_diagnostics_action.trigger()
        self.assertFalse(invalid_destination.exists())
        get_save_file.assert_not_called()
        warning.assert_called_once()
        self.assertIn("必须是 32 位十六进制编号", self.window.result_status_label.text())
        self._assert_diagnostic_export_state_unchanged(controller, before)

        missing_destination = Path(self.temp_dir.name) / "missing-error.zip"
        with (
            patch(
                "pyml_workbench.gui.QInputDialog.getItem",
                return_value=("按错误编号", True),
            ),
            patch(
                "pyml_workbench.gui.QInputDialog.getText",
                return_value=("f" * 32, True),
            ),
            patch(
                "pyml_workbench.gui.QFileDialog.getSaveFileName",
                return_value=(str(missing_destination), ""),
            ),
            patch("pyml_workbench.gui.QMessageBox.warning") as warning,
        ):
            self.window.export_diagnostics_action.trigger()
        self.assertFalse(missing_destination.exists())
        warning.assert_called_once()
        self.assertIn("没有找到该 error_id", self.window.result_status_label.text())
        self._assert_diagnostic_export_state_unchanged(controller, before)

    def test_diagnostics_menu_preserves_an_existing_destination(self):
        controller = self._start_diagnostic_controller()
        before = self._diagnostic_export_state(controller)
        destination = Path(self.temp_dir.name) / "keep-existing.zip"
        original = b"existing user file"
        destination.write_bytes(original)

        with (
            patch(
                "pyml_workbench.gui.QInputDialog.getItem",
                return_value=("当前界面会话", True),
            ),
            patch(
                "pyml_workbench.gui.QFileDialog.getSaveFileName",
                return_value=(str(destination), ""),
            ),
            patch("pyml_workbench.gui.QMessageBox.warning") as warning,
        ):
            self.window.export_diagnostics_action.trigger()

        self.assertEqual(destination.read_bytes(), original)
        warning.assert_called_once()
        self.assertIn("为保护原文件", self.window.result_status_label.text())
        self._assert_diagnostic_export_state_unchanged(controller, before)

    def test_finished_signal_drains_final_worker_stdout_before_parsing(self):
        payload = b'{"type":"result","action":"inspect","model_id":"C01"}'

        class BufferedProcess:
            def readAllStandardOutput(self):
                nonlocal payload
                chunk, payload = payload, b""
                return chunk

            def readAllStandardError(self):
                return b""

            @staticmethod
            def state():
                return QProcess.ProcessState.NotRunning

        captured = []
        self.window.process = BufferedProcess()
        self.window._worker_output_buffer = b""
        self.window._worker_action = "inspect"
        self.window._handle_worker_payload = captured.append
        self.window._set_worker_busy = lambda _busy: None
        self.window._cleanup_config_file = lambda: None

        self.window._on_process_finished(0, QProcess.ExitStatus.NormalExit)

        self.assertEqual(captured, [{"type": "result", "action": "inspect", "model_id": "C01"}])

    def test_batch_finished_signal_drains_final_worker_stdout_before_parsing(self):
        payload = b'{"type":"result","phase":"finished","outcomes":[]}'

        class BufferedProcess:
            def readAllStandardOutput(self):
                nonlocal payload
                chunk, payload = payload, b""
                return chunk

            def readAllStandardError(self):
                return b""

            @staticmethod
            def state():
                return QProcess.ProcessState.NotRunning

        dialog = BatchSearchDialog(self.window)
        self.addCleanup(dialog.close)
        captured = []
        dialog.process = BufferedProcess()
        dialog._worker_action = "batch-run"
        dialog._handle_worker_payload = captured.append
        dialog._refresh_controls = lambda: None

        dialog._process_finished(0, QProcess.ExitStatus.NormalExit)

        self.assertEqual(
            captured,
            [{"type": "result", "phase": "finished", "outcomes": []}],
        )

    def _prepare_test_response(self, *, task="clustering", count=0, events=None, status="not_supported", predict=False):
        digest = "a" * 64
        self.window.task_combo.setCurrentIndex(self.window.task_combo.findData(task))
        self.window._frozen = True
        self.window._prepared_signature = digest
        self.window._session_id = "session-1"
        self.window._session_file_path = Path(tempfile.gettempdir(), "session-1.joblib").resolve()
        self.window._session_capabilities = {"predict": predict, "transform": False}
        self.window._render_metrics = lambda: None
        self.window._render_audit = lambda _audit: None
        self.window._append_log = lambda _message: None
        event_list = events if events is not None else [
            "selection_frozen",
            "user_selection_frozen",
            "test_not_evaluated_capability_guard",
        ]
        return {
            "state": "tested",
            "session_id": "session-1",
            "session_path": str(self.window._session_file_path),
            "frozen_config_sha256": digest,
            "capabilities": self.window._session_capabilities,
            "metrics": {"test": {"status": status, "reason": "no fitted predict capability"}},
            "test_evaluation_count": count,
            "audit": {"events": event_list, "test_evaluation_count": count},
        }

    def _run_worker(self, arguments):
        stdout, stderr = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = worker.main(arguments)
        self.assertEqual(code, 0, stderr.getvalue())
        return [json.loads(line) for line in stdout.getvalue().splitlines()]

    def _wait_for_window_worker(self, window, *, timeout=60):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self.application.processEvents()
            if (
                window.process.state() == QProcess.ProcessState.NotRunning
                and window._worker_action is None
            ):
                break
            time.sleep(0.005)
        self.application.processEvents()
        self.assertEqual(window.process.state(), QProcess.ProcessState.NotRunning)
        self.assertIsNone(window._worker_action)
        self.assertIsNone(window.last_error, window.last_error)

    def _wait_for_batch_worker(self, dialog, *, timeout=120):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            self.application.processEvents()
            if (
                dialog.process.state() == QProcess.ProcessState.NotRunning
                and dialog._worker_action is None
            ):
                break
            time.sleep(0.005)
        self.application.processEvents()
        self.assertEqual(dialog.process.state(), QProcess.ProcessState.NotRunning)
        self.assertIsNone(dialog._worker_action)
        self.assertIsNone(dialog._last_error, dialog._last_error)

    def test_header_opens_and_reuses_responsive_nonmodal_batch_dialog(self):
        self.window.batch_search_button.click()
        self.application.processEvents()
        dialog = self.window._batch_dialog
        self.assertIsInstance(dialog, BatchSearchDialog)
        self.assertTrue(dialog.isVisible())
        self.assertFalse(dialog.isModal())

        self.window.open_batch_search()
        self.assertIs(self.window._batch_dialog, dialog)

        dialog.close()
        self.application.processEvents()
        self.assertIsNone(self.window._batch_dialog)

    def test_recent_files_menu_reloads_and_clears_only_saved_paths(self):
        import pandas as pd

        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.csv"
            second = Path(directory) / "second.csv"
            pd.DataFrame({"feature": [1, 2], "target": [0, 1]}).to_csv(first, index=False)
            pd.DataFrame({"feature": [3, 4], "target": [1, 0]}).to_csv(second, index=False)

            self.assertTrue(self.window.load_data(first))
            self.assertTrue(self.window.load_data(second))
            missing = Path(directory) / "missing.csv"
            self.assertFalse(self.window.load_data(missing))
            self.assertEqual(
                [item.path for item in self.window.preferences.recent_files],
                [str(second.resolve()), str(first.resolve())],
            )
            first_action = next(
                action
                for action in self.window.recent_files_menu.actions()
                if action.toolTip() == str(first.resolve())
            )
            first_action.trigger()

            self.assertEqual(self.window.dataset.source_path, str(first.resolve()))
            self.assertTrue(first.is_file())
            self.assertTrue(second.is_file())
            self.assertTrue(self.window.clear_recent_files())
            self.assertEqual(self.window.preferences.recent_files, ())
            self.assertTrue(first.is_file())
            self.assertTrue(second.is_file())

    def test_application_exception_hooks_record_once_and_restore_host_hooks(self):
        from types import SimpleNamespace

        from pyml_workbench.gui import _application_exception_hooks

        calls = []
        host_main = lambda *args: calls.append(("host-main", args))
        host_thread = lambda args: calls.append(("host-thread", args.exc_value))

        class Controller:
            def record_error(self, exc, *, context=None):
                calls.append(("record", str(exc), context))

        original_main, original_thread = sys.excepthook, threading.excepthook
        sys.excepthook = host_main
        threading.excepthook = host_thread
        try:
            with _application_exception_hooks(Controller()):
                owned_main, owned_thread = sys.excepthook, threading.excepthook
                self.assertIsNot(owned_main, host_main)
                self.assertIsNot(owned_thread, host_thread)
                error = ValueError("main failure")
                owned_main(ValueError, error, None)
                thread_error = RuntimeError("thread failure")
                owned_thread(SimpleNamespace(exc_value=thread_error))
                self.assertIs(sys.excepthook, owned_main)
                self.assertIs(threading.excepthook, owned_thread)
            self.assertIs(sys.excepthook, host_main)
            self.assertIs(threading.excepthook, host_thread)
        finally:
            sys.excepthook = original_main
            threading.excepthook = original_thread

        self.assertEqual(
            [entry[:2] for entry in calls if entry[0] == "record"],
            [("record", "main failure"), ("record", "thread failure")],
        )
        self.assertEqual(
            [entry[0] for entry in calls],
            ["record", "host-main", "record", "host-thread"],
        )

    def test_log_directory_persists_across_restart_and_reaches_worker_environment(self):
        class SwitchableController:
            def __init__(self, log_root=None):
                self.log_root = log_root
                self.switches = []

            def switch_directory(self, log_root):
                self.switches.append(log_root)
                self.log_root = log_root

        log_directory = Path(self.temp_dir.name) / "chosen-logs"
        controller = SwitchableController()
        self.window.logging_controller = controller

        self.assertTrue(self.window.set_log_directory(log_directory))
        normalized = str(log_directory.resolve())
        self.assertEqual(controller.log_root, normalized)
        self.assertEqual(self.window.preference_store.load().log_directory, normalized)
        self.assertEqual(
            self.window._worker_process_environment().value("PYML_LOG_ROOT"),
            normalized,
        )

        restarted = WorkbenchWindow(
            preferences_path=self.window.preference_store.path
        )
        self.addCleanup(restarted.close)
        self.assertEqual(restarted.preferences.log_directory, normalized)
        self.assertEqual(
            restarted._worker_process_environment().value("PYML_LOG_ROOT"),
            normalized,
        )

    def test_log_directory_failures_keep_active_logger_and_restore_preference(self):
        from pyml_workbench.preferences import PreferenceError

        class SwitchableController:
            def __init__(self, log_root):
                self.log_root = log_root
                self.switches = []

            def switch_directory(self, log_root):
                self.switches.append(log_root)
                raise OSError("synthetic logger startup failure")

        old_directory = Path(self.temp_dir.name) / "old-logs"
        new_directory = Path(self.temp_dir.name) / "new-logs"
        self.window.preferences = self.window.preference_store.set_log_directory(
            old_directory
        )
        old_normalized = str(old_directory.resolve())
        controller = SwitchableController(old_normalized)
        self.window.logging_controller = controller

        self.assertFalse(self.window.set_log_directory(new_directory))
        self.assertEqual(controller.log_root, old_normalized)
        self.assertEqual(controller.switches, [str(new_directory.resolve())])
        self.assertEqual(self.window.preferences.log_directory, old_normalized)
        self.assertEqual(self.window.preference_store.load().log_directory, old_normalized)
        self.assertIn("仍使用原设置", self.window.result_status_label.text())

        controller.switches.clear()
        with patch.object(
            self.window.preference_store,
            "set_log_directory",
            side_effect=PreferenceError("synthetic preference write failure"),
        ):
            self.assertFalse(self.window.set_log_directory(new_directory))
        self.assertEqual(controller.switches, [])
        self.assertEqual(controller.log_root, old_normalized)
        self.assertEqual(self.window.preference_store.load().log_directory, old_normalized)

    def test_configuration_json_roundtrip_preserves_exact_ui_config_without_running(self):
        import pandas as pd

        source = Path(self.temp_dir.name) / "preset-data.csv"
        pd.DataFrame({
            "feature_b": [0, 1, 0, 1] * 20,
            "feature_a": list(range(80)),
            "target": [0, 1] * 40,
        }).to_csv(source, index=False)
        self.window.task_combo.setCurrentIndex(
            self.window.task_combo.findData("classification")
        )
        self.window.model_combo.setCurrentIndex(self.window.model_combo.findData("C01"))
        self.assertTrue(self.window.load_data(source))
        self.window.target_combo.setCurrentIndex(
            self.window.target_combo.findData("target")
        )
        control, kind = self.window.common_parameter_widgets["max_iter"]
        self.assertEqual(kind, "int")
        control.setValue(137)
        self.window.seed_spin.setValue(17)
        output = Path(self.temp_dir.name) / "results"
        self.window.output_dir_edit.setText(str(output))

        expected = self.window.build_experiment_config()
        preset = Path(self.temp_dir.name) / "configuration.pyml.json"
        self.assertTrue(self.window.save_configuration_json(preset))
        self.window.seed_spin.setValue(42)
        self.window.output_dir_edit.setText(str(Path(self.temp_dir.name) / "other"))
        control.setValue(50)

        self.assertTrue(self.window.load_configuration_json(preset))
        actual = self.window.build_experiment_config()
        self.assertEqual(actual.to_dict(), expected.to_dict())
        self.assertEqual(
            self.window._config_sha256(actual),
            self.window._config_sha256(expected),
        )
        self.assertIsNone(self.window._worker_action)
        with patch.object(self.window, "apply_experiment_config", return_value=False):
            self.assertFalse(self.window.load_configuration_json(preset))

        reordered_features = ExperimentConfig(
            dataset=DatasetConfig(
                source_path=expected.dataset.source_path,
                target_column=expected.dataset.target_column,
                feature_columns=("feature_a", "feature_b"),
            ),
            task=expected.task,
            model_id=expected.model_id,
            parameters=expected.parameters,
            split=expected.split,
            output_dir=expected.output_dir,
        )
        self.assertTrue(self.window.apply_experiment_config(reordered_features))
        self.assertEqual(
            self.window.build_experiment_config().to_dict(),
            reordered_features.to_dict(),
        )
        unspecified_features = ExperimentConfig(
            dataset=DatasetConfig(
                source_path=expected.dataset.source_path,
                target_column=expected.dataset.target_column,
                feature_columns=None,
            ),
            task=expected.task,
            model_id=expected.model_id,
            parameters=expected.parameters,
            split=expected.split,
            output_dir=expected.output_dir,
        )
        self.assertTrue(self.window.apply_experiment_config(unspecified_features))
        self.assertEqual(
            self.window.build_experiment_config().to_dict(),
            unspecified_features.to_dict(),
        )

        # A deliberately absent output directory is part of the config and must
        # survive a save/load cycle; starting a fit still requires an explicit one.
        self.window.output_dir_edit.clear()
        no_output = self.window.build_experiment_config()
        self.assertIsNone(no_output.output_dir)
        no_output_preset = Path(self.temp_dir.name) / "configuration-no-output.json"
        self.assertTrue(self.window.save_configuration_json(no_output_preset))
        self.window.output_dir_edit.setText(str(output))
        self.assertTrue(self.window.load_configuration_json(no_output_preset))
        restored_no_output = self.window.build_experiment_config()
        self.assertEqual(restored_no_output.to_dict(), no_output.to_dict())
        self.assertFalse(self.window.start_training())
        self.assertIsNone(self.window._worker_action)

    def test_missing_source_configuration_keeps_sheet_and_options_and_blocks_training(self):
        missing = Path(self.temp_dir.name) / "missing.xlsx"
        config = ExperimentConfig(
            dataset=DatasetConfig(
                source_path=str(missing),
                sheet_name="Validation",
                target_column="label",
                feature_columns=("x", "z"),
            ),
            task="classification",
            model_id="C01",
            parameters={"max_iter": 91},
            split=SplitConfig(seed=29),
            output_dir=None,
        )

        self.assertTrue(self.window.apply_experiment_config(config))
        self.assertIsNone(self.window.dataset)
        self.assertEqual(self.window.source_path_edit.text(), str(missing))
        self.assertEqual(self.window.sheet_combo.currentData(), "Validation")
        self.assertEqual(self.window.target_combo.currentData(), "label")
        self.assertEqual(self.window.selected_feature_columns(), ["x", "z"])
        self.assertEqual(self.window.seed_spin.value(), 29)
        self.assertEqual(self.window.output_dir_edit.text(), "")
        self.assertIn("数据源不可用", self.window.result_status_label.text())
        self.assertFalse(self.window.start_training())
        self.assertIsNone(self.window._worker_action)

    def test_worker_null_audit_is_allowed_only_for_untested_train_and_freeze(self):
        captured = []
        self.window._apply_train_result = lambda payload: captured.append(("train", payload))
        self.window._apply_freeze_result = lambda payload: captured.append(("freeze", payload))
        self.window._apply_test_result = lambda payload: captured.append(("test", payload))
        for action, state in (("train", "trained"), ("freeze", "frozen")):
            payload = {"type": "result", "action": action, "state": state,
                       "metrics": {}, "capabilities": {}, "audit": None, "test_evaluation_count": 0}
            self.window._handle_worker_payload(payload)
            self.assertEqual(captured[-1][0], action)
            for count in (False, 0.0, 1, -1, "0", None):
                before = len(captured)
                self.window._handle_worker_payload(dict(payload, test_evaluation_count=count))
                self.assertEqual(len(captured), before)
            for field in ("action", "state"):
                before = len(captured)
                self.window._handle_worker_payload(dict(payload, **{field: [action]}))
                self.assertEqual(len(captured), before)
        before = len(captured)
        self.window._handle_worker_payload({"type": "result", "action": "test", "state": "tested",
            "audit": None, "test_evaluation_count": 1})
        self.assertEqual(len(captured), before)

    def test_worker_payload_rejects_non_objects_and_untrusted_error_ids(self):
        messages = []
        self.window._append_log = messages.append

        self.window._handle_worker_payload(["not", "an", "object"])
        self.assertIsNone(self.window.last_error_id)
        self.assertTrue(any("顶层必须是对象" in item for item in messages))

        self.window._handle_worker_payload({
            "type": "error",
            "error_type": "RuntimeError",
            "message": "synthetic worker error",
            "error_id": "g" * 32,
            "traceback": "Traceback (most recent call last):\nRuntimeError",
        })
        self.assertIsNone(self.window.last_error_id)

        dialog = BatchSearchDialog(self.window)
        self.addCleanup(dialog.close)
        dialog._append_log = messages.append
        dialog._handle_worker_payload(["not", "an", "object"])
        self.assertTrue(any("顶层必须是对象" in item for item in messages))
        dialog._handle_worker_payload({
            "type": "error",
            "error_type": "RuntimeError",
            "message": "synthetic batch error",
            "error_id": "g" * 32,
            "traceback": "Traceback (most recent call last):\nRuntimeError",
        })
        self.assertEqual(dialog._error_details[-1]["error_id"], "")
        self.assertNotIn("g" * 32, dialog.log_text.toPlainText())

    def test_batch_worker_progress_rejects_invalid_count_and_duration_values(self):
        dialog = BatchSearchDialog(self.window)
        self.addCleanup(dialog.close)
        messages = []
        dialog._append_log = messages.append

        invalid_progress = (
            {"actual_fit_count": True},
            {"proposal_count": -1},
            {"actual_fit_count": 1},
            {"elapsed_seconds": "not-a-number"},
            {"elapsed_seconds": float("nan")},
            {"elapsed_seconds": float("inf")},
        )
        for data in invalid_progress:
            with self.subTest(data=data):
                before = len(messages)
                dialog._handle_worker_payload({"type": "progress", "data": data})
                self.assertGreater(len(messages), before)
                self.assertIn("无法解析 worker 输出", messages[-1])

    def test_batch_worker_result_rejects_malformed_nested_payloads(self):
        dialog = BatchSearchDialog(self.window)
        self.addCleanup(dialog.close)
        messages = []
        dialog._append_log = messages.append

        dialog._handle_worker_payload({
            "type": "result",
            "phase": "training_exploration_exported",
            "result": {"artifact_paths": "model.joblib"},
        })
        self.assertIn("result.artifact_paths.model", messages[-1])

        dialog._handle_worker_payload({
            "type": "result",
            "phase": "finished",
            "outcomes": [{
                "diagnostic_errors": [{
                    "error_id": "x" * 32,
                    "error_type": "RuntimeError",
                    "message": "bad ID",
                    "traceback": "traceback",
                }],
            }],
        })
        self.assertIn("diagnostic_errors 字段格式无效", messages[-1])

    def test_batch_trial_error_id_reaches_gui_log_and_selected_diagnostic_archive(self):
        import zipfile
        import pandas as pd

        from pyml_workbench.diagnostics import read_records
        from pyml_workbench.diagnostics_archive import export_diagnostics_zip
        from pyml_workbench.gui import _ApplicationLogController

        root = Path(self.temp_dir.name)
        source = root / "batch-error.csv"
        pd.DataFrame({"x": list(range(80)), "target": [0, 1] * 40}).to_csv(
            source, index=False
        )
        config = ExperimentConfig(
            dataset=DatasetConfig(
                source_path=str(source),
                target_column="target",
                feature_columns=("x",),
            ),
            task="classification",
            model_id="C01",
            split=SplitConfig(seed=42),
        )
        spec = SearchSpec(
            method="grid",
            space=SearchSpace.from_dict({
                "fields": {
                    "C": {"type": "real", "low": 0.1, "high": 1.0, "values": [0.1]}
                }
            }),
            max_fits=1,
            max_proposals=1,
            timeout_seconds=60,
        )
        history = root / "history.sqlite3"
        job = create_search_jobs(history, root / "artifacts", [(config, spec)])[0]
        log_root = Path(os.environ["PYML_LOG_ROOT"])
        controller = _ApplicationLogController(log_root)
        self.window.logging_controller = controller
        self.addCleanup(controller.close)
        dialog = BatchSearchDialog(self.window)
        self.addCleanup(dialog.close)
        dialog.root_edit.setText(str(root))

        stdout, stderr = io.StringIO(), io.StringIO()
        with patch(
            "pyml_workbench.search.prepare_experiment",
            side_effect=RuntimeError("synthetic worker trial failure"),
        ), contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            code = worker.main([
                "batch-run", "--history", str(history), "--job-id", job["job_id"],
                "--parallel-workers", "1",
            ])

        self.assertEqual(code, 0, stderr.getvalue())
        worker_events = [json.loads(line) for line in stdout.getvalue().splitlines()]
        finished = next(
            event for event in reversed(worker_events)
            if event.get("type") == "result" and event.get("phase") == "finished"
        )
        details = finished["outcomes"][0]["diagnostic_errors"][0]
        error_id = details["error_id"]
        self.assertRegex(error_id, r"^[0-9a-f]{32}$")
        self.assertIn("synthetic worker trial failure", details["traceback"])

        dialog._handle_worker_payload(finished)
        self.assertTrue(any(item["error_id"] == error_id for item in dialog._error_details))
        dialog._record_external_errors()
        records = read_records(log_root, error_ids=[error_id])
        self.assertGreaterEqual(len(records), 2)
        self.assertEqual({item["role"] for item in records}, {"worker", "gui"})
        self.assertTrue(all(item["traceback"] == details["traceback"] for item in records))

        archive_path = root / "selected-error.zip"
        worker_session = next(item["session_id"] for item in records if item["role"] == "worker")
        export_diagnostics_zip(log_root, archive_path, session_ids=[worker_session])
        with zipfile.ZipFile(archive_path) as archive:
            tracebacks = json.loads(archive.read("tracebacks.json"))
        self.assertEqual([item["error_id"] for item in tracebacks["errors"]], [error_id])

    def test_sequence_json_editor_and_curve_views_preserve_refit_scope(self):
        import numpy as np
        import pandas as pd

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "sequence.csv"
            pd.DataFrame({
                "group": ["a"] * 30,
                "time": list(range(30)),
                "value": np.linspace(0, 1, 30),
            }).to_csv(source, index=False)
            self.assertTrue(self.window.load_data(source))
            self.window.task_combo.setCurrentIndex(self.window.task_combo.findData("sequence_modeling"))
            self.window.model_combo.setCurrentIndex(self.window.model_combo.findData("H01"))
            self.assertFalse(self.window.sequence_config_edit.isHidden())
            self.window.sequence_config_edit.setPlainText(json.dumps({
                "group_column": "group",
                "time_column": "time",
                "order_mode": "time",
                "observation_columns": ["value"],
            }))
            for index in range(self.window.feature_list.count()):
                item = self.window.feature_list.item(index)
                if item.text() in {"group", "time"}:
                    self.assertEqual(item.checkState(), Qt.CheckState.Unchecked)
                    self.assertFalse(item.flags() & Qt.ItemFlag.ItemIsEnabled)
            self.assertEqual(self.window.selected_feature_columns(), ["value"])
            config = self.window.build_experiment_config()
            self.assertEqual(config.sequence.to_dict()["group_column"], "group")
            self.assertEqual(config.sequence.to_dict()["time_column"], "time")
            self.window.sequence_config_edit.setPlainText('{"group_column":')
            with self.assertRaisesRegex(ValueError, "不是有效 JSON"):
                self.window.build_experiment_config()
            self.window.sequence_config_edit.setPlainText("[]")
            with self.assertRaisesRegex(ValueError, "JSON 对象"):
                self.window.build_experiment_config()

        train_validation = {
            "schema_version": 1,
            "model_id": "N06",
            "fit_scope": "train",
            "epochs": [1, 2],
            "train_loss": [1.5, 0.8],
            "validation_loss": [1.7, 0.9],
            "validation_available": True,
        }
        self.window.training_curves = train_validation
        self.window._render_training_curves()
        self.assertEqual(len(self.window.curve_axes.lines), 2)

        refit = {
            "schema_version": 1,
            "model_id": "N06",
            "fit_scope": "train_validation",
            "epochs": [1, 2],
            "train_loss": [1.4, 0.7],
            "validation_loss": None,
            "validation_available": False,
        }
        self.window.training_curves = refit
        self.window._render_training_curves()
        self.assertEqual(len(self.window.curve_axes.lines), 1)
        self.assertIn("未计算验证损失", self.window.curve_status_label.text())

        dialog = BatchSearchDialog()
        self.addCleanup(dialog.close)
        dialog._draw_batch_training_curves(refit, "最终重拟合")
        self.assertEqual(len(dialog.job_curve_axes.lines), 1)
        self.assertIn("未计算验证损失", dialog.job_curve_status.text())

    def test_batch_dialog_runs_json_worker_without_blocking_gui_and_persists_history(self):
        import numpy as np
        import pandas as pd
        from pyml_workbench.history import HistoryStore

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "synthetic.csv"
            pd.DataFrame({
                "x": np.linspace(-2, 2, 48),
                "target": np.tile([0, 1], 24),
            }).to_csv(source, index=False)
            dialog = BatchSearchDialog()
            self.addCleanup(dialog.close)
            dialog.root_edit.setText(str(root / "batch"))
            dialog.datasets_edit.setPlainText(json.dumps([{
                "source_path": str(source),
                "task": "classification",
                "target_column": "target",
                "feature_columns": ["x"],
                "models": ["C01"],
                "seed": 42,
            }]))
            dialog.spaces_edit.setPlainText(json.dumps({
                "C01": {"fields": {}, "fixed": {"max_iter": 80}},
            }))
            dialog.max_fits_spin.setValue(1)
            dialog.max_proposals_spin.setValue(1)
            dialog.minutes_spin.setValue(1)
            ticks = []
            timer = QTimer()
            timer.timeout.connect(lambda: ticks.append(time.monotonic()))
            timer.start(10)

            self.assertTrue(dialog.start_batch())
            deadline = time.monotonic() + 20
            while dialog.process.state() != QProcess.ProcessState.NotRunning and time.monotonic() < deadline:
                self.application.processEvents()
                time.sleep(0.005)
            self.application.processEvents()
            timer.stop()

            self.assertEqual(dialog.process.state(), QProcess.ProcessState.NotRunning)
            self.assertIsNone(dialog._last_error, dialog._last_error)
            self.assertGreater(len(ticks), 0)
            history_path = root / "batch" / "history.sqlite3"
            store = HistoryStore(history_path)
            jobs = store.list_jobs()
            self.assertEqual(len(jobs), 1)
            self.assertEqual(jobs[0]["status"], "completed")
            self.assertEqual(jobs[0]["actual_fit_count"], 1)
            self.assertEqual(jobs[0]["proposal_count"], 1)
            self.assertIn("search_started", dialog.log_text.toPlainText())
            self.assertIn("search_finished", dialog.log_text.toPlainText())
            group_label = dialog.jobs_table.item(0, 1).text()
            self.assertIn("columns=", group_label)
            self.assertIn("objective=balanced_accuracy max", group_label)

    def test_single_workbench_hmm_selection_uses_the_qprocess_final_test_and_export_paths(self):
        import numpy as np
        import pandas as pd

        from pyml_workbench import load_model

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "sequence.csv"
            pd.DataFrame({"value": np.linspace(-2, 2, 90)}).to_csv(source, index=False)
            self.assertTrue(self.window.load_data(source))
            self.window.task_combo.setCurrentIndex(
                self.window.task_combo.findData("sequence_modeling")
            )
            self.window.model_combo.setCurrentIndex(
                self.window.model_combo.findData("H01")
            )
            self.assertEqual(self.window.model_combo.currentData(), "H01")
            self.assertTrue(self.window.deferred_box.isHidden())
            self.assertEqual(self.window.selected_feature_columns(), ["value"])
            self.window.common_parameter_widgets["n_components"][0].setValue(2)
            self.window.common_parameter_widgets["n_iter"][0].setValue(2)
            export_dir = root / "model-output"
            self.window.output_dir_edit.setText(str(export_dir))
            config = self.window.build_experiment_config()
            self.assertEqual(config.task, "sequence_modeling")
            self.assertEqual(config.model_id, "H01")
            self.assertIsNone(config.dataset.target_column)
            self.assertIsNone(config.sequence)  # GUI defaults to one row-ordered series.
            test_responses = []
            train_responses = []
            apply_train_result = self.window._apply_train_result
            apply_test_result = self.window._apply_test_result

            def capture_train_result(payload):
                train_responses.append(dict(payload))
                apply_train_result(payload)

            def capture_test_result(payload):
                test_responses.append(dict(payload))
                apply_test_result(payload)

            self.window._apply_train_result = capture_train_result
            self.window._apply_test_result = capture_test_result
            self.assertTrue(self.window.start_training())
            self._wait_for_window_worker(self.window)
            self.assertEqual(len(train_responses), 1)
            summary = train_responses[0]["sequence_summary"]
            self.assertEqual(summary["totals"]["source_rows"], 90)
            self.assertEqual(summary["totals"]["windows"], 0)
            self.assertIsNone(train_responses[0]["training_curves"])
            summary_text = self.window.sequence_summary_text.toPlainText()
            self.assertIn("实际序列划分", summary_text)
            self.assertIn("总计：源行 90", summary_text)
            self.assertIn("train: 源行 54 (60.0%)", summary_text)
            self.assertIn("validation: 源行 18 (20.0%)", summary_text)
            self.assertIn("test: 源行 18 (20.0%)", summary_text)
            for split_name, part in summary["splits"].items():
                expected = (
                    f"{split_name}: 源行 {part['source_rows']} ({part['source_rows_proportion']:.1%}) · "
                    f"组 {part['groups']} ({part['groups_proportion']:.1%}) · "
                    f"窗口 {part['windows']} (—)"
                )
                self.assertIn(expected, summary_text)
            self.assertTrue(self.window._prepared)
            self.assertFalse(self.window._frozen)
            self.assertNotIn("test", self.window.current_metrics)

            self.window.finalize_button.click()
            self._wait_for_window_worker(self.window)
            self.assertTrue(self.window._frozen)
            self.assertIsNone(self.window._test_result_payload)

            # Testing automatically starts the GUI's export action only after
            # its test response has passed the session/audit checks.
            self.window.finalize_button.click()
            self._wait_for_window_worker(self.window)
            self.assertEqual(len(test_responses), 1)
            tested = test_responses[0]
            self.assertEqual(tested["state"], "tested")
            self.assertEqual(tested["test_evaluation_count"], 1)
            self.assertIn("log_likelihood_per_observation", tested["metrics"]["test"])
            self.assertEqual(self.window.bound_result["metrics"], tested["metrics"])
            self.assertTrue(self.window.artifact_paths)

            model = load_model(self.window.artifact_paths["model"])
            self.assertEqual(model.model_id, "H01")
            prediction = model.predict({
                "observations": np.asarray([[0.1], [0.2], [0.3]], dtype=np.float32),
                "lengths": [3],
            })
            self.assertEqual(len(prediction), 3)
            self.assertTrue(self.window.load_data(source))
            self.assertEqual(self.window.sequence_summary_text.toPlainText(), "")

    def test_reloaded_hmm_and_window_models_infer_csv_through_gui_worker(self):
        import numpy as np
        import pandas as pd

        from pyml_workbench.config import DatasetConfig, ExperimentConfig
        from pyml_workbench.experiment import run_experiment
        from pyml_workbench.sequence import SequenceConfig

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rows = []
            for group in range(15):
                for step in range(12):
                    value = float(group * 12 + step)
                    rows.append({
                        "group": f"g{group:02d}",
                        "time": step,
                        "value": value,
                        "target": float(group * 0.5 + step * 0.25),
                    })
            training = pd.DataFrame(rows)
            training_path = root / "training.csv"
            training.to_csv(training_path, index=False)
            inference_frame = training.drop(columns="target")
            inference_path = root / "inference.csv"
            inference_frame.to_csv(inference_path, index=False)

            for model_id in ("H01", "N04", "N06"):
                with self.subTest(model_id=model_id):
                    is_hmm = model_id == "H01"
                    sequence = (
                        SequenceConfig(
                            group_column="group",
                            time_column="time",
                            order_mode="time",
                            observation_columns=("value",),
                        )
                        if is_hmm
                        else SequenceConfig(
                            group_column="group",
                            time_column="time",
                            order_mode="time",
                            window=3,
                            horizon=1,
                        )
                    )
                    config = ExperimentConfig(
                        dataset=DatasetConfig(
                            source_path=str(training_path),
                            target_column=None if is_hmm else "target",
                            feature_columns=("value",),
                        ),
                        task="sequence_modeling" if is_hmm else "regression",
                        model_id=model_id,
                        parameters=(
                            {"n_components": 2, "n_iter": 2}
                            if is_hmm
                            else {
                                "hidden_size": 4,
                                "batch_size": 16,
                                "max_epochs": 2,
                                "patience": 1,
                                "random_state": 42,
                            }
                        ),
                        output_dir=str(root / f"{model_id}-artifacts"),
                        sequence=sequence,
                    )
                    result = run_experiment(config)
                    model_path = result.artifact_paths["model"]

                    # Both inspect and inference run in real QProcess workers,
                    # each loading the exported model bundle from disk.
                    self.assertTrue(self.window.reload_model(model_path))
                    self._wait_for_window_worker(self.window)
                    self.assertEqual(self.window.loaded_model_id, model_id)
                    self.window.inference_data_edit.setText(str(inference_path))
                    output_path = root / f"{model_id}-inference.csv"
                    self.assertTrue(self.window.start_inference(output_path))
                    self._wait_for_window_worker(self.window)
                    self.assertTrue(output_path.is_file())
                    output = pd.read_csv(output_path)

                    if is_hmm:
                        expected_positions = np.arange(len(inference_frame), dtype=np.int64)
                        self.assertTrue({
                            "source_row_position", "source_index", "group_id", "hidden_state",
                            "posterior_0", "posterior_1",
                        }.issubset(output.columns))
                        self.assertEqual(len(output), len(inference_frame))
                        np.testing.assert_array_equal(output["source_row_position"], expected_positions)
                        np.testing.assert_array_equal(output["group_id"][:12], ["g00"] * 12)
                        np.testing.assert_allclose(
                            output[["posterior_0", "posterior_1"]].sum(axis=1),
                            np.ones(len(output)),
                            rtol=1e-6,
                            atol=1e-6,
                        )
                        self.assertTrue(np.isfinite(output[["posterior_0", "posterior_1"]].to_numpy()).all())
                    else:
                        expected_positions = np.concatenate([
                            np.arange(group * 12 + 3, group * 12 + 12, dtype=np.int64)
                            for group in range(15)
                        ])
                        self.assertTrue({
                            "source_row_position", "source_index", "group_id", "prediction",
                            "window_source_row_positions", "window_source_indexes",
                        }.issubset(output.columns))
                        self.assertEqual(len(output), len(expected_positions))
                        np.testing.assert_array_equal(output["source_row_position"], expected_positions)
                        self.assertTrue(np.isfinite(output["prediction"].to_numpy()).all())
                        source_positions = json.loads(output.iloc[0]["window_source_row_positions"])
                        self.assertEqual(source_positions, [0, 1, 2])
                        self.assertTrue(all(output.iloc[offset]["group_id"] == "g00" for offset in range(9)))
                        self.assertEqual(
                            json.loads(output.iloc[9]["window_source_row_positions"]),
                            [12, 13, 14],
                        )

    def test_batch_gui_qprocess_runs_hmm_and_deep_search_freeze_refit_and_test_once(self):
        import pandas as pd
        import joblib

        from pyml_workbench.batch import finalize_frozen_search, get_job_summary
        from pyml_workbench.history import HistoryStore

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "sequence.csv"
            rows = []
            for group in range(15):
                for step in range(12):
                    value = float(group * 12 + step)
                    rows.append({
                        "group": f"g{group:02d}",
                        "time": step,
                        "value": value,
                        "target": float(group * 0.5 + step * 0.25),
                    })
            pd.DataFrame(rows).to_csv(source, index=False)
            sequence_base = {
                "group_column": "group",
                "time_column": "time",
                "order_mode": "time",
            }
            dataset_requests = [
                {
                    "source_path": str(source),
                    "task": "sequence_modeling",
                    "feature_columns": ["value"],
                    "models": ["H01"],
                    "seed": 42,
                    "objective": {
                        "metric": "log_likelihood_per_observation",
                        "direction": "max",
                        "split": "validation",
                    },
                    "sequence": {**sequence_base, "observation_columns": ["value"]},
                },
                {
                    "source_path": str(source),
                    "task": "regression",
                    "target_column": "target",
                    "feature_columns": ["value"],
                    "models": ["N06"],
                    "seed": 42,
                    "objective": {"metric": "rmse", "direction": "min", "split": "validation"},
                    "sequence": {**sequence_base, "window": 3, "horizon": 1},
                },
            ]
            spaces = {
                "H01": {"fields": {}, "fixed": {"n_components": 2, "n_iter": 2}},
                "N06": {
                    "fields": {},
                    "fixed": {
                        "hidden_size": 4,
                        "batch_size": 16,
                        "max_epochs": 2,
                        "patience": 1,
                        "random_state": 42,
                    },
                },
            }

            dialog = BatchSearchDialog()
            self.addCleanup(dialog.close)
            dialog._show_error = lambda message: setattr(dialog, "_last_error", str(message))
            batch_root = root / "batch"
            dialog.root_edit.setText(str(batch_root))
            dialog.datasets_edit.setPlainText(json.dumps(dataset_requests))
            dialog.spaces_edit.setPlainText(json.dumps(spaces))
            dialog.max_fits_spin.setValue(1)
            dialog.max_proposals_spin.setValue(1)
            dialog.minutes_spin.setValue(2)

            phases = []
            handle_payload = dialog._handle_worker_payload

            def capture_batch_phase(payload):
                if payload.get("type") == "result":
                    phases.append(payload.get("phase"))
                handle_payload(payload)

            dialog._handle_worker_payload = capture_batch_phase
            self.assertTrue(dialog.start_batch(), dialog._last_error)
            self._wait_for_batch_worker(dialog)
            self.assertIn("queued", phases)
            self.assertIn("finished", phases)

            history_path = batch_root / "history.sqlite3"
            store = HistoryStore(history_path)
            jobs = store.list_jobs()
            self.assertEqual({job["model_id"] for job in jobs}, {"H01", "N06"})
            self.assertTrue(all(job["status"] == "completed" for job in jobs))
            self.assertTrue(all(job["actual_fit_count"] == 1 for job in jobs))

            # The completed N06 job should expose the persisted split manifest
            # and the actual search-winner curves in its selected-job panel.
            n06_job = next(item for item in jobs if item["model_id"] == "N06")
            n06_row = next(
                row for row in range(dialog.jobs_table.rowCount())
                if dialog.jobs_table.item(row, 0).text() == n06_job["job_id"]
            )
            dialog.jobs_table.selectRow(n06_row)
            self.application.processEvents()
            details = dialog.job_details_text.toPlainText()
            self.assertIn("序列划分", details)
            self.assertIn("train: 源行 ", details)
            self.assertIn("validation: 源行 ", details)
            self.assertIn("test: 源行 ", details)
            self.assertEqual(len(dialog.job_curve_axes.lines), 2)
            self.assertIn("搜索 winner", dialog.job_curve_status.text())

            for model_id in ("H01", "N06"):
                with self.subTest(model_id=model_id):
                    job = next(item for item in jobs if item["model_id"] == model_id)
                    table_row = next(
                        row for row in range(dialog.jobs_table.rowCount())
                        if dialog.jobs_table.item(row, 0).text() == job["job_id"]
                    )
                    dialog.jobs_table.selectRow(table_row)
                    self.application.processEvents()
                    self.assertTrue(dialog.freeze_button.isEnabled())
                    dialog.freeze_button.click()
                    self._wait_for_batch_worker(dialog)
                    self.assertIn("selection_frozen", phases)

                    frozen_path = Path(job["artifact_dir"]) / "frozen-selection.joblib"
                    selection = joblib.load(frozen_path)
                    self.assertEqual(selection.config.model_id, model_id)
                    if model_id == "N06":
                        self.assertEqual(selection.metadata["extended_provenance"]["selected_epochs"], 2)

                    self.assertTrue(dialog.finalize_button.isEnabled())
                    dialog.finalize_button.click()
                    self._wait_for_batch_worker(dialog)
                    self.assertIn("finalized", phases)

                    summary = get_job_summary(history_path, job["job_id"])
                    permission = summary["test_permission"]
                    self.assertEqual(permission["state"], "completed")
                    test_events = [
                        event["event_type"]
                        for event in summary["events"]
                        if event["event_type"] in {"test_started", "test_completed"}
                    ]
                    self.assertEqual(test_events, ["test_started", "test_completed"])
                    self.assertTrue(summary["artifact_paths"]["model"])

                    cached = finalize_frozen_search(history_path, job["job_id"])
                    self.assertTrue(cached.cached)
                    self.assertEqual(cached.session.fit_scope, "train_validation")
                    self.assertEqual(cached.session.test_evaluation_count, 1)
                    self.assertIn("test", cached.session.metrics)
                    envelope = joblib.load(Path(job["artifact_dir"]) / "final-session.joblib")
                    self.assertEqual(envelope["state"], "tested")
                    self.assertEqual(envelope["session"].test_evaluation_count, 1)
                    if model_id == "N06":
                        self.assertEqual(len(dialog.job_curve_axes.lines), 1)
                        self.assertIn("train+validation 重拟合", dialog.job_curve_status.text())

    def test_capability_guard_can_complete_without_scoring_test_rows(self):
        for task in ("clustering", "dimensionality reduction", "anomaly detection"):
            with self.subTest(task=task):
                response = self._prepare_test_response(task=task)

                self.window._apply_test_result(response)

                self.assertEqual(self.window._test_result_payload, response)
                self.assertEqual(self.window.state_label.text(), "最终测试完成 · 导出中")
                self.assertIsNone(self.window.last_error)
                self.window._test_result_payload = None

        # Exercise the actual K07 worker lifecycle too, so this branch is fed
        # the same tested-session payload and cached export response as the GUI.
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "clusters.csv"
            export_dir = root / "artifacts"
            import numpy as np
            import pandas as pd

            rng = np.random.default_rng(7)
            centers = np.array([[-3.0, -3.0], [-3.0, 3.0], [3.0, -3.0], [3.0, 3.0]])
            frame = pd.DataFrame(
                np.vstack([rng.normal(center, 0.3, size=(24, 2)) for center in centers]),
                columns=["x1", "x2"],
            )
            frame.to_csv(source, index=False)
            config = root / "config.json"
            config.write_text(json.dumps({
                "dataset": {"source_path": str(source), "feature_columns": ["x1", "x2"]},
                "task": "clustering",
                "model_id": "K07",
                "parameters": {"n_clusters": 3},
                "split": {"seed": 42},
                "output_dir": str(export_dir),
            }), encoding="utf-8")
            session = root / "session.joblib"
            train = self._run_worker(["train", "--config", str(config), "--session", str(session)])[-1]
            session_id = train["session_id"]
            common = ["--config", str(config), "--session", str(session), "--session-id", session_id]
            frozen = self._run_worker(["freeze", *common])[-1]
            self.assertEqual(frozen["state"], "frozen")
            tested = self._run_worker(["test", *common])[-1]
            self.assertEqual(tested["state"], "tested")
            self.assertEqual(tested["test_evaluation_count"], 0)
            self.assertEqual(tested["metrics"]["test"]["status"], "not_supported")

            self.window.task_combo.setCurrentIndex(self.window.task_combo.findData("clustering"))
            self.window._frozen = True
            self.window._prepared_signature = tested["frozen_config_sha256"]
            self.window._session_id = session_id
            self.window._session_file_path = Path(tested["session_path"]).resolve()
            self.window._session_capabilities = tested["capabilities"]
            self.window._render_metrics = lambda: None
            self.window._render_audit = lambda _audit: None
            self.window._append_log = lambda _message: None
            self.window._apply_test_result(tested)
            self.assertIsNotNone(self.window._test_result_payload)

            exported = self._run_worker([
                "export", *common, "--output", str(export_dir),
            ])[-1]
            combined = {}
            self.window._apply_training_result = lambda payload: combined.update(payload)
            self.window._discard_session = lambda **_kwargs: None
            self.window._apply_export_result(exported)
            self.assertEqual(combined["metrics"]["test"]["status"], "not_supported")
            self.assertTrue(combined["artifact_paths"])

    def test_capability_guard_cannot_override_bound_session_capability(self):
        response = self._prepare_test_response(task="clustering")
        self.window._session_capabilities = {"predict": True, "transform": False}

        self.window._apply_test_result(response)

        self.assertIsNone(self.window._test_result_payload)
        self.assertEqual(self.window.state_label.text(), "审计校验失败")
        self.assertIn("审计", self.window.last_error)

    def test_test_result_requires_explicit_user_freeze_event(self):
        response = self._prepare_test_response(
            count=1,
            status="ok",
            predict=True,
            events=["selection_frozen", "test_metrics_computed_once"],
        )
        response["metrics"]["test"] = {"accuracy": 0.5}

        self.window._apply_test_result(response)

        self.assertIsNone(self.window._test_result_payload)
        self.assertEqual(self.window.state_label.text(), "审计校验失败")
        self.assertIn("用户冻结", self.window.last_error)

    def test_float_parameter_preserves_small_defaults_and_scientific_input(self):
        control, kind = WorkbenchWindow._parameter_control("alpha", 1e-10)
        self.assertEqual(kind, "float")
        self.assertIsInstance(control, QLineEdit)
        self.assertEqual(control.text(), "1e-10")
        self.window.common_parameter_widgets = {"alpha": (control, kind)}
        self.window.advanced_parameters.setPlainText("{}")

        self.assertEqual(self.window._collect_parameters()["alpha"], 1e-10)
        control.setText("2.5e-12")
        self.assertEqual(self.window._collect_parameters()["alpha"], 2.5e-12)

    def test_float_parameter_rejects_nonfinite_values_and_other_controls_stay_typed(self):
        control, kind = WorkbenchWindow._parameter_control("alpha", 1e-10)
        self.window.common_parameter_widgets = {"alpha": (control, kind)}
        self.window.advanced_parameters.setPlainText("{}")
        control.setText("NaN")
        with self.assertRaisesRegex(ValueError, "实数|有限实数"):
            self.window._collect_parameters()

        boolean, _ = WorkbenchWindow._parameter_control("enabled", True)
        integer, _ = WorkbenchWindow._parameter_control("max_iter", 100)
        enum, _ = WorkbenchWindow._parameter_control("solver", "lbfgs")
        self.assertIsInstance(boolean, QCheckBox)
        self.assertIsInstance(integer, QLineEdit)
        self.assertNotIsInstance(integer, QSpinBox)
        self.assertIsInstance(enum, QLineEdit)

    def test_sparse_configuration_preserves_null_and_unbounded_integers(self):
        import pandas as pd

        source = Path(self.temp_dir.name) / "unbounded-data.csv"
        pd.DataFrame({
            "x": list(range(80)),
            "target": [0, 1] * 40,
        }).to_csv(source, index=False)
        huge_seed = (1 << 63) + 27
        huge_max_iter = 1_500_007
        config = ExperimentConfig(
            dataset=DatasetConfig(
                source_path=str(source),
                target_column="target",
                feature_columns=("x",),
            ),
            task="classification",
            model_id="C01",
            parameters={"max_iter": huge_max_iter, "n_jobs": None},
            split=SplitConfig(seed=huge_seed),
        )

        self.assertTrue(self.window.apply_experiment_config(config))
        self.assertIsInstance(self.window.seed_spin, QLineEdit)
        self.assertEqual(self.window.seed_spin.text(), str(huge_seed))
        self.assertEqual(self.window.seed_spin.value(), huge_seed)
        self.assertEqual(
            self.window.common_parameter_widgets["max_iter"][0].text(),
            str(huge_max_iter),
        )
        self.assertEqual(
            self.window.common_parameter_widgets["n_jobs"][0].text(), "null"
        )
        self.assertEqual(self.window._collect_parameters(), config.parameters)
        self.assertEqual(
            self.window.build_experiment_config().to_dict(), config.to_dict()
        )

    def test_configuration_preflight_failure_preserves_form_and_session(self):
        session_path = Path(self.temp_dir.name) / "existing-session.joblib"
        session_path.write_bytes(b"preserve me")
        self.window.task_combo.setCurrentIndex(
            self.window.task_combo.findData("classification")
        )
        self.window.model_combo.setCurrentIndex(
            self.window.model_combo.findData("C01")
        )
        self.window.seed_spin.setValue(19)
        self.window.source_path_edit.setText("before.csv")
        self.window._session_id = "existing-session"
        self.window._session_file_path = session_path
        self.window._prepared = True
        invalid = ExperimentConfig(
            dataset=DatasetConfig(
                source_path="replacement.csv",
                target_column="target",
                feature_columns=("x",),
            ),
            task="classification",
            model_id="missing-model",
            split=SplitConfig(seed=99),
        )

        self.assertFalse(self.window.apply_experiment_config(invalid))
        self.assertEqual(self.window.task_combo.currentData(), "classification")
        self.assertEqual(self.window.model_combo.currentData(), "C01")
        self.assertEqual(self.window.seed_spin.text(), "19")
        self.assertEqual(self.window.source_path_edit.text(), "before.csv")
        self.assertEqual(self.window._session_id, "existing-session")
        self.assertEqual(self.window._session_file_path, session_path)
        self.assertEqual(session_path.read_bytes(), b"preserve me")

    def test_cancelled_crash_is_logged_as_cancelled_not_as_business_error(self):
        class RecordingController:
            def __init__(self):
                self.cancelled = []
                self.exits = []
                self.errors = []

            def record_cancelled(self, *, context=None):
                self.cancelled.append(context)

            def record_process_exit(self, returncode, *, context=None, error_id=None):
                self.exits.append((returncode, context, error_id))

            def record_external_error(self, *args, **kwargs):
                self.errors.append((args, kwargs))

        controller = RecordingController()
        self.window.logging_controller = controller
        self.window._worker_action = "train"
        self.window._cancel_requested = True
        self.window._worker_output_buffer = json.dumps({
            "type": "error",
            "error_type": "RuntimeError",
            "message": "worker was terminated",
            "error_id": "a" * 32,
            "traceback": "RuntimeError: worker was terminated",
        }).encode("utf-8")

        self.window._on_process_error(QProcess.ProcessError.Crashed)
        self.window._on_process_finished(1, QProcess.ExitStatus.CrashExit)

        self.assertEqual(self.window.state_label.text(), "已取消")
        self.assertIsNone(self.window.last_error)
        self.assertIsNone(self.window.last_error_id)
        self.assertEqual(controller.cancelled, [{"stage": "train", "model": "C01"}])
        self.assertEqual(controller.exits, [])
        self.assertEqual(controller.errors, [])

    def test_batch_comparison_group_separates_data_split_task_columns_and_objective(self):
        base = {
            "job_id": "job-a", "dataset_id": "source-a", "task": "classification",
            "config_json": json.dumps({"dataset": {"sheet_name": "Sheet1", "target_column": "target", "feature_columns": ["x"]}, "split": {"seed": 42}}),
            "snapshot_json": json.dumps({"source_sha256": "source-a", "data_sha256": "data-a", "split_sha256": "split-a", "feature_columns": ["x"]}),
            "split_json": json.dumps({"seed": 42, "split_sha256": "split-a"}),
            "search_json": json.dumps({"objective": {"metric": "balanced_accuracy", "direction": "max", "split": "validation"}}),
        }
        _label, expected, _parts = BatchSearchDialog._comparison_group(base)
        variants = []
        for path, value in (
            (("snapshot_json", "data_sha256"), "data-b"),
            (("snapshot_json", "split_sha256"), "split-b"),
            (("task",), "regression"),
            (("snapshot_json", "feature_columns"), ["x", "z"]),
            (("config_json", "dataset", "target_column"), "other_target"),
            (("search_json", "objective", "metric"), "rmse"),
            (("search_json", "objective", "direction"), "min"),
            (("search_json", "objective", "split"), "train_exploratory"),
        ):
            variant = json.loads(json.dumps(base))
            if path[0] in {"snapshot_json", "config_json", "search_json", "split_json"}:
                value_object = json.loads(variant[path[0]])
                cursor = value_object
                for key in path[1:-1]:
                    cursor = cursor[key]
                cursor[path[-1]] = value
                variant[path[0]] = json.dumps(value_object)
            else:
                variant[path[0]] = value
            variants.append(variant)
        for variant in variants:
            with self.subTest(job=variant["task"], snapshot=variant["snapshot_json"]):
                self.assertNotEqual(BatchSearchDialog._comparison_group(variant)[1], expected)

    def test_training_exploration_job_offers_export_but_not_validation_freeze_or_test(self):
        from pyml_workbench.history import HistoryStore

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            history_path = root / "history.sqlite3"
            store = HistoryStore(history_path)
            job_id = "training-exploration-job"
            store.create_job(
                job_id=job_id,
                dataset_id="a" * 64,
                task="dimensionality reduction",
                model_id="D10",
                snapshot={
                    "source_sha256": "a" * 64,
                    "data_sha256": "b" * 64,
                    "split_sha256": "c" * 64,
                    "feature_columns": ["x1", "x2", "x3"],
                },
                split_summary={"seed": 42, "split_sha256": "c" * 64},
                search_spec={
                    "method": "grid",
                    "objective": {
                        "metric": "trustworthiness",
                        "direction": "max",
                        "split": "train_exploratory",
                    },
                },
                budget={"max_actual_fits": 1, "max_proposals": 1},
                artifact_dir=root / "artifacts" / job_id,
                config={
                    "dataset": {"source_path": str(root / "embedding.csv"), "feature_columns": ["x1", "x2", "x3"]},
                    "task": "dimensionality reduction",
                    "model_id": "D10",
                    "split": {"seed": 42},
                },
            )
            store.record_proposal(
                job_id=job_id,
                trial_id="trial-1",
                parameters={"n_components": 2},
                objective_direction="max",
                max_proposals=1,
                result={"parameters": {"n_components": 2}},
            )
            store.mark_fit_started(job_id=job_id, trial_id="trial-1", max_actual_fits=1)
            store.finish_trial(
                job_id=job_id,
                trial_id="trial-1",
                status="succeeded",
                objective_value=0.75,
                metrics={"trustworthiness": 0.75},
                result={"parameters": {"n_components": 2}},
                duration_seconds=1.0,
            )
            store.set_job_status(job_id, "completed", reason="fit_limit")
            validation_job_id = "validation-without-score"
            store.create_job(
                job_id=validation_job_id,
                dataset_id="a" * 64,
                task="dimensionality reduction",
                model_id="D10",
                snapshot={
                    "source_sha256": "a" * 64,
                    "data_sha256": "b" * 64,
                    "split_sha256": "c" * 64,
                    "feature_columns": ["x1", "x2", "x3"],
                },
                split_summary={"seed": 42, "split_sha256": "c" * 64},
                search_spec={
                    "method": "grid",
                    "objective": {
                        "metric": "trustworthiness",
                        "direction": "max",
                        "split": "validation",
                    },
                },
                budget={"max_actual_fits": 1, "max_proposals": 1},
                artifact_dir=root / "artifacts" / validation_job_id,
                config={
                    "dataset": {"source_path": str(root / "embedding.csv"), "feature_columns": ["x1", "x2", "x3"]},
                    "task": "dimensionality reduction",
                    "model_id": "D10",
                    "split": {"seed": 42},
                },
            )
            store.set_job_status(validation_job_id, "completed", reason="no_valid_trial")

            dialog = BatchSearchDialog()
            self.addCleanup(dialog.close)
            dialog.root_edit.setText(str(root))
            dialog.refresh_history()
            row_by_id = {
                dialog.jobs_table.item(row, 0).text(): row
                for row in range(dialog.jobs_table.rowCount())
            }
            dialog.jobs_table.selectRow(row_by_id[job_id])
            self.application.processEvents()

            self.assertEqual(dialog.jobs_table.item(row_by_id[job_id], 8).text(), "不排名·训练探索")
            self.assertEqual(dialog.jobs_table.item(row_by_id[validation_job_id], 8).text(), "无分数")
            score_item = dialog.jobs_table.item(row_by_id[job_id], 9)
            self.assertEqual(score_item.text(), "训练探索：0.75")
            self.assertIn("训练探索分数", score_item.toolTip())
            self.assertIn("score=train_exploratory", dialog.jobs_table.item(row_by_id[job_id], 1).text())
            self.assertTrue(dialog.export_exploration_button.isEnabled())
            self.assertFalse(dialog.freeze_button.isEnabled())
            self.assertFalse(dialog.finalize_button.isEnabled())

            launched = {}
            dialog._start_worker = lambda arguments, *, action: launched.update(
                arguments=arguments, action=action
            )
            dialog.export_exploration_button.click()
            self.assertEqual(launched["action"], "batch-export-exploration")
            self.assertEqual(
                launched["arguments"],
                [
                    "batch-export-exploration", "--history", str(history_path),
                    "--job-id", job_id, "--output",
                    str(root / "exports" / f"training-exploration-{job_id}"),
                ],
            )

    def test_validation_leaderboard_and_csv_rank_real_scores_by_objective_direction(self):
        from unittest.mock import patch

        import numpy as np
        import pandas as pd
        from pyml_workbench.history import HistoryStore

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "ranking.csv"
            rng = np.random.default_rng(71)
            values = rng.uniform(-1, 1, size=(240, 2))
            frame = pd.DataFrame({
                "x1": values[:, 0],
                "x2": values[:, 1],
                "class_target": (values[:, 0] + 0.25 * values[:, 1] > 0).astype(int),
                "reg_target": values[:, 0] ** 2 + 0.3 * values[:, 1] + rng.normal(0, 0.08, 240),
            })
            frame.to_csv(source, index=False)

            requests = []
            for task, target, model in (
                ("classification", "class_target", "C14"),
                ("regression", "reg_target", "R12"),
            ):
                for depth in (1, 6):
                    config = ExperimentConfig(
                        dataset=DatasetConfig(
                            source_path=str(source),
                            target_column=target,
                            feature_columns=("x1", "x2"),
                        ),
                        task=task,
                        model_id=model,
                        split=SplitConfig(seed=42),
                    )
                    spec = SearchSpec(
                        method="grid",
                        space=SearchSpace.from_dict({"fields": {}, "fixed": {"max_depth": depth}}),
                        objective=ObjectiveSpec(),
                        max_fits=1,
                        max_proposals=1,
                        timeout_seconds=120,
                        seed=42,
                    )
                    requests.append((config, spec))

            history_path = root / "history.sqlite3"
            jobs = create_search_jobs(history_path, root / "artifacts", requests)
            outcomes = run_search_jobs(
                history_path,
                [job["job_id"] for job in jobs],
                max_workers=1,
            )
            self.assertTrue(all(item["status"] == "completed" for item in outcomes), outcomes)
            store = HistoryStore(history_path)
            scores = {
                job["job_id"]: max(
                    trial["objective_value"] for trial in store.list_trials(job["job_id"])
                    if trial["status"] == "succeeded"
                ) if job["task"] == "classification" else min(
                    trial["objective_value"] for trial in store.list_trials(job["job_id"])
                    if trial["status"] == "succeeded"
                )
                for job in store.list_jobs()
            }
            classifier_jobs = [job for job in store.list_jobs() if job["task"] == "classification"]
            regression_jobs = [job for job in store.list_jobs() if job["task"] == "regression"]
            classifier_best = max(classifier_jobs, key=lambda job: scores[job["job_id"]])
            classifier_other = next(job for job in classifier_jobs if job["job_id"] != classifier_best["job_id"])
            regression_best = min(regression_jobs, key=lambda job: scores[job["job_id"]])
            regression_other = next(job for job in regression_jobs if job["job_id"] != regression_best["job_id"])
            self.assertNotEqual(scores[classifier_best["job_id"]], scores[classifier_other["job_id"]])
            self.assertNotEqual(scores[regression_best["job_id"]], scores[regression_other["job_id"]])

            dialog = BatchSearchDialog()
            self.addCleanup(dialog.close)
            dialog.root_edit.setText(str(root))
            dialog.refresh_history()
            row_by_id = {
                dialog.jobs_table.item(row, 0).text(): row
                for row in range(dialog.jobs_table.rowCount())
            }
            self.assertEqual(dialog.jobs_table.item(row_by_id[classifier_best["job_id"]], 8).text(), "1")
            self.assertEqual(dialog.jobs_table.item(row_by_id[classifier_other["job_id"]], 8).text(), "2")
            self.assertEqual(dialog.jobs_table.item(row_by_id[regression_best["job_id"]], 8).text(), "1")
            self.assertEqual(dialog.jobs_table.item(row_by_id[regression_other["job_id"]], 8).text(), "2")
            self.assertIn("验证：", dialog.jobs_table.item(row_by_id[classifier_best["job_id"]], 9).text())
            self.assertIn("验证：", dialog.jobs_table.item(row_by_id[regression_best["job_id"]], 9).text())

            export_path = root / "comparison.csv"
            with patch(
                "pyml_workbench.batch_gui.QFileDialog.getSaveFileName",
                return_value=(str(export_path), "CSV 文件 (*.csv)"),
            ):
                dialog.export_comparison()
            with export_path.open(encoding="utf-8-sig", newline="") as stream:
                exported_rows = {row["job_id"]: row for row in csv.DictReader(stream)}
            self.assertEqual(exported_rows[classifier_best["job_id"]]["group_rank"], "1")
            self.assertEqual(exported_rows[classifier_other["job_id"]]["group_rank"], "2")
            self.assertEqual(exported_rows[regression_best["job_id"]]["group_rank"], "1")
            self.assertEqual(exported_rows[regression_other["job_id"]]["group_rank"], "2")


if __name__ == "__main__":
    unittest.main()
