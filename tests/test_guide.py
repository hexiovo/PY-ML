"""Offline guide contracts for packaged resources and the real Qt button."""
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pyml_workbench
from pyml_workbench.guide import load_overall_guide
from pyml_workbench.gui import WorkbenchWindow
from PySide6.QtCore import QProcess, Qt
from PySide6.QtTest import QSignalSpy, QTest
from PySide6.QtWidgets import QApplication


class OverallGuideTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.application = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.window = WorkbenchWindow(
            preferences_path=Path(self.temporary.name) / "preferences.json"
        )
        self.addCleanup(self.window.close)

    def test_guide_matches_document_and_all_workflows_are_present(self):
        canonical = Path(__file__).resolve().parents[1] / "docs/overall-guide.md"
        self.assertEqual(load_overall_guide(), canonical.read_text(encoding="utf-8"))
        for topic in ("Genetic", "Annealing", "LSTM", "诊断包", "Python API", "最终测试"):
            self.assertIn(topic, load_overall_guide())
        self.assertIn(pyml_workbench.__version__, load_overall_guide())

    def test_header_button_opens_readable_non_modal_guide_without_worker(self):
        self.window.overall_guide_button.click()
        self.application.processEvents()
        dialog = self.window._guide_dialog
        self.assertTrue(dialog.isVisible())
        self.assertFalse(dialog.isModal())
        self.assertEqual(dialog.chapters.count(), 13)
        self.assertIn("PY-ML 工作台整体指南", dialog.browser.toPlainText())
        self.assertIn("error_id", dialog.browser.toPlainText())
        self.assertEqual(self.window.process.state(), QProcess.ProcessState.NotRunning)
        self.assertIsNone(self.window.dataset)
        self.assertIsNone(self.window._worker_action)
        self.assertIsNone(self.window.bound_result)

    def test_chapter_navigation_and_forward_backward_wrapping_search(self):
        self.window.overall_guide_button.click()
        dialog = self.window._guide_dialog
        dialog.chapters.setCurrentRow(6)
        self.assertEqual(dialog.browser.textCursor().block().text(), "7. 批量模型比较与自动调参")
        dialog.search_edit.setText("13. 常见问题与下一步")
        self.assertTrue(dialog.find_text())
        first = dialog.browser.textCursor().selectionStart()
        self.assertEqual(dialog.browser.textCursor().selectedText(), "13. 常见问题与下一步")
        self.assertTrue(dialog.find_text())
        self.assertEqual(dialog.browser.textCursor().selectionStart(), first)
        self.assertTrue(dialog.find_text(backward=True))
        self.assertEqual(dialog.browser.textCursor().selectionStart(), first)
        dialog.search_edit.setText("该词不存在123456789")
        self.assertFalse(dialog.find_text())
        self.assertIn("未找到", dialog.search_status.text())
        dialog.search_edit.clear()
        self.assertFalse(dialog.find_text())

    def test_menu_and_repeat_open_reuse_dialog_and_parent_close_hides_it(self):
        self.window.overall_guide_action.trigger()
        dialog = self.window._guide_dialog
        self.assertEqual(self.window.overall_guide_action.shortcut().toString(), "F1")
        dialog.close()
        self.window.overall_guide_button.click()
        self.assertIs(self.window._guide_dialog, dialog)
        self.assertTrue(dialog.isVisible())
        self.window.close()
        self.assertFalse(dialog.isVisible())

    def test_enter_searches_forward_without_triggering_a_dialog_default_button(self):
        self.window.overall_guide_button.click()
        dialog = self.window._guide_dialog
        dialog.search_edit.setText("PY-ML")
        dialog.search_edit.setFocus()
        self.application.processEvents()
        previous_clicks = QSignalSpy(dialog.previous_button.clicked)
        QTest.keyClick(dialog.search_edit, Qt.Key.Key_Return)
        self.application.processEvents()
        self.assertEqual(previous_clicks.count(), 0)
        self.assertEqual(dialog.browser.textCursor().selectionStart(), 0)
        self.assertEqual(dialog.browser.textCursor().selectedText(), "PY-ML")

    def test_missing_guide_is_visible_error_and_can_be_reopened_after_repair(self):
        with (
            patch("pyml_workbench.guide.load_overall_guide", side_effect=FileNotFoundError("guide missing")),
            patch("pyml_workbench.gui.QMessageBox.warning") as warning,
        ):
            self.window.overall_guide_button.click()
            warning.assert_called_once()
            self.assertIn("完整解压", warning.call_args.args[2])
        self.assertIsNone(self.window._guide_dialog)
        self.assertIsNone(self.window.last_error)
        self.window.overall_guide_button.click()
        self.assertTrue(self.window._guide_dialog.isVisible())


if __name__ == "__main__":
    unittest.main()
