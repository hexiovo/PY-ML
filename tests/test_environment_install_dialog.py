"""Exercise the install button and QProcess protocol without a network dependency."""
import json
import os
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
import unittest
from unittest.mock import patch

from pyml_workbench.gui import WorkbenchWindow  # Preserve the core-before-Qt import order.
from pyml_workbench.environment_dialog import EnvironmentCheckDialog
from PySide6.QtCore import QProcess
from PySide6.QtWidgets import QApplication


class FakeProcess:
    def __init__(self, event=None):
        self.output = (json.dumps(event) + '\n').encode() if event else b''
        self.running = False
        self.command = None

    def readAllStandardOutput(self):
        value, self.output = self.output, b''
        return value

    def readAllStandardError(self):
        return b''

    def state(self):
        return QProcess.ProcessState.Running if self.running else QProcess.ProcessState.NotRunning

    def processId(self):
        return 0

    def start(self, executable, arguments):
        self.command = (executable, arguments)
        self.running = True

    def kill(self):
        self.running = False


class InstallDialogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication(['install-dialog-tests'])

    def setUp(self):
        self.dialog = EnvironmentCheckDialog()
        self.addCleanup(self.dialog.close)
        self.report = dict(mode='EXE 环境 · 标准深度版', summary='缺少组件', advice=[],
                           install_supported=True, install_requirements=['hmmlearn==0.3.3'],
                           install_reason='安装缺少的组件')

    def test_button_is_next_to_copy_and_disabled_before_a_complete_report(self):
        buttons = self.dialog.layout().itemAt(3).layout()
        self.assertEqual(buttons.indexOf(self.dialog.install_button),
                         buttons.indexOf(self.dialog.copy_button) + 1)
        self.assertEqual(self.dialog.install_button.text(), '一键安装')
        self.assertFalse(self.dialog.install_button.isEnabled())

    def test_complete_report_enables_install_and_click_starts_worker(self):
        self.dialog.process = FakeProcess({'type': 'complete', 'report': self.report})
        self.dialog._read_stdout()
        self.dialog._idle()
        self.assertTrue(self.dialog.install_button.isEnabled())
        with patch('pyml_workbench.environment_dialog.worker_command',
                   return_value=('worker.exe', ['environment-install'])):
            self.dialog.install_button.click()
        self.assertEqual(self.dialog.process.command, ('worker.exe', ['environment-install']))
        self.assertFalse(self.dialog.install_button.isEnabled())
        self.assertFalse(self.dialog.check_button.isEnabled())
        self.assertEqual(self.dialog.cancel_button.text(), '停止安装')
        self.dialog.cancel_check()
        self.assertFalse(self.dialog.process.running)

    def test_success_automatically_rechecks_and_retains_restart_notice(self):
        self.dialog._action = 'install'
        self.dialog.process = FakeProcess({'type': 'install_complete', 'result': {
            'restart_required': True, 'message': '安装完成'}})
        with patch.object(self.dialog, 'start_check') as recheck:
            self.dialog._finished(0, QProcess.ExitStatus.NormalExit)
            self.app.processEvents()
            recheck.assert_called_once()
        self.assertTrue(self.dialog._restart_required)

    def test_failed_or_cancelled_install_does_not_claim_success(self):
        self.dialog._action = 'install'
        self.dialog.process = FakeProcess({'type': 'failure', 'message': '网络错误'})
        with patch.object(self.dialog, 'start_check') as recheck:
            self.dialog._finished(1, QProcess.ExitStatus.NormalExit)
            self.app.processEvents()
            recheck.assert_not_called()
        self.assertFalse(self.dialog._complete)
        self.assertFalse(self.dialog.install_button.isEnabled())
        self.assertIn('网络错误', self.dialog.report_edit.toPlainText())


if __name__ == '__main__':
    unittest.main()
