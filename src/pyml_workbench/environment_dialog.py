"""Non-modal environment report with a separate, cancellable QProcess."""
from __future__ import annotations

import codecs
import json
import os
import subprocess

from PySide6.QtCore import QProcess, QTimer
from PySide6.QtWidgets import (
    QApplication, QDialog, QHBoxLayout, QLabel, QPlainTextEdit,
    QProgressBar, QPushButton, QVBoxLayout,
)

from .environment_check import GROUP_LABELS, STATUS_LABELS
from .runtime import worker_command


class EnvironmentCheckDialog(QDialog):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("检查运行环境")
        self.resize(860, 600)
        layout = QVBoxLayout(self)
        self.status_label = QLabel("点击“开始检查”，核对当前程序实际使用的运行环境。")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)
        self.progress = QProgressBar()
        self.progress.setRange(0, 1)
        self.progress.setValue(0)
        layout.addWidget(self.progress)
        self.report_edit = QPlainTextEdit()
        self.report_edit.setReadOnly(True)
        layout.addWidget(self.report_edit)
        buttons = QHBoxLayout()
        self.check_button = QPushButton("开始检查")
        self.check_button.clicked.connect(self.start_check)
        buttons.addWidget(self.check_button)
        self.cancel_button = QPushButton("停止检查")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self.cancel_check)
        buttons.addWidget(self.cancel_button)
        self.copy_button = QPushButton("复制报告")
        self.copy_button.setEnabled(False)
        self.copy_button.clicked.connect(self.copy_report)
        buttons.addWidget(self.copy_button)
        self.install_button = QPushButton("一键安装")
        self.install_button.setObjectName("installDependenciesButton")
        self.install_button.setAccessibleName("一键安装缺少的组件")
        self.install_button.setEnabled(False)
        self.install_button.setToolTip("完成环境检查后，可安装报告中缺少的组件。")
        self.install_button.clicked.connect(self.start_install)
        buttons.addWidget(self.install_button)
        buttons.addStretch()
        close_button = QPushButton("关闭")
        close_button.clicked.connect(self.close)
        buttons.addWidget(close_button)
        layout.addLayout(buttons)
        self.process = QProcess(self)
        self.process.readyReadStandardOutput.connect(self._read_stdout)
        self.process.readyReadStandardError.connect(self._read_stderr)
        self.process.finished.connect(self._finished)
        self.process.errorOccurred.connect(self._process_error)
        self.timeout = QTimer(self)
        self.timeout.setSingleShot(True)
        self.timeout.setInterval(60000)
        self.timeout.timeout.connect(self._timed_out)
        self.total_timeout = QTimer(self)
        self.total_timeout.setSingleShot(True)
        self.total_timeout.setInterval(300000)
        self.total_timeout.timeout.connect(self._timed_out)
        self._complete = False
        self._action = "check"
        self._last_report = None
        self._install_succeeded = False
        self._restart_required = False
        self._stopped = False
        self._buffer = ""
        self._stderr = ""
        self._decoder = codecs.getincrementaldecoder("utf-8")("replace")
        self._stderr_decoder = codecs.getincrementaldecoder("utf-8")("replace")

    def start_check(self) -> None:
        if self.process.state() != QProcess.ProcessState.NotRunning:
            return
        self._action = "check"
        self._last_report = None
        self._complete = self._stopped = False
        self._buffer = self._stderr = ""
        self._decoder.reset()
        self._stderr_decoder.reset()
        self.report_edit.clear()
        self.copy_button.setEnabled(False)
        try:
            executable, arguments = worker_command(["environment-check"])
        except (OSError, ValueError) as exc:
            self._fail(str(exc))
            return
        self.check_button.setEnabled(False)
        self.install_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        self.cancel_button.setText("停止检查")
        self.progress.setRange(0, 0)
        self.status_label.setText("正在启动独立检查进程…")
        self.timeout.start(60000)
        self.total_timeout.start(300000)
        self.process.start(executable, arguments)

    def start_install(self) -> None:
        if (self.process.state() != QProcess.ProcessState.NotRunning
                or not self._last_report or not self._last_report.get("install_supported")):
            return
        try:
            executable, arguments = worker_command(["environment-install"])
        except (OSError, ValueError) as exc:
            self._fail(str(exc))
            return
        self._action = "install"
        self._install_succeeded = False
        self._complete = self._stopped = False
        self._buffer = self._stderr = ""
        self._decoder.reset()
        self._stderr_decoder.reset()
        self._append("\n一键安装：" + "、".join(self._last_report["install_requirements"]))
        self._last_report = None
        self.check_button.setEnabled(False)
        self.install_button.setEnabled(False)
        self.cancel_button.setText("停止安装")
        self.cancel_button.setEnabled(True)
        self.progress.setRange(0, 0)
        self.status_label.setText("正在准备安装；完成后自动重新检查。")
        self.timeout.start(300000)
        self.total_timeout.start(1800000)
        self.process.start(executable, arguments)

    def _append(self, text: str) -> None:
        self.report_edit.appendPlainText(text)
        self.copy_button.setEnabled(True)

    def _read_stdout(self) -> None:
        self._buffer += self._decoder.decode(bytes(self.process.readAllStandardOutput()))
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            if not line.strip():
                continue
            try:
                event = json.loads(line)
                if self._stopped:
                    continue
                kind = event["type"]
                self.timeout.start()
                if kind == "progress":
                    self.status_label.setText(event["message"])
                elif kind == "row":
                    row = event["row"]
                    group = GROUP_LABELS.get(row["group"], "运行环境")
                    scope = "必需" if row["required"] else "可选"
                    self._append(f"[{STATUS_LABELS[row['status']]}] {row['name']} · {group}（{scope}）\n"
                                 f"  当前：{row['installed']}；要求：{row['expected']}\n  {row['detail']}\n")
                elif kind == "complete":
                    report = event["report"]
                    self._last_report = report
                    self._complete = True
                    suffix = "\n新安装的组件将在重启工作台后用于训练。" if self._restart_required else ""
                    self.status_label.setText(report["mode"] + "\n" + report["summary"] + suffix)
                    self._append(report["mode"] + "\n" + report["summary"] + "\n\n" + "\n".join(report["advice"]))
                    if suffix:
                        self._append(suffix.strip())
                elif kind == "install_log":
                    self._append(event["message"])
                elif kind == "install_complete":
                    self._complete = True
                    self._install_succeeded = True
                    self._restart_required = bool(event["result"]["restart_required"])
                    self.status_label.setText(event["result"]["message"])
                    self._append(event["result"]["message"])
                elif kind == "failure":
                    self._fail(event["message"])
            except (ValueError, KeyError, TypeError) as exc:
                self._fail(f"检查进程返回无效报告：{exc}")
                self._stop_process()

    def _read_stderr(self) -> None:
        self._stderr = (self._stderr + self._stderr_decoder.decode(
            bytes(self.process.readAllStandardError())))[-8000:]

    def _idle(self) -> None:
        self.timeout.stop()
        self.total_timeout.stop()
        self.progress.setRange(0, 1)
        self.progress.setValue(1 if self._complete else 0)
        running = self.process.state() != QProcess.ProcessState.NotRunning
        self.check_button.setEnabled(not running)
        self.check_button.setText("重新检查")
        self.cancel_button.setEnabled(running)
        supported = bool(self._complete and self._last_report
                         and self._last_report.get("install_supported"))
        self.install_button.setEnabled(not running and supported)
        if self._last_report:
            self.install_button.setToolTip(self._last_report.get("install_reason", ""))

    def _fail(self, message: str) -> None:
        self._complete = False
        self._last_report = None
        self._install_succeeded = False
        self._stopped = True
        self.status_label.setText("检查未完成，请查看下方原因。")
        self._append(message)
        self._idle()

    def _finished(self, exit_code: int, exit_status: QProcess.ExitStatus) -> None:
        self._read_stdout()
        self._read_stderr()
        if not self._stopped and (not self._complete or exit_code != 0 or exit_status != QProcess.ExitStatus.NormalExit):
            self._fail("后台检查未正常完成。若为 EXE，请重新完整解压发行包；若为 Python 环境，请同步项目依赖。\n"
                       + self._stderr.strip())
        self._idle()
        if (self._action == "install" and self._install_succeeded and not self._stopped
                and exit_code == 0 and exit_status == QProcess.ExitStatus.NormalExit):
            self._install_succeeded = False
            QTimer.singleShot(0, self.start_check)

    def _process_error(self, error: QProcess.ProcessError) -> None:
        if error == QProcess.ProcessError.FailedToStart:
            self._fail("无法启动后台检查：" + self.process.errorString())

    def _timed_out(self) -> None:
        self._fail("检查超时：" + self.status_label.text() + "。可重新检查；持续失败时请检查对应组件或重新解压发行包。")
        self._stop_process()

    def cancel_check(self) -> None:
        installing = self._action == "install"
        self._fail("已停止安装；可重新检查后重试。" if installing else
                   "已停止检查；以上结果不构成完整环境检查。")
        self._stop_process()

    def _stop_process(self) -> None:
        if self._action == "install" and os.name == "nt" and self.process.processId():
            try:
                subprocess.run(["taskkill", "/PID", str(self.process.processId()), "/T", "/F"],
                               creationflags=subprocess.CREATE_NO_WINDOW, timeout=5,
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except (OSError, subprocess.TimeoutExpired):
                pass
        self.process.kill()

    def copy_report(self) -> None:
        QApplication.clipboard().setText(self.report_edit.toPlainText())

    def closeEvent(self, event) -> None:
        if self.process.state() != QProcess.ProcessState.NotRunning:
            self.cancel_check()
            self.process.waitForFinished(1500)
        event.accept()
