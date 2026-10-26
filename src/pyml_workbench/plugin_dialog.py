"""Unified settings for all supported data-conversion plugin sources."""
from __future__ import annotations

from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from .plugins import (
    HYPEREEG_SOURCE,
    PYFNIRS_SOURCE,
    PluginError,
    PluginSettings,
    PluginSettingsStore,
    detect_hypereeg_data_conversion,
    detect_pyfnirs_data_conversion,
)


_SOURCES = (
    (HYPEREEG_SOURCE, detect_hypereeg_data_conversion),
    (PYFNIRS_SOURCE, detect_pyfnirs_data_conversion),
)


class PluginSettingsDialog(QDialog):
    """Edit both conversion sources together while retaining their separate keys."""

    settings_saved = Signal()

    def __init__(self, store: PluginSettingsStore, parent=None):
        super().__init__(parent)
        self.setObjectName("pluginSettingsDialog")
        self.store = store
        self.directory_edits: dict[str, QLineEdit] = {}
        self.python_edits: dict[str, QLineEdit] = {}
        self.status_labels: dict[str, QLabel] = {}
        self.setWindowTitle("插件设置")
        self.resize(700, 560)

        layout = QVBoxLayout(self)
        self.explanation_label = QLabel(
            "在此同时管理 HyperEEG 和 PYfNIRs 插件来源。每项配置独立保存；"
            "未配置或检查无效的入口保持灰色，保存有效设置后立即启用。"
        )
        self.explanation_label.setWordWrap(True)
        layout.addWidget(self.explanation_label)

        self.store_error_label = QLabel()
        self.store_error_label.setWordWrap(True)
        self.store_error_label.setStyleSheet("color: #B45309;")
        layout.addWidget(self.store_error_label)

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        content = QWidget(scroll)
        content_layout = QVBoxLayout(content)
        for source, _detector in _SOURCES:
            content_layout.addWidget(self._build_source_group(source))
        content_layout.addStretch(1)
        scroll.setWidget(content)
        layout.addWidget(scroll, 1)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QDialogButtonBox.StandardButton.Save).setText("保存")
        buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("取消")
        buttons.accepted.connect(self.save_settings)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        try:
            registry = self.store.load_all()
        except PluginError as exc:
            registry = {}
            self.store_error_label.setText(f"无法读取现有插件设置：{exc}")
        for source, _detector in _SOURCES:
            settings = registry.get(source, PluginSettings())
            self.directory_edits[source].setText(settings.directory or "")
            self.python_edits[source].setText(settings.python_executable or "")
            self.check_source(source)

    def _build_source_group(self, source: str) -> QGroupBox:
        group = QGroupBox(source, self)
        form = QFormLayout(group)

        directory_edit = QLineEdit(group)
        directory_edit.setObjectName(f"{source.casefold()}PluginDirectoryEdit")
        directory_edit.setPlaceholderText("选择包含 pyml-plugin.json 的插件目录")
        self.directory_edits[source] = directory_edit

        directory_row = QHBoxLayout()
        directory_row.addWidget(directory_edit, 1)
        browse_directory = QPushButton("浏览…", group)
        browse_directory.clicked.connect(
            lambda _checked=False, key=source: self.choose_directory(key)
        )
        directory_row.addWidget(browse_directory)
        form.addRow("插件目录", directory_row)

        python_edit = QLineEdit(group)
        python_edit.setObjectName(f"{source.casefold()}PluginPythonEdit")
        python_edit.setPlaceholderText("独立 EXE 无需填写；Python 插件可选择其环境")
        self.python_edits[source] = python_edit

        python_row = QHBoxLayout()
        python_row.addWidget(python_edit, 1)
        browse_python = QPushButton("浏览…", group)
        browse_python.clicked.connect(
            lambda _checked=False, key=source: self.choose_python(key)
        )
        python_row.addWidget(browse_python)
        form.addRow("Python 程序（可选）", python_row)

        status = QLabel(group)
        status.setObjectName(f"{source.casefold()}PluginStatusLabel")
        status.setWordWrap(True)
        self.status_labels[source] = status
        form.addRow("状态", status)

        clear_button = QPushButton("清除此项配置", group)
        clear_button.clicked.connect(
            lambda _checked=False, key=source: self.clear_source(key)
        )
        form.addRow("", clear_button)

        directory_edit.textChanged.connect(
            lambda _text, key=source: self.check_source(key)
        )
        python_edit.textChanged.connect(
            lambda _text, key=source: self.check_source(key)
        )
        return group

    def _settings_for(self, source: str) -> PluginSettings:
        return PluginSettings(
            self.directory_edits[source].text().strip() or None,
            self.python_edits[source].text().strip() or None,
        )

    def choose_directory(self, source: str) -> None:
        selected = QFileDialog.getExistingDirectory(
            self,
            f"选择 {source} 插件目录",
            self.directory_edits[source].text(),
        )
        if selected:
            self.directory_edits[source].setText(selected)

    def choose_python(self, source: str) -> None:
        selected, _ = QFileDialog.getOpenFileName(
            self,
            f"选择 {source} 插件环境的 Python 程序",
            self.python_edits[source].text(),
            "Python 程序 (python.exe pythonw.exe);;所有文件 (*)",
        )
        if selected:
            self.python_edits[source].setText(selected)

    def clear_source(self, source: str) -> None:
        self.directory_edits[source].clear()
        self.python_edits[source].clear()

    def check_source(self, source: str) -> None:
        settings = self._settings_for(source)
        status = self.status_labels[source]
        if not settings.directory:
            status.setText("未设置；对应菜单入口保持灰色禁用。")
            status.setStyleSheet("color: #6B7280;")
            return
        detector = dict(_SOURCES)[source]
        try:
            plugin = detector(settings)
        except PluginError as exc:
            status.setText(f"当前不可用：{exc}")
            status.setStyleSheet("color: #6B7280;")
        else:
            status.setText(f"已检测到 {plugin.name} {plugin.version}；保存后启用对应入口。")
            status.setStyleSheet("color: #047857;")

    def save_settings(self) -> None:
        updates: dict[str, PluginSettings | None] = {}
        for source, _detector in _SOURCES:
            settings = self._settings_for(source)
            if settings.directory or settings.python_executable:
                updates[source] = settings
            else:
                updates[source] = None
        try:
            self.store.update_sources(updates)
        except PluginError as exc:
            QMessageBox.warning(self, "无法保存插件设置", str(exc))
            return
        self.settings_saved.emit()
        self.accept()
