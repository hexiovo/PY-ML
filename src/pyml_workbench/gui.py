"""Chinese Qt desktop workbench for one frozen tabular ML experiment."""
from __future__ import annotations

import hashlib
import json
import math
import os
import codecs
import re
import shutil
import sys
import tempfile
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from .catalog import list_models
from .config import DatasetConfig, ExperimentConfig, SplitConfig
from .data import LoadedDataset, load_dataset
from .parameters import parameter_defaults, parameter_schema
from .preferences import PreferenceError, PreferenceStore, Preferences
from .runtime import worker_command
from .sequence import SequenceConfig

_WORKER_ERROR_ID_RE = re.compile(r"^[0-9a-f]{32}$")
_NO_PARAMETER_OVERRIDE = object()

# Matplotlib's Qt backend imports dateutil after loading Qt. Prime its lazy
# six.moves module first to avoid this runtime's Shiboken import-hook failure.
import dateutil.rrule  # noqa: F401

from matplotlib import rcParams
from matplotlib.font_manager import findfont as find_matplotlib_font
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

# Use an installed CJK-capable font for the charts rendered in this process.
# This stays local to the workbench and does not write Matplotlib user config.
rcParams["font.family"] = ["Microsoft YaHei"]
rcParams["axes.unicode_minus"] = False

# Import the project package before Qt. This order is required by the bundled
# Windows runtime's Shiboken import hook (see P03 startup verification).
from PySide6.QtCore import (
    QAbstractTableModel,
    QModelIndex,
    QProcess,
    QProcessEnvironment,
    QUrl,
    Qt,
    QTimer,
)
from PySide6.QtGui import QAction, QDesktopServices, QFont, QFontDatabase
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QHeaderView,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QMenu,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QTableView,
    QVBoxLayout,
    QWidget,
)


TASKS = (
    ("分类", "classification"),
    ("回归", "regression"),
    ("聚类", "clustering"),
    ("降维", "dimensionality reduction"),
    ("异常检测", "anomaly detection"),
    ("序列建模 / HMM", "sequence_modeling"),
)

EVENT_LABELS = {
    "dataset_loaded": "已读取数据",
    "target_validated": "已检查目标列",
    "split_created": "已建立训练 / 验证 / 测试划分",
    "preprocessor_fit_train_only": "预处理器只在训练集拟合",
    "estimator_fit_train_only": "模型只在训练集拟合",
    "train_metrics_computed": "已计算训练集指标",
    "validation_metrics_computed": "已计算验证集指标",
    "selection_frozen": "参数快照已记录；尚未由用户冻结",
    "test_metrics_computed_once": "已一次性计算最终测试指标",
    "test_not_evaluated_capability_guard": "模型能力限制阻止最终测试评估",
}

OPERATION_LABELS = {
    "predict": "预测",
    "predict_proba": "预测概率",
    "decision_function": "决策函数",
    "score_samples": "样本异常分数",
    "transform": "特征变换",
}


def _default_preferences_path() -> Path:
    if os.name == "nt":
        root = Path(os.environ.get("APPDATA") or Path.home() / "AppData" / "Roaming")
        return root / "PY-ML" / "preferences.json"
    root = Path(
        os.environ.get("XDG_CONFIG_HOME") or Path.home() / ".config"
    )
    return root / "pyml-workbench" / "preferences.json"


class _ApplicationLogController:
    """Application-owned logger lifecycle with failure-safe directory switching."""

    def __init__(self, log_root: str | Path | None):
        from .diagnostics import LogSession

        self._session_factory = LogSession
        self.log_root = None if log_root is None else str(log_root)
        self.session = LogSession(
            log_root=log_root, session_id=uuid.uuid4().hex, role="gui"
        )

    @property
    def available(self) -> bool:
        return bool(self.session.available)

    @property
    def last_error(self) -> str | None:
        return self.session.last_error

    def switch_directory(self, log_root: str | Path | None) -> None:
        candidate = self._session_factory(
            log_root=log_root, session_id=uuid.uuid4().hex, role="gui"
        )
        if not candidate.available:
            message = candidate.last_error or "日志处理器不可用"
            candidate.close()
            raise RuntimeError(message)
        previous = self.session
        self.session = candidate
        self.log_root = None if log_root is None else str(log_root)
        previous.close()

    def record_error(self, exc: BaseException, *, context=None):
        return self.session.record_error(exc, context=context)

    def record_external_error(
        self,
        error_type: str,
        message: str,
        traceback_text: str | None = None,
        *,
        context=None,
        error_id: str | None = None,
    ):
        return self.session.record_external_error(
            error_type,
            message,
            traceback_text,
            context=context,
            error_id=error_id,
        )

    def record_process_exit(self, returncode: int, *, context=None, error_id=None):
        return self.session.record_process_exit(
            returncode, context=context, error_id=error_id
        )

    def record_event(self, event: str, *, level="INFO", message=None, context=None):
        return self.session.record_event(
            event, level=level, message=message, context=context
        )

    def record_cancelled(self, *, context=None):
        return self.session.record_cancelled(context=context)

    def close(self) -> None:
        self.session.close()


@contextmanager
def _application_exception_hooks(controller):
    """Install hooks for the application owner and restore only our own hooks."""
    original_sys_hook = sys.excepthook
    original_thread_hook = threading.excepthook

    def record(exc: BaseException, stage: str) -> None:
        try:
            controller.record_error(exc, context={"stage": stage})
        except BaseException:
            pass

    def owned_sys_hook(exc_type, exc, tb):
        record(exc, "uncaught_main")
        original_sys_hook(exc_type, exc, tb)

    def owned_thread_hook(args):
        record(args.exc_value, "uncaught_thread")
        original_thread_hook(args)

    sys.excepthook = owned_sys_hook
    threading.excepthook = owned_thread_hook
    try:
        yield
    finally:
        if sys.excepthook is owned_sys_hook:
            sys.excepthook = original_sys_hook
        if threading.excepthook is owned_thread_hook:
            threading.excepthook = original_thread_hook


class DataFrameTableModel(QAbstractTableModel):
    """Small read-only model used for dataset and worksheet previews."""

    def __init__(self, frame: pd.DataFrame | None = None, parent=None):
        super().__init__(parent)
        self._frame = frame.copy(deep=False) if frame is not None else pd.DataFrame()

    def set_frame(self, frame: pd.DataFrame | None) -> None:
        self.beginResetModel()
        self._frame = frame.copy(deep=False) if frame is not None else pd.DataFrame()
        self.endResetModel()

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self._frame.index)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return 0 if parent.isValid() else len(self._frame.columns)

    def data(self, index: QModelIndex, role: int = Qt.ItemDataRole.DisplayRole):
        if not index.isValid() or role != Qt.ItemDataRole.DisplayRole:
            return None
        value = self._frame.iat[index.row(), index.column()]
        try:
            if pd.isna(value):
                return ""
        except (TypeError, ValueError):
            pass
        if isinstance(value, float):
            return f"{value:.6g}"
        return str(value)

    def headerData(self, section: int, orientation: Qt.Orientation, role: int = Qt.ItemDataRole.DisplayRole):
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        if orientation == Qt.Orientation.Horizontal and 0 <= section < len(self._frame.columns):
            return str(self._frame.columns[section])
        if orientation == Qt.Orientation.Vertical and 0 <= section < len(self._frame.index):
            return str(self._frame.index[section])
        return None


class DataOverviewDialog(QDialog):
    """Read-only full-table column metadata and pandas descriptive statistics."""

    def __init__(self, frame: pd.DataFrame, parent=None):
        super().__init__(parent)
        self.setWindowTitle("数据概况")
        self.resize(940, 680)
        layout = QVBoxLayout(self)
        self.summary_label = QLabel(f"完整数据：{len(frame):,} 行 × {len(frame.columns):,} 列")
        self.summary_label.setAccessibleName("完整数据行列数")
        layout.addWidget(self.summary_label)

        self.column_table = QTableView(self)
        self.column_table.setAccessibleName("完整数据列类型与缺失数量")
        column_info = pd.DataFrame({
            "列": [str(name) for name in frame.columns],
            "类型": [str(dtype) for dtype in frame.dtypes],
            "缺失数": [int(value) for value in frame.isna().sum().to_numpy()],
        })
        self.column_table.setModel(DataFrameTableModel(column_info, self.column_table))
        self.column_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.column_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(QLabel("逐列信息（覆盖完整数据）"))
        layout.addWidget(self.column_table, 1)

        self.describe_table = QTableView(self)
        self.describe_table.setAccessibleName("完整数据描述统计")
        description = frame.describe(include="all").transpose()
        description.index.name = "列"
        description = description.reset_index()
        self.describe_frame = description.copy(deep=True)
        self.describe_table.setModel(DataFrameTableModel(description, self.describe_table))
        self.describe_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self.describe_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(QLabel("描述统计（pandas describe，覆盖完整数据）"))
        layout.addWidget(self.describe_table, 2)

        close_button = QPushButton("关闭")
        close_button.clicked.connect(self.close)
        layout.addWidget(close_button)


class PlotSourceChooserDialog(QDialog):
    """Choose one owned source, one partition, columns, and an allowlisted plot."""

    def __init__(self, sources, *, unavailable=(), parent=None):
        super().__init__(parent)
        from .plotting import available_plot_specs

        self.setWindowTitle("选择图表来源")
        self.resize(620, 620)
        self._available_plot_specs = available_plot_specs
        self._source_groups: dict[str, dict[str, Any]] = {}
        self._sources = tuple(sources)
        self._unavailable = dict(unavailable)
        self.selected_source = None
        self.selected_spec = None
        layout = QVBoxLayout(self)

        self.source_combo = QComboBox(self)
        self.source_combo.setAccessibleName("图表数据来源")
        layout.addWidget(QLabel("来源"))
        layout.addWidget(self.source_combo)
        self.partition_combo = QComboBox(self)
        self.partition_combo.setAccessibleName("图表数据分区")
        layout.addWidget(QLabel("分区（每张图只使用一个分区）"))
        layout.addWidget(self.partition_combo)

        self.column_list = QListWidget(self)
        self.column_list.setAccessibleName("图表列选择")
        layout.addWidget(QLabel("列和缓存输出（勾选要用于图表的字段）"))
        layout.addWidget(self.column_list, 2)

        self.plot_combo = QComboBox(self)
        self.plot_combo.setAccessibleName("基础图表类型")
        layout.addWidget(QLabel("图表"))
        layout.addWidget(self.plot_combo)
        self.note_label = QLabel(self)
        self.note_label.setWordWrap(True)
        self.note_label.setAccessibleName("图表来源可用性说明")
        layout.addWidget(self.note_label)
        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            parent=self,
        )
        self.buttons.button(QDialogButtonBox.StandardButton.Ok).setText("打开图表")
        self.buttons.button(QDialogButtonBox.StandardButton.Cancel).setText("取消")
        layout.addWidget(self.buttons)

        for source in self._sources:
            key = json.dumps(
                [source.source_kind, source.source_id, source.owner_kind, source.owner_id],
                ensure_ascii=False,
                separators=(",", ":"),
            )
            self._source_groups.setdefault(key, {})[source.partition] = source
        for key, partitions in self._source_groups.items():
            first = next(iter(partitions.values()))
            if first.owner_kind == "loaded_dataset":
                title = f"已加载数据 · {first.source_id}"
            elif first.owner_kind == "session":
                title = f"本次实验 · {first.source_id}"
            else:
                title = f"批量任务 · {first.owner_id}"
            self.source_combo.addItem(title, key)

        self.source_combo.currentIndexChanged.connect(self._source_changed)
        self.partition_combo.currentIndexChanged.connect(self._refresh_columns)
        self.column_list.itemChanged.connect(self._refresh_plot_specs)
        self.plot_combo.currentIndexChanged.connect(self._refresh_accept_state)
        self.buttons.accepted.connect(self._accept_selection)
        self.buttons.rejected.connect(self.reject)
        self._source_changed()
        self._refresh_unavailable_note()

    def _source_changed(self, *_args):
        key = self.source_combo.currentData()
        partitions = self._source_groups.get(key, {})
        self.partition_combo.blockSignals(True)
        self.partition_combo.clear()
        for partition in ("all", "train", "validation", "test"):
            if partition in partitions:
                self.partition_combo.addItem(partition, partition)
        self.partition_combo.blockSignals(False)
        self._refresh_columns()

    def _current_source(self):
        key = self.source_combo.currentData()
        partition = self.partition_combo.currentData()
        return self._source_groups.get(key, {}).get(partition)

    def _refresh_columns(self, *_args):
        source = self._current_source()
        self.column_list.blockSignals(True)
        self.column_list.clear()
        if source is not None:
            for kind, columns in (("column", source.columns), ("output", source.outputs)):
                for column in columns:
                    item = QListWidgetItem(
                        f"{column.label} [{column.dtype}]"
                        if kind == "column"
                        else f"{column.label}（缓存输出）[{column.dtype}]",
                        self.column_list,
                    )
                    item.setData(Qt.ItemDataRole.UserRole, (kind, column.column_id))
                    item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                    item.setCheckState(Qt.CheckState.Checked)
        self.column_list.blockSignals(False)
        self._refresh_plot_specs()

    def _selected_source(self):
        source = self._current_source()
        if source is None:
            return None
        selected = {
            self.column_list.item(index).data(Qt.ItemDataRole.UserRole)[1]
            for index in range(self.column_list.count())
            if self.column_list.item(index).checkState() == Qt.CheckState.Checked
        }
        try:
            from .plot_cache import select_plot_source_columns

            return select_plot_source_columns(source, selected)
        except (TypeError, ValueError):
            return None

    def _refresh_plot_specs(self, *_args):
        selected_source = self._selected_source()
        specs = () if selected_source is None else self._available_plot_specs(selected_source)
        self.plot_combo.blockSignals(True)
        self.plot_combo.clear()
        columns = {
            column.column_id: str(column.label)
            for source in self._sources
            for column in source.columns + source.outputs
        }
        for spec in specs:
            details = " · ".join(columns.get(column_id, column_id) for column_id in spec.column_ids)
            label = spec.display_title if not details else f"{spec.display_title}：{details}"
            self.plot_combo.addItem(label, spec)
        self.plot_combo.blockSignals(False)
        if not specs:
            self.note_label.setText("当前选择没有可用图表。请勾选足够的列，或选择可用来源与分区。")
        self._refresh_accept_state()

    def _refresh_unavailable_note(self):
        if not self._unavailable:
            return
        details = "；".join(f"{name}：{reason}" for name, reason in self._unavailable.items())
        self.note_label.setText(f"部分分区不可用：{details}")

    def _refresh_accept_state(self, *_args):
        button = self.buttons.button(QDialogButtonBox.StandardButton.Ok)
        button.setEnabled(self.plot_combo.count() > 0 and self._selected_source() is not None)

    def _accept_selection(self):
        self.selected_source = self._selected_source()
        self.selected_spec = self.plot_combo.currentData()
        if self.selected_source is not None and self.selected_spec is not None:
            self.accept()


class _IntegerLineEdit(QLineEdit):
    """Text field that preserves arbitrary-width integer configuration values."""

    def __init__(self, value: int, *, allow_none: bool = False, parent=None):
        super().__init__(parent)
        self._allow_none = allow_none
        self.setValue(value)

    def setValue(self, value: int | None) -> None:
        if value is None:
            if not self._allow_none:
                raise ValueError("此字段必须是整数")
            self.setText("null")
            return
        if type(value) is not int:
            raise TypeError("此字段必须是整数")
        self.setText(str(value))

    def value(self) -> int | None:
        text = self.text().strip()
        if self._allow_none and text.casefold() == "null":
            return None
        if not re.fullmatch(r"[+-]?[0-9]+", text):
            expected = "整数或 null" if self._allow_none else "整数"
            raise ValueError(f"此字段必须输入{expected}")
        return int(text)


class WorkbenchWindow(QMainWindow):
    """Single-run UI with an explicit validation freeze and final-test action."""

    def __init__(
        self,
        *,
        preferences_path: str | Path | None = None,
        logging_controller=None,
    ):
        super().__init__()
        self.setWindowTitle("PY-ML 机器学习工作台")
        self.resize(1420, 930)
        self.setMinimumSize(1060, 720)
        self.setFont(QFont("Microsoft YaHei UI", 9))

        self.dataset: LoadedDataset | None = None
        self._split_fractions = SplitConfig()
        self._feature_selection_unspecified = False
        self.bound_result: dict[str, Any] | None = None
        self.current_metrics: dict[str, Any] = {}
        self.training_curves: dict[str, Any] | None = None
        self.received_events: list[str] = []
        self.loaded_model_path: str | None = None
        self.loaded_model_capabilities: dict[str, bool] = {}
        self.loaded_model_id: str | None = None
        self._worker_action: str | None = None
        self._config_file_path: Path | None = None
        self._run_signature: str | None = None
        self._prepared_signature: str | None = None
        self._session_id: str | None = None
        self._session_file_path: Path | None = None
        self._session_capabilities: dict[str, bool] = {}
        self._prepared = False
        self._frozen = False
        self._test_result_payload: dict[str, Any] | None = None
        self._finalize_requested = False
        self._cancel_requested = False
        self._params_initializing = False
        self._applying_configuration = False
        self._loaded_parameters: dict[str, Any] | None = None
        self._loaded_advanced_text: str | None = None
        self._loaded_advanced_names: set[str] = set()
        self._edited_parameter_names: set[str] = set()
        self._batch_dialog = None
        self._guide_dialog = None
        self._data_overview_dialog: DataOverviewDialog | None = None
        self._plot_dialogs: list[QDialog] = []
        self._session_plot_sources: dict[str, Any] = {}
        self.stderr_text = ""
        self._stderr_decoder = None
        self.last_error: str | None = None
        self.last_error_id: str | None = None
        self.last_error_traceback = ""
        self.artifact_paths: dict[str, str] = {}
        self.preference_store = PreferenceStore(
            preferences_path or _default_preferences_path()
        )
        self.preferences = Preferences()
        self.logging_controller = logging_controller

        self._build_ui()
        self._connect_ui()
        self._load_preferences()

        self.process = QProcess(self)
        self.process.setProcessChannelMode(QProcess.ProcessChannelMode.SeparateChannels)
        self.process.readyReadStandardOutput.connect(self._read_worker_stdout)
        self.process.readyReadStandardError.connect(self._read_worker_stderr)
        self.process.finished.connect(self._on_process_finished)
        self.process.errorOccurred.connect(self._on_process_error)

        self.setStyleSheet(self._style_sheet())
        self._refresh_deferred_models()
        self._refresh_model_choices()
        self._update_target_state()
        self._set_worker_busy(False)

    def _build_ui(self) -> None:
        central = QWidget(self)
        root = QVBoxLayout(central)
        root.setContentsMargins(18, 16, 18, 18)
        root.setSpacing(12)

        header = QHBoxLayout()
        title_block = QVBoxLayout()
        title = QLabel("PY-ML 机器学习工作台")
        title.setObjectName("pageTitle")
        subtitle = QLabel("导入表格，选择一个模型；先看验证结果，再确认一次性最终测试。")
        subtitle.setObjectName("pageSubtitle")
        title_block.addWidget(title)
        title_block.addWidget(subtitle)
        header.addLayout(title_block, 1)
        self.overall_guide_button = QPushButton("整体指南")
        self.overall_guide_button.setObjectName("overallGuideButton")
        self.overall_guide_button.setAccessibleName("打开整体指南")
        self.overall_guide_button.setToolTip("离线阅读完整操作指南，支持章节目录和搜索（F1）")
        header.addWidget(self.overall_guide_button)
        self.batch_search_button = QPushButton("批量搜索队列")
        self.batch_search_button.setAccessibleName("打开批量搜索队列")
        self.batch_search_button.setToolTip("配置多个数据集与模型组合，运行并恢复参数搜索队列")
        header.addWidget(self.batch_search_button)
        self.copy_error_button = QPushButton("复制错误详情")
        self.copy_error_button.setAccessibleName("复制错误 ID 和 traceback")
        self.copy_error_button.setEnabled(False)
        self.copy_error_button.clicked.connect(self.copy_error_details)
        header.addWidget(self.copy_error_button)
        self.state_label = QLabel("尚未加载数据")
        self.state_label.setObjectName("stateBadge")
        self.state_label.setAccessibleName("工作流状态")
        header.addWidget(self.state_label, 0, Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        root.addLayout(header)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        left_panel = self._build_left_panel()
        left_panel.setMinimumHeight(1020)
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        left_scroll.setWidget(left_panel)
        splitter.addWidget(left_scroll)
        splitter.addWidget(self._build_right_panel())
        splitter.setSizes([600, 760])
        splitter.setStretchFactor(0, 5)
        splitter.setStretchFactor(1, 6)
        root.addWidget(splitter, 1)
        self.setCentralWidget(central)
        self._build_application_menu()

    def _build_application_menu(self) -> None:
        file_menu = self.menuBar().addMenu("文件")
        self.recent_files_menu = file_menu.addMenu("最近文件")
        self.save_config_action = QAction("保存参数配置 JSON…", self)
        self.save_config_action.triggered.connect(
            lambda _checked=False: self.save_configuration_json()
        )
        file_menu.addAction(self.save_config_action)
        self.load_config_action = QAction("加载参数配置 JSON…", self)
        self.load_config_action.triggered.connect(
            lambda _checked=False: self.load_configuration_json()
        )
        file_menu.addAction(self.load_config_action)
        analysis_menu = self.menuBar().addMenu("分析")
        self.data_overview_action = QAction("数据概况…", self)
        self.data_overview_action.setObjectName("dataOverviewAction")
        self.data_overview_action.setEnabled(False)
        self.data_overview_action.triggered.connect(
            lambda _checked=False: self.show_data_overview()
        )
        analysis_menu.addAction(self.data_overview_action)
        self.basic_plot_action = QAction("基础图表…", self)
        self.basic_plot_action.setObjectName("basicPlotAction")
        self.basic_plot_action.setEnabled(False)
        self.basic_plot_action.triggered.connect(
            lambda _checked=False: self.open_plot_selector()
        )
        analysis_menu.addAction(self.basic_plot_action)
        self.result_plot_action = QAction("绘制结果图…", self)
        self.result_plot_action.setObjectName("resultPlotAction")
        self.result_plot_action.setEnabled(False)
        self.result_plot_action.triggered.connect(
            lambda _checked=False: self.open_plot_selector()
        )
        analysis_menu.addAction(self.result_plot_action)
        file_menu.addSeparator()
        self.export_diagnostics_action = QAction("导出诊断包…", self)
        self.export_diagnostics_action.setObjectName("exportDiagnosticsAction")
        self.export_diagnostics_action.setToolTip(
            "由你明确选择当前界面会话或一个 error_id，再选择 ZIP 保存位置；不会自动上传。"
        )
        self.export_diagnostics_action.triggered.connect(
            lambda _checked=False: self.export_diagnostic_archive()
        )
        file_menu.addAction(self.export_diagnostics_action)
        file_menu.addSeparator()
        self.clear_recent_files_action = QAction("清除最近文件记录", self)
        self.clear_recent_files_action.triggered.connect(
            lambda _checked=False: self.clear_recent_files()
        )
        file_menu.addAction(self.clear_recent_files_action)
        file_menu.addSeparator()
        self.choose_log_directory_action = QAction("选择日志目录…", self)
        self.choose_log_directory_action.triggered.connect(
            lambda _checked=False: self.choose_log_directory()
        )
        file_menu.addAction(self.choose_log_directory_action)
        self.reset_log_directory_action = QAction("恢复默认日志目录", self)
        self.reset_log_directory_action.triggered.connect(
            lambda _checked=False: self.set_log_directory(None)
        )
        file_menu.addAction(self.reset_log_directory_action)
        help_menu = self.menuBar().addMenu("帮助")
        self.overall_guide_action = QAction("整体指南", self)
        self.overall_guide_action.setShortcut("F1")
        self.overall_guide_action.triggered.connect(self.show_overall_guide)
        help_menu.addAction(self.overall_guide_action)

    def show_overall_guide(self) -> None:
        if self._guide_dialog is None:
            from .guide import OverallGuideDialog, load_overall_guide

            try:
                markdown = load_overall_guide()
            except (OSError, UnicodeError) as exc:
                if self.logging_controller is not None:
                    self.logging_controller.record_error(exc, context={"action": "open_guide"})
                QMessageBox.warning(
                    self, "整体指南不可用",
                    f"无法读取指南：{exc}\n请重新安装本地包或完整解压发行 ZIP。",
                )
                return
            self._guide_dialog = OverallGuideDialog(markdown, self)
        self._guide_dialog.show()
        self._guide_dialog.raise_()
        self._guide_dialog.activateWindow()

    def _load_preferences(self) -> None:
        try:
            self.preferences = self.preference_store.load()
        except PreferenceError as exc:
            self.result_status_label.setText(f"偏好设置读取失败：{exc}")
            self.preferences = Preferences()
        self._refresh_recent_files_menu()

    def _refresh_recent_files_menu(self) -> None:
        self.recent_files_menu.clear()
        if not self.preferences.recent_files:
            empty = self.recent_files_menu.addAction("（没有最近文件）")
            empty.setEnabled(False)
        else:
            for recent in self.preferences.recent_files:
                path = Path(recent.path)
                label = path.name or recent.path
                if not recent.exists:
                    label += "（文件不存在）"
                action = self.recent_files_menu.addAction(label)
                action.setToolTip(recent.path)
                action.setEnabled(recent.exists)
                action.triggered.connect(
                    lambda _checked=False, item_path=recent.path: self.load_data(
                        item_path
                    )
                )
        self.clear_recent_files_action.setEnabled(bool(self.preferences.recent_files))

    def clear_recent_files(self) -> bool:
        """Clear remembered paths without touching any source files."""
        try:
            self.preferences = self.preference_store.clear_recent_files()
        except PreferenceError as exc:
            self.result_status_label.setText(f"清除最近文件记录失败：{exc}")
            return False
        self._refresh_recent_files_menu()
        self.result_status_label.setText("最近文件记录已清除；源文件未更改。")
        return True

    def choose_log_directory(self) -> bool:
        current = self.preferences.log_directory or str(Path.home())
        directory = QFileDialog.getExistingDirectory(self, "选择日志目录", current)
        return self.set_log_directory(directory) if directory else False

    def set_log_directory(self, path: str | Path | None) -> bool:
        """Persist and activate the chosen log directory as one UI operation."""
        previous = self.preferences.log_directory
        try:
            updated = self.preference_store.set_log_directory(path)
        except PreferenceError as exc:
            self.result_status_label.setText(f"日志目录偏好保存失败：{exc}")
            return False
        try:
            if self.logging_controller is not None:
                self.logging_controller.switch_directory(updated.log_directory)
        except Exception as exc:
            try:
                self.preference_store.set_log_directory(previous)
            except PreferenceError as rollback_exc:
                self.result_status_label.setText(
                    f"日志目录启动失败：{exc}；恢复旧偏好也失败：{rollback_exc}"
                )
                return False
            self.result_status_label.setText(
                f"日志目录启动失败，仍使用原设置：{exc}"
            )
            return False
        self.preferences = updated
        self.result_status_label.setText(
            "日志目录已恢复为默认值。"
            if updated.log_directory is None
            else f"日志目录已切换：{updated.log_directory}"
        )
        return True

    def _latest_diagnostic_error_id(self, session) -> str:
        candidate = self.last_error_id
        if isinstance(candidate, str) and _WORKER_ERROR_ID_RE.fullmatch(candidate):
            return candidate
        try:
            from .diagnostics import read_records

            records = read_records(
                session.log_root,
                session_id=session.session_id,
                level="ERROR",
                limit=1,
            )
        except Exception:
            return ""
        candidate = records[-1].get("error_id") if records else None
        return (
            candidate
            if isinstance(candidate, str)
            and _WORKER_ERROR_ID_RE.fullmatch(candidate)
            else ""
        )

    def _set_diagnostic_export_feedback(
        self, dialog_parent, message: str, *, failed: bool
    ) -> None:
        self.result_status_label.setText(message)
        if dialog_parent is not self:
            status_label = getattr(dialog_parent, "status_label", None)
            if status_label is not None:
                status_label.setText(message)
        if failed:
            QMessageBox.warning(dialog_parent or self, "诊断包导出", message)

    def export_diagnostic_archive(
        self, *, dialog_parent=None, latest_error_id: str | None = None
    ) -> bool:
        """Export only a user-selected diagnostic session or error ID."""
        parent = dialog_parent or self
        controller = self.logging_controller
        session = getattr(controller, "session", None)
        if (
            session is None
            or not getattr(session, "available", False)
            or not isinstance(getattr(session, "session_id", None), str)
            or not _WORKER_ERROR_ID_RE.fullmatch(session.session_id)
            or not getattr(session, "log_root", None)
        ):
            reason = getattr(controller, "last_error", None) or "日志处理器未启动"
            self._set_diagnostic_export_feedback(
                parent,
                f"当前诊断日志不可用，未创建 ZIP；业务结果和日志设置未更改：{reason}",
                failed=True,
            )
            return False

        choices = ["当前界面会话", "按错误编号"]
        scope, accepted = QInputDialog.getItem(
            parent,
            "导出诊断包",
            "选择导出范围。内容经过 allowlist 与脱敏，最大 20 MiB；不会上传。",
            choices,
            0,
            False,
        )
        if not accepted:
            return False
        selection: dict[str, list[str]]
        if scope == choices[0]:
            selection = {"session_ids": [session.session_id]}
        elif scope == choices[1]:
            prefill = (
                latest_error_id
                if isinstance(latest_error_id, str)
                and _WORKER_ERROR_ID_RE.fullmatch(latest_error_id)
                else self._latest_diagnostic_error_id(session)
            )
            raw_error_id, accepted = QInputDialog.getText(
                parent,
                "导出诊断包",
                "输入 32 位 error_id；可以粘贴以前保存的编号。",
                QLineEdit.EchoMode.Normal,
                prefill,
            )
            if not accepted:
                return False
            selected_error_id = raw_error_id.strip().lower()
            if not _WORKER_ERROR_ID_RE.fullmatch(selected_error_id):
                self._set_diagnostic_export_feedback(
                    parent,
                    "error_id 必须是 32 位十六进制编号；未创建 ZIP，也没有更改业务结果或日志设置。",
                    failed=True,
                )
                return False
            selection = {"error_ids": [selected_error_id]}
        else:
            self._set_diagnostic_export_feedback(
                parent,
                "导出范围无效；未创建 ZIP，也没有更改业务结果或日志设置。",
                failed=True,
            )
            return False

        suggested = Path(session.log_root) / (
            "pyml-diagnostics-" + datetime.now().strftime("%Y%m%d-%H%M%S") + ".zip"
        )
        destination, _ = QFileDialog.getSaveFileName(
            parent,
            "保存诊断 ZIP",
            str(suggested),
            "诊断 ZIP (*.zip)",
        )
        if not destination:
            return False
        destination = str(Path(destination).expanduser())
        if Path(destination).suffix.casefold() != ".zip":
            destination += ".zip"

        try:
            from .diagnostics_archive import export_diagnostics_zip

            archive_path = export_diagnostics_zip(
                session.log_root,
                destination,
                **selection,
            )
        except Exception as exc:
            detail = str(exc)
            folded = detail.casefold()
            if "already exists" in folded:
                message = "目标 ZIP 已存在。为保护原文件，本次没有覆盖它；请选择一个新文件名。"
            elif "not found" in folded or "missing" in folded:
                message = (
                    "当前日志目录中没有找到该 error_id 对应的已保存记录；"
                    "请检查编号或当前日志目录。没有创建 ZIP。"
                )
            else:
                message = f"诊断包导出失败：{detail}；业务结果和日志设置未更改。"
            self._set_diagnostic_export_feedback(parent, message, failed=True)
            return False

        try:
            size_bytes = archive_path.stat().st_size
            message = f"诊断包已导出：{archive_path}（{size_bytes} 字节）"
        except OSError:
            message = f"诊断包已导出：{archive_path}"
        self._set_diagnostic_export_feedback(parent, message, failed=False)
        return True

    def save_configuration_json(self, path: str | Path | None = None) -> bool:
        """Save the validated current ExperimentConfig without running a model."""
        try:
            config = self.build_experiment_config()
        except Exception as exc:
            self.result_status_label.setText(f"无法保存参数配置：{exc}")
            return False
        destination = str(path) if path is not None else ""
        if not destination:
            destination, _ = QFileDialog.getSaveFileName(
                self,
                "保存参数配置 JSON",
                str(Path(self.dataset.source_path).with_suffix(".pyml.json")),
                "PY-ML 参数配置 (*.json)",
            )
        if not destination:
            return False
        if Path(destination).suffix.casefold() != ".json":
            destination += ".json"
        try:
            from .presets import save_preset

            digest = save_preset(destination, config)
        except Exception as exc:
            self.result_status_label.setText(f"参数配置保存失败：{exc}")
            return False
        self.result_status_label.setText(
            f"参数配置已保存：{destination} · SHA-256 {digest}"
        )
        return True

    def load_configuration_json(self, path: str | Path | None = None) -> bool:
        """Load a preset into widgets only; fitting and evaluation stay explicit."""
        source = str(path) if path is not None else ""
        if not source:
            source, _ = QFileDialog.getOpenFileName(
                self, "加载参数配置 JSON", "", "PY-ML 参数配置 (*.json)"
            )
        if not source:
            return False
        try:
            from .presets import load_preset

            config = load_preset(source)
            if not self.apply_experiment_config(config):
                return False
        except Exception as exc:
            self.result_status_label.setText(f"参数配置加载失败：{exc}")
            return False
        return True

    def apply_experiment_config(self, config: ExperimentConfig) -> bool:
        """Populate the complete single-run form and invalidate prior results."""
        if self.process.state() != QProcess.ProcessState.NotRunning:
            self.result_status_label.setText("后台任务运行期间不能加载参数配置。")
            return False
        try:
            config = ExperimentConfig.from_dict(config.to_dict())
            if type(config.split.seed) is not int or config.split.seed < 0:
                raise ValueError("split.seed 必须是非负整数")
            task_index = self.task_combo.findData(config.task)
            if task_index < 0:
                raise ValueError(f"当前界面不支持配置中的任务：{config.task}")
            model_ids = {row["id"] for row in list_models(config.task)}
            if config.model_id not in model_ids:
                raise ValueError(f"当前任务没有可运行的模型：{config.model_id}")
            parameters_json = json.dumps(
                config.parameters, ensure_ascii=False, allow_nan=False
            )
            prepared_dataset = None
            source_path = Path(config.dataset.source_path).expanduser()
            if source_path.is_file():
                prepared_dataset = load_dataset(
                    config.dataset.source_path,
                    sheet_name=config.dataset.sheet_name,
                )
            try:
                defaults = parameter_defaults(config.model_id)
                common_names = {
                    row["name"] for row in parameter_schema(config.model_id)
                    if row["group"] == "常用参数"
                }
            except Exception:
                defaults = {}
                common_names = set()
            for name, value in config.parameters.items():
                if name not in common_names or name not in defaults:
                    continue
                default = defaults[name]
                if isinstance(default, bool) and type(value) is not bool:
                    raise ValueError(f"参数 {name} 必须是布尔值")
                if type(default) is int and value is not None and type(value) is not int:
                    raise ValueError(f"参数 {name} 必须是整数或 null")
                if isinstance(default, float) and (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                ):
                    raise ValueError(f"参数 {name} 必须是有限实数")
                if isinstance(default, str) and not isinstance(value, str):
                    raise ValueError(f"参数 {name} 必须是文本")
        except Exception as exc:
            self.result_status_label.setText(f"参数配置加载失败：{exc}")
            return False

        previous_initializing = self._params_initializing
        previous_applying = self._applying_configuration
        self._applying_configuration = True
        self._params_initializing = True
        try:
            loaded = self._populate_configuration_form(
                config, task_index, parameters_json, prepared_dataset
            )
        except Exception as exc:
            self.result_status_label.setText(
                f"参数配置应用失败；原有 session 尚未删除：{exc}"
            )
            return False
        finally:
            self._params_initializing = previous_initializing
            self._applying_configuration = previous_applying

        self._update_sequence_config_state()
        self._discard_session(remove_config=True)
        self._clear_bound_result()
        self._run_signature = None
        self._prepared_signature = None
        self._prepared = False
        self._frozen = False
        self._test_result_payload = None
        self._finalize_requested = False
        self._cancel_requested = False
        self.state_label.setText("参数配置已加载")
        if loaded:
            self.result_status_label.setText(
                "参数配置已回填到界面；尚未启动拟合、搜索或测试。"
            )
        else:
            self.result_status_label.setText(
                f"参数配置已回填，但数据源不可用：{config.dataset.source_path}。选择文件并加载后再运行。"
            )
        self._update_worker_controls()
        return True

    def _populate_configuration_form(
        self,
        config: ExperimentConfig,
        task_index: int,
        parameters_json: str,
        prepared_dataset: LoadedDataset | None,
    ) -> bool:
        self.task_combo.setCurrentIndex(task_index)
        model_index = self.model_combo.findData(config.model_id)
        if model_index < 0:
            raise ValueError(f"当前任务没有可运行的模型：{config.model_id}")
        self.model_combo.setCurrentIndex(model_index)
        self._refresh_parameters()
        self.sequence_config_edit.setPlainText(
            json.dumps(
                config.sequence.to_dict() if config.sequence is not None else {},
                ensure_ascii=False,
                indent=2,
                allow_nan=False,
            )
        )
        self._split_fractions = config.split
        self.seed_spin.setValue(config.split.seed)
        self.source_path_edit.setText(config.dataset.source_path)
        loaded = self.load_data(
            config.dataset.source_path,
            sheet_name=config.dataset.sheet_name if prepared_dataset is not None else None,
            prepared_dataset=prepared_dataset,
            invalidate_result=False,
        )
        if config.output_dir is not None:
            self.output_dir_edit.setText(config.output_dir)
        else:
            self.output_dir_edit.clear()

        if loaded:
            target = config.dataset.target_column
            target_index = self.target_combo.findData(target)
            if target is not None and target_index < 0:
                self.target_combo.addItem(f"{target}（配置列缺失）", target)
                target_index = self.target_combo.count() - 1
            self.target_combo.setCurrentIndex(max(0, target_index))
            requested_features = config.dataset.feature_columns
            if requested_features is not None:
                existing_names = [
                    self.feature_list.item(index).text()
                    for index in range(self.feature_list.count())
                ]
                requested_set = set(requested_features)
                ordered_names = list(requested_features) + [
                    name for name in existing_names if name not in requested_set
                ]
                self.feature_list.blockSignals(True)
                self.feature_list.clear()
                for name in ordered_names:
                    item = QListWidgetItem(name)
                    item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
                    item.setCheckState(
                        Qt.CheckState.Checked
                        if name in requested_set
                        else Qt.CheckState.Unchecked
                    )
                    if name not in existing_names:
                        item.setToolTip("该特征列不在当前数据文件中")
                    self.feature_list.addItem(item)
                self.feature_list.blockSignals(False)
            self._update_target_state()
        else:
            self._restore_unavailable_dataset_fields(config)
        self._feature_selection_unspecified = config.dataset.feature_columns is None

        remaining = dict(config.parameters)
        for name, (control, kind) in self.common_parameter_widgets.items():
            if name not in remaining:
                continue
            value = remaining.pop(name)
            if kind == "bool":
                control.setChecked(value)
            elif kind == "int":
                control.setValue(value)
            elif kind == "float":
                control.setText(json.dumps(value, allow_nan=False))
            elif kind == "optional":
                control.setText(json.dumps(value, ensure_ascii=False, allow_nan=False))
            elif kind == "str":
                if isinstance(value, str):
                    control.setText(value)
                else:
                    control.setText(json.dumps(value, ensure_ascii=False, allow_nan=False))
        self.advanced_parameters.setPlainText(
            json.dumps(remaining, ensure_ascii=False, indent=2, allow_nan=False)
        )
        self._loaded_parameters = json.loads(parameters_json)
        self._loaded_advanced_text = self.advanced_parameters.toPlainText().strip() or "{}"
        self._loaded_advanced_names = set(remaining)
        self._edited_parameter_names.clear()
        return loaded

    def _restore_unavailable_dataset_fields(self, config: ExperimentConfig) -> None:
        self.sheet_combo.blockSignals(True)
        self.sheet_combo.clear()
        if config.dataset.sheet_name is not None:
            self.sheet_combo.addItem(config.dataset.sheet_name, config.dataset.sheet_name)
            self.sheet_combo.setCurrentIndex(0)
            self.sheet_combo.setEnabled(True)
        else:
            self.sheet_combo.addItem("（未指定工作表）", None)
            self.sheet_combo.setEnabled(False)
        self.sheet_combo.blockSignals(False)
        self.target_combo.blockSignals(True)
        self.target_combo.clear()
        self.target_combo.addItem("（不使用目标列）", None)
        if config.dataset.target_column is not None:
            self.target_combo.addItem(config.dataset.target_column, config.dataset.target_column)
            self.target_combo.setCurrentIndex(1)
        self.target_combo.blockSignals(False)
        self.feature_list.blockSignals(True)
        self.feature_list.clear()
        for name in config.dataset.feature_columns or ():
            item = QListWidgetItem(name)
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked)
            self.feature_list.addItem(item)
        self.feature_list.blockSignals(False)

    def _build_left_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 8, 0)
        layout.setSpacing(10)

        self.data_group = QGroupBox("1. 数据与列选择")
        data_layout = QVBoxLayout(self.data_group)
        path_row = QHBoxLayout()
        self.source_path_edit = QLineEdit()
        self.source_path_edit.setAccessibleName("训练数据文件路径")
        self.source_path_edit.setPlaceholderText("CSV、XLSX 或 XLS 文件")
        self.source_path_edit.setMinimumWidth(0)
        self.source_browse_button = QPushButton("选择文件")
        self.source_browse_button.setMinimumHeight(34)
        self.load_data_button = QPushButton("加载预览")
        self.load_data_button.setMinimumHeight(34)
        path_row.addWidget(self.source_path_edit, 1)
        path_row.addWidget(self.source_browse_button)
        path_row.addWidget(self.load_data_button)
        data_layout.addLayout(path_row)

        sheet_row = QHBoxLayout()
        sheet_row.addWidget(QLabel("工作表"))
        self.sheet_combo = QComboBox()
        self.sheet_combo.setAccessibleName("训练数据工作表")
        self.sheet_combo.setMinimumHeight(32)
        sheet_row.addWidget(self.sheet_combo, 1)
        self.data_info_label = QLabel("支持 CSV、XLSX、XLS；预览不会修改源文件。")
        self.data_info_label.setObjectName("helperText")
        data_layout.addLayout(sheet_row)
        data_layout.addWidget(self.data_info_label)

        self.preview_table = QTableView()
        self.preview_table.setModel(DataFrameTableModel())
        self.preview_table.setAlternatingRowColors(True)
        self.preview_table.setSelectionBehavior(QTableView.SelectionBehavior.SelectRows)
        self.preview_table.setMinimumHeight(155)
        self.preview_table.horizontalHeader().setStretchLastSection(True)
        self.preview_table.verticalHeader().setDefaultSectionSize(25)
        data_layout.addWidget(self.preview_table, 1)

        select_row = QHBoxLayout()
        target_form = QFormLayout()
        target_form.setContentsMargins(0, 0, 0, 0)
        self.target_combo = QComboBox()
        self.target_combo.setAccessibleName("目标列")
        self.target_combo.addItem("（不使用目标列）", None)
        target_form.addRow("目标列", self.target_combo)
        select_row.addLayout(target_form, 1)
        data_layout.addLayout(select_row)
        data_layout.addWidget(QLabel("特征列（勾选要使用的列）"))
        self.feature_list = QListWidget()
        self.feature_list.setAccessibleName("特征列选择")
        self.feature_list.setMinimumHeight(100)
        self.feature_list.setMaximumHeight(170)
        data_layout.addWidget(self.feature_list)
        layout.addWidget(self.data_group, 5)

        self.model_group = QGroupBox("2. 任务、模型与参数")
        model_layout = QVBoxLayout(self.model_group)
        model_form = QFormLayout()
        self.task_combo = QComboBox()
        self.task_combo.setAccessibleName("机器学习任务")
        for label, task in TASKS:
            self.task_combo.addItem(label, task)
        self.model_combo = QComboBox()
        self.model_combo.setAccessibleName("活动模型")
        model_form.addRow("任务", self.task_combo)
        model_form.addRow("活动模型", self.model_combo)
        model_layout.addLayout(model_form)

        self.deferred_list = QListWidget()
        self.deferred_list.setAccessibleName("尚未实现且不可选择的模型")
        self.deferred_list.setEnabled(False)
        self.deferred_list.setMaximumHeight(72)
        self.deferred_list.setToolTip("这些目录项尚未实现，不能用于训练。")
        self.deferred_box = QGroupBox("规划中（未实现，不可选择）")
        deferred_box_layout = QVBoxLayout(self.deferred_box)
        deferred_box_layout.setContentsMargins(8, 8, 8, 8)
        deferred_box_layout.addWidget(self.deferred_list)
        model_layout.addWidget(self.deferred_box)

        self.param_scroll = QScrollArea()
        self.param_scroll.setWidgetResizable(True)
        self.param_scroll.setMinimumHeight(110)
        self.param_scroll.setMaximumHeight(190)
        self.param_widget = QWidget()
        self.param_form = QFormLayout(self.param_widget)
        self.param_form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        self.param_scroll.setWidget(self.param_widget)
        model_layout.addWidget(self.param_scroll)
        self.common_parameter_widgets: dict[str, tuple[QWidget, str]] = {}

        self.advanced_parameters = QPlainTextEdit()
        self.advanced_parameters.setAccessibleName("高级参数 JSON")
        self.advanced_parameters.setPlaceholderText('{"parameter_name": value}')
        self.advanced_parameters.setMaximumHeight(68)
        self.advanced_parameters.setPlainText("{}")
        self.advanced_parameters_label = QLabel("高级参数（仅 JSON 对象；填写模型支持的参数名）")
        self.advanced_parameters_label.setWordWrap(True)
        model_layout.addWidget(self.advanced_parameters_label)
        model_layout.addWidget(self.advanced_parameters)
        self.sequence_config_label = QLabel("序列配置 JSON（group_column、time_column、order_mode、observation_columns、window、horizon）")
        self.sequence_config_label.setWordWrap(True)
        self.sequence_config_edit = QPlainTextEdit()
        self.sequence_config_edit.setAccessibleName("序列配置 SequenceConfig JSON")
        self.sequence_config_edit.setPlaceholderText(
            '{\n  "group_column": "series_id",\n  "time_column": "timestamp",\n'
            '  "order_mode": "time",\n  "observation_columns": ["value"],\n'
            '  "window": 10,\n  "horizon": 1\n}'
        )
        self.sequence_config_edit.setPlainText("{}")
        self.sequence_config_edit.setMaximumHeight(105)
        self.sequence_config_help = QLabel("留空或 {} 表示使用默认设置：按行顺序、单一序列；已配置的 group/time 列会自动从特征中排除。")
        self.sequence_config_help.setObjectName("helperText")
        self.sequence_config_help.setWordWrap(True)
        model_layout.addWidget(self.sequence_config_label)
        model_layout.addWidget(self.sequence_config_edit)
        model_layout.addWidget(self.sequence_config_help)
        self.seed_spin = _IntegerLineEdit(42)
        self.seed_spin.setPlaceholderText("非负整数")
        self.seed_spin.setAccessibleName("数据划分随机种子")
        seed_form = QFormLayout()
        seed_form.addRow("划分随机种子", self.seed_spin)
        model_layout.addLayout(seed_form)
        layout.addWidget(self.model_group, 4)

        self.output_group = QGroupBox("3. 结果目录")
        output_layout = QHBoxLayout(self.output_group)
        self.output_dir_edit = QLineEdit()
        self.output_dir_edit.setAccessibleName("实验结果目录")
        self.output_dir_edit.setPlaceholderText("选择一个尚未创建或空的结果目录")
        self.output_browse_button = QPushButton("选择目录")
        output_layout.addWidget(self.output_dir_edit, 1)
        output_layout.addWidget(self.output_browse_button)
        layout.addWidget(self.output_group)

        action_row = QHBoxLayout()
        self.run_button = QPushButton("训练并验证")
        self.run_button.setObjectName("primaryButton")
        self.run_button.setMinimumHeight(42)
        self.run_button.setAccessibleName("训练并计算验证集指标")
        self.finalize_button = QPushButton("锁定并执行最终测试")
        self.finalize_button.setMinimumHeight(42)
        self.finalize_button.setEnabled(False)
        self.finalize_button.setAccessibleName("在冻结模型上执行一次最终测试")
        self.cancel_button = QPushButton("取消")
        self.cancel_button.setMinimumHeight(42)
        action_row.addWidget(self.run_button, 1)
        action_row.addWidget(self.finalize_button, 1)
        action_row.addWidget(self.cancel_button)
        layout.addLayout(action_row)
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(False)
        self.progress_bar.setRange(0, 0)
        self.progress_bar.hide()
        layout.addWidget(self.progress_bar)
        return panel

    def _build_right_panel(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(8, 0, 0, 0)
        layout.setSpacing(10)

        self.result_status_label = QLabel("尚无实验结果。训练后先查看训练集和验证集，再决定是否执行最终测试。")
        self.result_status_label.setObjectName("statusText")
        self.result_status_label.setWordWrap(True)
        layout.addWidget(self.result_status_label)

        self.result_tabs = QTabWidget()
        overview = QWidget()
        overview_layout = QVBoxLayout(overview)
        metric_row = QHBoxLayout()
        metric_row.addWidget(QLabel("图表指标"))
        overview_layout.setContentsMargins(8, 8, 8, 8)
        overview_layout.setSpacing(6)
        self.metric_combo = QComboBox()
        self.metric_combo.setAccessibleName("图表指标选择")
        metric_row.addWidget(self.metric_combo, 1)
        overview_layout.addLayout(metric_row)
        self.figure = Figure(figsize=(7.2, 3.2), tight_layout=True)
        self.metric_axes = self.figure.add_subplot(111)
        self.metric_canvas = FigureCanvasQTAgg(self.figure)
        self.metric_canvas.setMinimumHeight(170)
        self.metric_canvas.setMaximumHeight(185)
        overview_layout.addWidget(self.metric_canvas, 3)
        self.metrics_text = QPlainTextEdit()
        self.metrics_text.setReadOnly(True)
        self.metrics_text.setAccessibleName("训练、验证与测试指标")
        self.metrics_text.setMinimumHeight(80)
        self.metrics_text.setMaximumHeight(100)
        overview_layout.addWidget(self.metrics_text, 2)
        self.result_tabs.addTab(overview, "指标")

        self.sequence_summary_text = QPlainTextEdit()
        self.sequence_summary_text.setReadOnly(True)
        self.sequence_summary_text.setAccessibleName("序列训练验证测试实际划分统计")
        self.sequence_summary_text.setPlaceholderText("序列模型完成训练后会显示各分区的源行、组和窗口数量与比例。")
        self.result_tabs.addTab(self.sequence_summary_text, "序列划分")

        curves = QWidget()
        curves_layout = QVBoxLayout(curves)
        self.curve_status_label = QLabel("skorch 深度模型完成训练后会显示真实逐 epoch 损失。")
        self.curve_status_label.setObjectName("helperText")
        curves_layout.addWidget(self.curve_status_label)
        self.curve_figure = Figure(figsize=(7.2, 3.0), tight_layout=True)
        self.curve_axes = self.curve_figure.add_subplot(111)
        self.curve_canvas = FigureCanvasQTAgg(self.curve_figure)
        curves_layout.addWidget(self.curve_canvas, 1)
        self.result_tabs.addTab(curves, "训练曲线")

        self.events_text = QPlainTextEdit()
        self.events_text.setReadOnly(True)
        self.events_text.setAccessibleName("训练阶段事件日志")
        self.result_tabs.addTab(self.events_text, "阶段日志")

        artifacts_tab = QWidget()
        artifacts_layout = QVBoxLayout(artifacts_tab)
        self.artifacts_list = QListWidget()
        self.artifacts_list.setAccessibleName("实验导出文件")
        self.artifacts_list.itemDoubleClicked.connect(self._open_selected_artifact)
        artifacts_layout.addWidget(self.artifacts_list, 1)
        artifact_buttons = QHBoxLayout()
        self.open_artifact_button = QPushButton("打开文件或目录")
        self.export_artifact_button = QPushButton("导出选中文件副本")
        artifact_buttons.addWidget(self.open_artifact_button)
        artifact_buttons.addWidget(self.export_artifact_button)
        artifacts_layout.addLayout(artifact_buttons)
        self.audit_text = QPlainTextEdit()
        self.audit_text.setReadOnly(True)
        self.audit_text.setAccessibleName("实验审计信息")
        self.audit_text.setMaximumHeight(115)
        artifacts_layout.addWidget(self.audit_text)
        self.result_tabs.addTab(artifacts_tab, "导出与审计")
        layout.addWidget(self.result_tabs, 1)

        self.inference_group = QGroupBox("已加载模型的推理")
        inference_layout = QVBoxLayout(self.inference_group)
        model_path_row = QHBoxLayout()
        self.model_path_edit = QLineEdit()
        self.model_path_edit.setAccessibleName("待加载的 joblib 模型路径")
        self.model_path_edit.setPlaceholderText("选择导出的 model.joblib")
        self.model_browse_button = QPushButton("选择模型")
        self.reload_model_button = QPushButton("加载模型")
        model_path_row.addWidget(self.model_path_edit, 1)
        model_path_row.addWidget(self.model_browse_button)
        model_path_row.addWidget(self.reload_model_button)
        inference_layout.addLayout(model_path_row)
        self.model_info_label = QLabel("尚未加载模型；推理操作会依据已加载模型的实际能力启用。")
        self.model_info_label.setObjectName("helperText")
        self.model_info_label.setWordWrap(True)
        inference_layout.addWidget(self.model_info_label)

        inference_data_row = QHBoxLayout()
        self.inference_data_edit = QLineEdit()
        self.inference_data_edit.setAccessibleName("新数据文件路径")
        self.inference_data_edit.setPlaceholderText("选择要预测或变换的新表格")
        self.inference_data_browse_button = QPushButton("选择新数据")
        inference_data_row.addWidget(self.inference_data_edit, 1)
        inference_data_row.addWidget(self.inference_data_browse_button)
        inference_layout.addLayout(inference_data_row)
        inference_options_row = QHBoxLayout()
        self.inference_sheet_edit = QLineEdit()
        self.inference_sheet_edit.setAccessibleName("新数据工作表")
        self.inference_sheet_edit.setPlaceholderText("工作表名称（CSV 留空）")
        self.operation_combo = QComboBox()
        self.operation_combo.setAccessibleName("模型推理操作")
        self.operation_combo.setEnabled(False)
        self.inference_button = QPushButton("运行推理并导出 CSV")
        self.inference_button.setEnabled(False)
        self.inference_button.setMinimumHeight(34)
        inference_options_row.addWidget(QLabel("工作表"))
        inference_options_row.addWidget(self.inference_sheet_edit, 2)
        inference_options_row.addWidget(QLabel("操作"))
        inference_options_row.addWidget(self.operation_combo, 2)
        inference_options_row.addWidget(self.inference_button, 2)
        inference_layout.addLayout(inference_options_row)
        layout.addWidget(self.inference_group)
        return panel

    def _connect_ui(self) -> None:
        self.overall_guide_button.clicked.connect(self.show_overall_guide)
        self.batch_search_button.clicked.connect(self.open_batch_search)
        self.source_browse_button.clicked.connect(self._browse_source_file)
        self.load_data_button.clicked.connect(lambda: self.load_data(self.source_path_edit.text().strip()))
        self.source_path_edit.textChanged.connect(self._on_source_path_edited)
        self.sheet_combo.currentIndexChanged.connect(self._on_training_sheet_changed)
        self.target_combo.currentIndexChanged.connect(self._on_target_changed)
        self.feature_list.itemChanged.connect(self._on_feature_selection_changed)
        self.task_combo.currentIndexChanged.connect(self._on_task_changed)
        self.model_combo.currentIndexChanged.connect(self._on_model_changed)
        self.advanced_parameters.textChanged.connect(self._on_configuration_changed)
        self.sequence_config_edit.textChanged.connect(self._on_sequence_config_changed)
        self.seed_spin.textChanged.connect(self._on_configuration_changed)
        self.output_dir_edit.textChanged.connect(self._on_configuration_changed)
        self.output_browse_button.clicked.connect(self._browse_output_directory)
        self.run_button.clicked.connect(self.start_training)
        self.finalize_button.clicked.connect(self.finalize_training)
        self.cancel_button.clicked.connect(self.cancel_training)
        self.metric_combo.currentIndexChanged.connect(self._render_metric_chart)
        self.open_artifact_button.clicked.connect(self._open_selected_artifact)
        self.export_artifact_button.clicked.connect(self._export_selected_artifact)
        self.model_browse_button.clicked.connect(self._browse_model_file)
        self.model_path_edit.textChanged.connect(self._on_loaded_model_path_edited)
        self.reload_model_button.clicked.connect(self.reload_model)
        self.inference_data_browse_button.clicked.connect(self._browse_inference_file)
        self.inference_button.clicked.connect(self.start_inference)

    def copy_error_details(self) -> bool:
        if not self.last_error_id and not self.last_error_traceback and not self.last_error:
            return False
        lines = [f"error_id: {self.last_error_id or 'unavailable'}"]
        if self.last_error:
            lines.extend(("", self.last_error))
        if self.last_error_traceback:
            lines.extend(("", self.last_error_traceback.rstrip()))
        QApplication.clipboard().setText("\n".join(lines))
        self.result_status_label.setText("错误 ID 与 traceback 已复制到剪贴板。")
        return True

    def open_batch_search(self) -> None:
        """Open the durable multi-dataset search dialog without blocking this window."""
        if self._batch_dialog is not None:
            try:
                if self._batch_dialog.isVisible():
                    self._batch_dialog.raise_()
                    self._batch_dialog.activateWindow()
                    return
            except RuntimeError:
                self._batch_dialog = None

        from .batch_gui import BatchSearchDialog

        initial_config = None
        if self.dataset is not None:
            try:
                initial_config = self.build_experiment_config()
            except (TypeError, ValueError):
                pass
        dialog = BatchSearchDialog(self, initial_config=initial_config)
        dialog.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        dialog.finished.connect(self._batch_search_dialog_finished)
        self._batch_dialog = dialog
        dialog.show()

    def _batch_search_dialog_finished(self, _result: int) -> None:
        self._batch_dialog = None

    @staticmethod
    def _style_sheet() -> str:
        return """
        QWidget { background: #EFF6FF; color: #0F172A; font-family: "Microsoft YaHei UI"; font-size: 10pt; }
        QLabel#pageTitle { font-size: 20pt; font-weight: 650; color: #0F172A; }
        QLabel#pageSubtitle, QLabel#helperText { color: #475569; }
        QLabel#pageSubtitle { font-size: 10pt; }
        QLabel#stateBadge { background: #DBEAFE; color: #1D4ED8; border: 1px solid #93C5FD; border-radius: 9px; padding: 7px 12px; font-weight: 600; }
        QLabel#statusText { background: #FFFFFF; border: 1px solid #CBD5E1; border-left: 4px solid #2563EB; border-radius: 8px; padding: 10px 12px; }
        QGroupBox { background: #FFFFFF; border: 1px solid #CBD5E1; border-radius: 9px; margin-top: 12px; padding: 12px 10px 10px 10px; font-weight: 600; }
        QGroupBox::title { subcontrol-origin: margin; left: 11px; padding: 0 5px; color: #1E3A8A; }
        QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QPlainTextEdit, QListWidget, QTableView { background: #FFFFFF; border: 1px solid #94A3B8; border-radius: 5px; padding: 5px; selection-background-color: #BFDBFE; selection-color: #0F172A; }
        QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus, QPlainTextEdit:focus, QListWidget:focus, QTableView:focus { border: 2px solid #2563EB; }
        QPushButton { background: #FFFFFF; border: 1px solid #94A3B8; border-radius: 6px; padding: 7px 11px; font-weight: 550; }
        QPushButton:hover { background: #DBEAFE; border-color: #2563EB; }
        QPushButton:pressed { background: #BFDBFE; }
        QPushButton:disabled { background: #E2E8F0; color: #64748B; border-color: #CBD5E1; }
        QPushButton#primaryButton { background: #2563EB; color: #FFFFFF; border-color: #1D4ED8; font-weight: 650; }
        QPushButton#primaryButton:hover { background: #1D4ED8; }
        QHeaderView::section { background: #E2E8F0; color: #0F172A; padding: 6px; border: 0; border-bottom: 1px solid #94A3B8; font-weight: 600; }
        QTableView::item:alternate { background: #F8FAFC; }
        QTabWidget::pane { background: #FFFFFF; border: 1px solid #CBD5E1; border-radius: 6px; }
        QTabBar::tab { background: #E2E8F0; border: 1px solid #CBD5E1; padding: 8px 13px; margin-right: 3px; border-top-left-radius: 5px; border-top-right-radius: 5px; }
        QTabBar::tab:selected { background: #FFFFFF; color: #1D4ED8; font-weight: 650; }
        QProgressBar { border: 1px solid #CBD5E1; border-radius: 4px; background: #E2E8F0; min-height: 8px; max-height: 8px; }
        QProgressBar::chunk { background: #2563EB; border-radius: 4px; }
        """

    def _refresh_deferred_models(self) -> None:
        self.deferred_list.clear()
        items = [item for item in list_models(include_deferred=True) if item.get("implementation_status") != "available"]
        for model in items:
            entry = QListWidgetItem(f"{model['id']}  {model['display_name_zh']}  ·  未实现")
            entry.setFlags(Qt.ItemFlag.NoItemFlags)
            self.deferred_list.addItem(entry)
        self.deferred_box.setVisible(bool(items))

    def _refresh_model_choices(self) -> None:
        task = self.task_combo.currentData()
        models = list_models(task)
        self.model_combo.blockSignals(True)
        self.model_combo.clear()
        for model in models:
            self.model_combo.addItem(f"{model['id']}  {model['display_name_zh']}", model["id"])
        self.model_combo.blockSignals(False)
        self._refresh_parameters()
        self._update_sequence_config_state()

    def _clear_form_layout(self, layout: QFormLayout) -> None:
        while layout.count():
            item = layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def _refresh_parameters(self) -> None:
        was_initializing = self._params_initializing
        self._params_initializing = True
        try:
            self._clear_form_layout(self.param_form)
            self.common_parameter_widgets.clear()
            model_id = self.model_combo.currentData()
            if not model_id:
                return
            try:
                schema = {row["name"]: row for row in parameter_schema(model_id)}
                defaults = parameter_defaults(model_id)
                advanced_names = [name for name, spec in schema.items() if spec["group"] != "常用参数"]
                visible_advanced = advanced_names[:8]
                suffix = f" 等 {len(advanced_names)} 项" if len(advanced_names) > len(visible_advanced) else ""
                self.advanced_parameters_label.setText(
                    "高级参数（仅 JSON 对象；支持键：" + (", ".join(visible_advanced) + suffix if visible_advanced else "无") + "）"
                )
                self.advanced_parameters_label.setToolTip("可用高级参数：" + ", ".join(advanced_names))
                for name, spec in schema.items():
                    if spec["group"] != "常用参数":
                        continue
                    default = defaults.get(name)
                    control, kind = self._parameter_control(name, default)
                    self.param_form.addRow(spec["label_zh"], control)
                    self.common_parameter_widgets[name] = (control, kind)
                    if isinstance(control, QCheckBox):
                        control.toggled.connect(
                            lambda _checked, field=name: self._on_parameter_changed(field)
                        )
                    elif isinstance(control, QLineEdit):
                        control.textChanged.connect(
                            lambda _text, field=name: self._on_parameter_changed(field)
                        )
            except Exception as exc:
                warning = QLabel(f"读取模型参数失败：{exc}")
                warning.setWordWrap(True)
                self.param_form.addRow(warning)
        finally:
            self._params_initializing = was_initializing

    @staticmethod
    def _parameter_control(name: str, value: Any) -> tuple[QWidget, str]:
        if isinstance(value, bool):
            control = QCheckBox()
            control.setChecked(value)
            control.setAccessibleName(f"模型参数 {name}")
            return control, "bool"
        if isinstance(value, int):
            control = _IntegerLineEdit(value, allow_none=True)
            control.setPlaceholderText("整数或 null；留空使用模型默认值")
            control.setAccessibleName(f"模型参数 {name}")
            return control, "int"
        if isinstance(value, float):
            control = QLineEdit(repr(value))
            control.setPlaceholderText("实数，支持科学计数法，例如 1e-10")
            control.setAccessibleName(f"模型参数 {name}")
            return control, "float"
        if value is None:
            control = QLineEdit()
            control.setPlaceholderText("留空使用模型默认值；或输入 JSON 值")
            control.setAccessibleName(f"模型参数 {name}")
            return control, "optional"
        control = QLineEdit(str(value))
        control.setAccessibleName(f"模型参数 {name}")
        return control, "str"

    def _refresh_target_and_features(self) -> None:
        self.target_combo.blockSignals(True)
        self.target_combo.clear()
        self.target_combo.addItem("（不使用目标列）", None)
        columns = list(self.dataset.columns) if self.dataset is not None else []
        for column in columns:
            self.target_combo.addItem(str(column), str(column))
        self.target_combo.setCurrentIndex(0)
        self.target_combo.blockSignals(False)

        self.feature_list.blockSignals(True)
        self.feature_list.clear()
        for column in columns:
            item = QListWidgetItem(str(column))
            item.setFlags(item.flags() | Qt.ItemFlag.ItemIsUserCheckable)
            item.setCheckState(Qt.CheckState.Checked)
            self.feature_list.addItem(item)
        self.feature_list.blockSignals(False)
        self._update_target_state()

    def _update_target_state(self) -> None:
        supervised = self.task_combo.currentData() in {"classification", "regression"}
        self.target_combo.setEnabled(supervised and self.dataset is not None)
        target = self.target_combo.currentData() if supervised else None
        structural = self._configured_sequence_structure_columns()
        self.feature_list.blockSignals(True)
        for index in range(self.feature_list.count()):
            item = self.feature_list.item(index)
            is_excluded = item.text() == target or item.text() in structural
            flags = item.flags() | Qt.ItemFlag.ItemIsUserCheckable
            if is_excluded:
                item.setFlags(flags & ~Qt.ItemFlag.ItemIsEnabled)
                item.setCheckState(Qt.CheckState.Unchecked)
            else:
                item.setFlags(flags | Qt.ItemFlag.ItemIsEnabled)
        self.feature_list.blockSignals(False)

    def _configured_sequence_structure_columns(self) -> set[str]:
        if self.model_combo.currentData() not in {"H01", "H02", "H03", "N04", "N06"}:
            return set()
        text = self.sequence_config_edit.toPlainText().strip() or "{}"
        try:
            payload = json.loads(text)
            if not isinstance(payload, dict):
                return set()
            sequence = SequenceConfig.from_dict(payload)
        except (json.JSONDecodeError, TypeError, ValueError):
            return set()
        return {
            value for value in (sequence.group_column, sequence.time_column)
            if isinstance(value, str) and value.strip()
        }

    def selected_feature_columns(self) -> list[str]:
        return [
            self.feature_list.item(index).text()
            for index in range(self.feature_list.count())
            if self.feature_list.item(index).checkState() == Qt.CheckState.Checked
        ]

    def selected_target_column(self) -> str | None:
        if self.task_combo.currentData() not in {"classification", "regression"}:
            return None
        value = self.target_combo.currentData()
        return str(value) if value is not None else None

    def load_data(
        self,
        path: str | Path,
        sheet_name: str | None = None,
        *,
        prepared_dataset: LoadedDataset | None = None,
        invalidate_result: bool = True,
    ) -> bool:
        """Load a complete source table and a bounded preview; return success."""
        if self.process.state() != QProcess.ProcessState.NotRunning:
            self.result_status_label.setText("后台任务运行期间不能更换训练数据。")
            return False
        path_text = str(path).strip()
        if not path_text:
            self.result_status_label.setText("请先选择 CSV、XLSX 或 XLS 数据文件。")
            return False
        try:
            dataset = (
                prepared_dataset
                if prepared_dataset is not None
                else load_dataset(path_text, sheet_name=sheet_name)
            )
        except Exception as exc:
            self.dataset = None
            self.data_overview_action.setEnabled(False)
            self.basic_plot_action.setEnabled(False)
            self._feature_selection_unspecified = False
            self.preview_table.model().set_frame(None)
            self._refresh_target_and_features()
            self.data_info_label.setText(f"数据加载失败：{exc}")
            self.data_info_label.setStyleSheet("color: #B91C1C;")
            if invalidate_result:
                self._invalidate_result("数据加载失败")
            self._update_worker_controls()
            return False

        self.dataset = dataset
        self.data_overview_action.setEnabled(True)
        self.basic_plot_action.setEnabled(True)
        self._feature_selection_unspecified = False
        self.source_path_edit.blockSignals(True)
        self.source_path_edit.setText(dataset.source_path)
        self.source_path_edit.blockSignals(False)
        self.sheet_combo.blockSignals(True)
        self.sheet_combo.clear()
        if dataset.sheet_names:
            self.sheet_combo.addItems(list(dataset.sheet_names))
            if dataset.selected_sheet is not None:
                self.sheet_combo.setCurrentText(dataset.selected_sheet)
            self.sheet_combo.setEnabled(True)
        else:
            self.sheet_combo.addItem("（CSV）")
            self.sheet_combo.setEnabled(False)
        self.sheet_combo.blockSignals(False)
        self.preview_table.model().set_frame(dataset.preview(30))
        self._refresh_target_and_features()
        sheet_text = f" · 工作表 {dataset.selected_sheet}" if dataset.selected_sheet else ""
        self.data_info_label.setStyleSheet("")
        self.data_info_label.setText(f"{len(dataset.frame):,} 行 × {len(dataset.frame.columns)} 列{sheet_text}；下方显示前 30 行。")
        if not self.output_dir_edit.text().strip():
            self.output_dir_edit.setText(self.default_output_directory(dataset.source_path))
        self.state_label.setText("数据已加载")
        self.result_status_label.setText("数据已就绪。请检查预览，明确选择特征列和监督任务的目标列。")
        if invalidate_result:
            self._invalidate_result("训练数据已更换")
        self._update_worker_controls()
        try:
            self.preferences = self.preference_store.record_recent_file(
                dataset.source_path
            )
        except PreferenceError as exc:
            self.result_status_label.setText(
                f"数据已就绪；无法更新最近文件记录：{exc}"
            )
        else:
            self._refresh_recent_files_menu()
        return True

    def show_data_overview(self) -> DataOverviewDialog | None:
        """Show full-table statistics for the already-loaded in-memory frame."""
        if self.dataset is None:
            self.result_status_label.setText("请先加载数据，再查看完整数据概况。")
            return None
        dialog = DataOverviewDialog(self.dataset.frame, self)
        self._data_overview_dialog = dialog
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()
        return dialog

    def open_plot_selector(self):
        """Open the source/partition/column chooser, then a non-modal plot window."""
        from .plot_cache import loaded_dataset_plot_source
        from .plot_dialog import PlotDialog
        from .plotting import build_plot_payload

        sources = []
        unavailable = {}
        if self.dataset is not None:
            try:
                sources.append(loaded_dataset_plot_source(self.dataset))
            except Exception as exc:
                unavailable["loaded dataset"] = str(exc)
        sources.extend(self._session_plot_sources.values())
        for partition in ("train", "validation", "test"):
            if partition not in self._session_plot_sources and self.bound_result is not None:
                unavailable.setdefault(partition, f"本次 session 没有可用的 {partition} 缓存。")
        if not sources:
            self.result_status_label.setText("没有可用于绘图的已加载数据或本次任务缓存。")
            return None
        chooser = PlotSourceChooserDialog(sources, unavailable=unavailable, parent=self)
        if chooser.exec() != QDialog.DialogCode.Accepted:
            return None
        try:
            payload = build_plot_payload(chooser.selected_source, chooser.selected_spec)
            dialog = PlotDialog([payload], parent=self)
        except Exception as exc:
            self.result_status_label.setText(f"无法生成所选图表：{exc}")
            return None
        self._plot_dialogs.append(dialog)
        dialog.destroyed.connect(
            lambda _obj=None, target=dialog: self._plot_dialogs.remove(target)
            if target in self._plot_dialogs else None
        )
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()
        return dialog

    def _capture_session_plot_sources(self) -> None:
        if self._session_file_path is None or self._session_id is None or self._prepared_signature is None:
            self._session_plot_sources = {}
            return
        try:
            from .plot_cache import load_single_session_plot_sources

            collection = load_single_session_plot_sources(
                self._session_file_path,
                session_id=self._session_id,
                frozen_config_sha256=self._prepared_signature,
            )
            self._session_plot_sources = collection.by_partition()
            for partition, reason in collection.unavailable:
                self._append_log(f"结果图 {partition} 分区不可用：{reason}")
        except Exception as exc:
            self._session_plot_sources = {}
            self._append_log(f"未能在清理 session 前复制结果图缓存：{exc}")

    @staticmethod
    def default_output_directory(source_path: str | Path) -> str:
        source = Path(source_path).expanduser().resolve()
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        return str(source.parent / f"{source.stem}_pyml_{stamp}")

    def build_experiment_config(self) -> ExperimentConfig:
        """Read current widgets as a validated immutable run configuration."""
        if self.dataset is None:
            raise ValueError("请先加载训练数据")
        entered_path = Path(self.source_path_edit.text().strip()).expanduser().resolve()
        if entered_path != Path(self.dataset.source_path).resolve():
            raise ValueError("训练数据路径已更改，请点击“加载预览”读取该文件")
        task = self.task_combo.currentData()
        model_id = self.model_combo.currentData()
        if not model_id:
            raise ValueError("当前任务没有可运行的活动模型")
        self._update_target_state()
        features = self.selected_feature_columns()
        if not features:
            raise ValueError("请至少勾选一个特征列")
        target = self.selected_target_column()
        parameters = self._collect_parameters()
        sequence = None
        if model_id in {"H01", "H02", "H03", "N04", "N06"}:
            sequence_text = self.sequence_config_edit.toPlainText().strip() or "{}"
            try:
                sequence_value = json.loads(sequence_text)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"序列配置不是有效 JSON：第 {exc.lineno} 行，第 {exc.colno} 列：{exc.msg}"
                ) from exc
            if not isinstance(sequence_value, dict):
                raise ValueError("序列配置必须是 JSON 对象")
            sequence = SequenceConfig.from_dict(sequence_value) if sequence_value else None
        output_text = self.output_dir_edit.text().strip()
        output_dir = str(Path(output_text).expanduser().resolve()) if output_text else None
        source_path = Path(self.dataset.source_path).resolve()
        if output_dir is not None:
            output_path = Path(output_dir)
            if output_path == source_path or output_path == source_path.parent:
                raise ValueError("结果目录不能覆盖输入文件或其父目录")
            if output_path.exists() and (not output_path.is_dir() or any(output_path.iterdir())):
                raise ValueError("结果目录必须不存在或为空；为避免覆盖，请选择一个新目录")
        config = ExperimentConfig(
            dataset=DatasetConfig(
                source_path=self.dataset.source_path,
                sheet_name=self.dataset.selected_sheet,
                target_column=target,
                feature_columns=(
                    None if self._feature_selection_unspecified else tuple(features)
                ),
            ),
            task=str(task),
            model_id=str(model_id),
            parameters=parameters,
            split=SplitConfig(
                seed=self.seed_spin.value(),
                train_fraction=self._split_fractions.train_fraction,
                validation_fraction=self._split_fractions.validation_fraction,
                test_fraction=self._split_fractions.test_fraction,
            ),
            output_dir=output_dir,
            sequence=sequence,
        )
        config.validate()
        return config

    def _collect_parameters(self) -> dict[str, Any]:
        advanced_text = self.advanced_parameters.toPlainText().strip() or "{}"
        try:
            advanced = json.loads(advanced_text)
        except json.JSONDecodeError as exc:
            raise ValueError(f"高级参数不是有效 JSON：第 {exc.lineno} 行，第 {exc.colno} 列：{exc.msg}") from exc
        if not isinstance(advanced, dict):
            raise ValueError("高级参数必须是 JSON 对象")

        if self._loaded_parameters is None:
            values: dict[str, Any] = {}
            for name, (control, kind) in self.common_parameter_widgets.items():
                value = self._read_common_parameter(name, control, kind)
                if value is not _NO_PARAMETER_OVERRIDE:
                    values[name] = value
            duplicates = sorted(set(values).intersection(advanced))
            if duplicates:
                raise ValueError(f"参数同时出现在常用参数和高级 JSON 中：{duplicates}")
            values.update(advanced)
            return values

        values = dict(self._loaded_parameters)
        advanced_changed = advanced_text != self._loaded_advanced_text
        if advanced_changed:
            values = {
                name: value for name, value in values.items()
                if name not in self._loaded_advanced_names
            }
        for name in self._edited_parameter_names:
            field = self.common_parameter_widgets.get(name)
            if field is None:
                continue
            control, kind = field
            value = self._read_common_parameter(name, control, kind)
            if value is _NO_PARAMETER_OVERRIDE:
                values.pop(name, None)
            else:
                values[name] = value
        if advanced_changed:
            duplicates = sorted(set(values).intersection(advanced))
            if duplicates:
                raise ValueError(f"参数同时出现在常用参数和高级 JSON 中：{duplicates}")
            values.update(advanced)
        return values

    @staticmethod
    def _read_common_parameter(name: str, control: QWidget, kind: str) -> Any:
        if kind == "bool":
            return control.isChecked()  # type: ignore[union-attr]
        if kind == "int":
            text = control.text().strip()  # type: ignore[union-attr]
            if not text:
                return _NO_PARAMETER_OVERRIDE
            try:
                return control.value()  # type: ignore[union-attr]
            except (TypeError, ValueError) as exc:
                raise ValueError(f"参数 {name} 必须是整数或 null") from exc
        if kind == "float":
            text = control.text().strip()  # type: ignore[union-attr]
            try:
                value = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"参数 {name} 必须是十进制或科学计数法实数") from exc
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"参数 {name} 必须是十进制或科学计数法实数")
            value = float(value)
            if not math.isfinite(value):
                raise ValueError(f"参数 {name} 必须是有限实数")
            return value
        if kind == "optional":
            text = control.text().strip()  # type: ignore[union-attr]
            if not text:
                return _NO_PARAMETER_OVERRIDE
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return text
        return control.text()  # type: ignore[union-attr]

    @staticmethod
    def _config_sha256(config: ExperimentConfig) -> str:
        encoded = json.dumps(config.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

    def start_training(self) -> bool:
        """Start a non-blocking worker action that fits and persists validation state."""
        try:
            config = self.build_experiment_config()
        except Exception as exc:
            self.last_error = str(exc)
            self.result_status_label.setText(f"配置错误：{exc}")
            self.state_label.setText("配置未通过")
            return False
        if config.output_dir is None:
            self.result_status_label.setText("请选择实验结果目录后再启动训练。")
            self.state_label.setText("配置未通过")
            return False
        self._discard_session(remove_config=True)
        self._clear_bound_result()
        config_path = self._write_temp_config(config.to_dict())
        session_path = Path(tempfile.gettempdir()) / f"pyml_session_{uuid.uuid4().hex}.joblib"
        self._clear_bound_result()
        self.received_events.clear()
        self.events_text.clear()
        self.stderr_text = ""
        self.last_error = None
        self._config_file_path = config_path
        self._session_file_path = session_path
        self._run_signature = self._config_sha256(config)
        self._prepared_signature = None
        self._prepared = False
        self._frozen = False
        self._finalize_requested = False
        self._cancel_requested = False
        self.state_label.setText("正在拟合并评估验证集")
        self.result_status_label.setText("后台进程正在训练；窗口保持响应。最终测试集此阶段不会被读取。")
        self._set_worker_busy(True)
        started = self._start_worker(
            ["train", "--config", str(config_path), "--session", str(session_path)], action="train"
        )
        if not started:
            self._set_worker_busy(False)
            self._discard_session(remove_config=True)
            return False
        return True

    def finalize_training(self) -> bool:
        """Advance the saved session through freeze, then through its one test action."""
        if not self._session_is_current() or self._worker_action is not None or self._finalize_requested:
            return False
        if self._test_result_payload is not None:
            self._finalize_requested = True
            self.result_status_label.setText("测试结果已缓存，正在从同一 session 导出产物。")
            return self._start_session_action("export")
        if not self._frozen:
            if not self._validate_session_configuration():
                return False
            self._finalize_requested = True
            self.result_status_label.setText("正在冻结已验证的配置；最终测试集尚未读取。")
            self.state_label.setText("正在冻结配置")
            return self._start_session_action("freeze")
        if not self._validate_session_configuration():
            return False
        self._finalize_requested = True
        self.finalize_button.setEnabled(False)
        self.result_status_label.setText("正在使用同一训练会话计算最终测试集；该步骤只会执行一次。")
        self.state_label.setText("最终测试进行中")
        return self._start_session_action("test")

    def _session_is_current(self) -> bool:
        return bool(self._session_id and self._session_file_path and self._config_file_path and self._prepared)

    def _validate_session_configuration(self) -> bool:
        if not self._session_is_current():
            self.result_status_label.setText("没有可继续的验证 session，请重新训练并验证。")
            return False
        try:
            config = self.build_experiment_config()
            signature = self._config_sha256(config)
        except Exception as exc:
            self.result_status_label.setText(f"session 配置无法复核：{exc}")
            return False
        if signature != self._run_signature or signature != self._prepared_signature:
            self._invalidate_result("配置与已训练 session 不一致")
            return False
        return True

    def _start_session_action(self, action: str) -> bool:
        if not self._session_id or not self._session_file_path or not self._config_file_path:
            return False
        args = [action, "--config", str(self._config_file_path), "--session", str(self._session_file_path),
                "--session-id", self._session_id]
        if action == "export":
            output_dir = str(Path(self.output_dir_edit.text().strip()).expanduser().resolve())
            args.extend(["--output", output_dir])
        self._set_worker_busy(True)
        if not self._start_worker(args, action=action):
            self._set_worker_busy(False)
            return False
        return True

    def cancel_training(self) -> bool:
        if self._worker_action is None or self.process.state() == QProcess.ProcessState.NotRunning:
            if self._session_id is not None:
                self._discard_session(remove_config=True)
                self._clear_bound_result()
                self.result_status_label.setText("验证 session 已丢弃，最终测试未运行。")
                self.state_label.setText("session 已取消")
                self._update_worker_controls()
                return True
            return False
        self._cancel_requested = True
        self.finalize_button.setEnabled(False)
        self.cancel_button.setEnabled(False)
        self.process.terminate()
        QTimer.singleShot(1800, self._kill_cancelled_worker_if_running)
        self.result_status_label.setText("正在停止后台动作；尚未完成的最终测试结果不会绑定。")
        return True

    def _kill_cancelled_worker_if_running(self) -> None:
        if self._cancel_requested and self.process.state() != QProcess.ProcessState.NotRunning:
            self.process.kill()

    @staticmethod
    def _write_temp_config(payload: dict[str, Any]) -> Path:
        handle = tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", suffix=".json", prefix="pyml_config_", delete=False)
        path = Path(handle.name)
        try:
            json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        finally:
            handle.close()
        return path

    def _start_worker(self, arguments: list[str], *, action: str) -> bool:
        if self.process.state() != QProcess.ProcessState.NotRunning:
            self.last_error = "已有后台任务运行"
            return False
        self._worker_action = action
        self._worker_output_buffer = b""
        self.stderr_text = ""
        self._stderr_decoder = codecs.getincrementaldecoder("utf-8")("replace")
        self.last_error = None
        self.last_error_id = None
        self.last_error_traceback = ""
        self.copy_error_button.setEnabled(False)
        self.process.setProcessEnvironment(self._worker_process_environment())
        self.process.setWorkingDirectory(str(Path.cwd()))
        executable, worker_arguments = worker_command(arguments)
        self.process.start(executable, worker_arguments)
        return True

    def _worker_process_environment(self) -> QProcessEnvironment:
        environment = QProcessEnvironment.systemEnvironment()
        controller_root = getattr(self.logging_controller, "log_root", None)
        log_root = controller_root or self.preferences.log_directory
        if log_root:
            environment.insert("PYML_LOG_ROOT", str(log_root))
        environment.insert("PYML_LOG_SESSION_ID", uuid.uuid4().hex)
        return environment

    def _read_worker_stdout(self) -> None:
        self._worker_output_buffer += bytes(self.process.readAllStandardOutput())
        while b"\n" in self._worker_output_buffer:
            line, self._worker_output_buffer = self._worker_output_buffer.split(b"\n", 1)
            if not line.strip():
                continue
            try:
                payload = json.loads(line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                self._append_log(f"无法解析 worker 输出：{line.decode('utf-8', errors='replace')} ({exc})")
                continue
            self._handle_worker_payload(payload)

    def _read_worker_stderr(self) -> None:
        decoder = self._stderr_decoder
        if decoder is None:
            decoder = codecs.getincrementaldecoder("utf-8")("replace")
            self._stderr_decoder = decoder
        chunk = decoder.decode(bytes(self.process.readAllStandardError()), final=False)
        self._append_worker_stderr(chunk)

    def _append_worker_stderr(self, chunk: str) -> None:
        if chunk:
            self.stderr_text += chunk
            for line in chunk.splitlines():
                self._append_log(f"stderr · {line}")

    def _flush_worker_stderr(self) -> None:
        decoder, self._stderr_decoder = self._stderr_decoder, None
        if decoder is not None:
            self._append_worker_stderr(decoder.decode(b"", final=True))
        if self.last_error_id and not self.last_error_traceback:
            self.last_error_traceback = self.stderr_text.strip()
        self.copy_error_button.setEnabled(
            bool(self.last_error_id or self.last_error_traceback or self.last_error)
        )

    def _handle_worker_payload(self, payload: dict[str, Any]) -> None:
        if not isinstance(payload, dict):
            self._append_log("无法解析 worker 输出：JSON 顶层必须是对象。")
            return
        kind = payload.get("type")
        if kind not in {"progress", "result", "error"}:
            self._append_log("无法解析 worker 输出：消息类型无效。")
            return
        if self._cancel_requested and kind in {"result", "error"}:
            self._append_log("后台动作正在取消；忽略尚未完成的结果或错误消息。")
            return
        action_value = payload.get("action")
        if action_value is not None and not isinstance(action_value, str):
            self._append_log("无法解析 worker 输出：action 必须是文本。")
            return
        if kind == "progress" and payload.get("data") is not None and not isinstance(payload["data"], dict):
            self._append_log("无法解析 worker 输出：progress.data 必须是对象。")
            return
        if kind == "error":
            if not isinstance(payload.get("error_type"), str) or not isinstance(payload.get("message"), str):
                self._append_log("无法解析 worker 输出：error 消息缺少有效类型或说明。")
                return
            if payload.get("traceback") is not None and not isinstance(payload.get("traceback"), str):
                self._append_log("无法解析 worker 输出：traceback 必须是文本。")
                return
        if kind == "result":
            for field_name in ("metrics", "audit", "capabilities", "artifact_paths"):
                if field_name in payload and not isinstance(payload[field_name], dict):
                    pretest_audit = (
                        field_name == "audit"
                        and payload[field_name] is None
                        and (payload.get("action") or self._worker_action, payload.get("state"))
                        in (("train", "trained"), ("freeze", "frozen"))
                        and type(payload.get("test_evaluation_count")) is int
                        and payload["test_evaluation_count"] == 0
                    )
                    if pretest_audit:
                        continue
                    self._append_log(f"无法解析 worker 输出：result.{field_name} 必须是对象。")
                    return
            if "feature_columns" in payload and not isinstance(payload["feature_columns"], list):
                self._append_log("无法解析 worker 输出：result.feature_columns 必须是数组。")
                return
        action = str(payload.get("action") or self._worker_action or "")
        if kind == "progress":
            operation = str(payload.get("name") or payload.get("stage") or payload.get("operation") or "")
            message = payload.get("message") or EVENT_LABELS.get(operation) or operation or "后台阶段"
            self._append_log(str(message))
            if operation in EVENT_LABELS:
                self.received_events.append(operation)
            if action == "train":
                self.result_status_label.setText(str(message))
        elif kind == "result":
            state = str(payload.get("state") or "")
            if action == "train" or (not action and state == "trained"):
                self._apply_train_result(payload)
            elif action == "freeze" or (not action and state == "frozen"):
                self._apply_freeze_result(payload)
            elif action == "export" or (not action and payload.get("artifact_paths")):
                self._apply_export_result(payload)
            elif action == "test" or (not action and state == "tested"):
                self._apply_test_result(payload)
            elif action == "inspect":
                self._apply_model_info(
                    str(payload.get("model_id", "unknown")),
                    str(self.model_path_edit.text().strip()),
                    dict(payload.get("capabilities", {})),
                    payload.get("feature_columns", []),
                    payload.get("frozen_config_sha256"),
                )
                self.state_label.setText("模型已加载")
                self.result_status_label.setText("模型已从 joblib 文件重新加载；操作列表按其实测能力生成。")
            elif action == "inference":
                self._apply_inference_result(payload)
            else:
                self._append_log(f"worker result：{json.dumps(payload, ensure_ascii=False)}")
        elif kind == "error":
            message = f"{payload.get('error_type', 'WorkerError')}: {payload.get('message', '')}"
            self.last_error = message
            candidate_id = payload.get("error_id")
            self.last_error_id = (
                candidate_id
                if isinstance(candidate_id, str) and _WORKER_ERROR_ID_RE.fullmatch(candidate_id)
                else None
            )
            self.last_error_traceback = str(payload.get("traceback") or "")
            self.copy_error_button.setEnabled(
                bool(self.last_error_id or self.last_error_traceback or self.last_error)
            )
            error_id_text = f" · error_id={self.last_error_id}" if self.last_error_id else ""
            self.result_status_label.setText(f"后台任务失败：{message}")
            self.state_label.setText("任务失败")
            self._append_log(f"错误{error_id_text} · {message}")
            if action == "train":
                self._discard_session(remove_config=True)
            elif action == "freeze":
                self._finalize_requested = False
                self.finalize_button.setText("冻结模型设置")
            elif action == "test":
                self._finalize_requested = False
                self.finalize_button.setText("重试最终测试（同一 session）")
            elif action == "export":
                self._finalize_requested = False
                self.finalize_button.setText("重试导出")
        else:
            self._append_log(f"worker 消息：{json.dumps(payload, ensure_ascii=False)}")

    def _apply_train_result(self, payload: dict[str, Any]) -> None:
        session_id = str(payload.get("session_id", ""))
        session_path = str(payload.get("session_path", ""))
        signature = str(payload.get("frozen_config_sha256", ""))
        state = str(payload.get("state", ""))
        if state != "trained" or not session_id or not session_path or not self._session_file_path:
            self.last_error = "训练 worker 未返回有效的持久 session 标识"
            self.result_status_label.setText(self.last_error)
            self.state_label.setText("session 无效")
            return
        if Path(session_path).expanduser().resolve() != self._session_file_path.expanduser().resolve():
            self.last_error = "worker 返回的 session 路径与本次请求不一致"
            self.result_status_label.setText(self.last_error)
            self.state_label.setText("session 路径不一致")
            self._discard_session(remove_config=True)
            return
        if signature != self._run_signature:
            self.last_error = "冻结配置指纹与启动时的配置不一致"
            self.result_status_label.setText(self.last_error)
            self.state_label.setText("session 配置不一致")
            self._discard_session(remove_config=True)
            return
        metrics = payload.get("metrics", {})
        if "test" in metrics or payload.get("test_evaluation_count", 0) != 0:
            self.last_error = "训练阶段提前访问了测试集；该 session 未接受"
            self.result_status_label.setText(self.last_error)
            self.state_label.setText("阶段契约失败")
            self._discard_session(remove_config=True)
            return
        self._session_id = session_id
        self._session_file_path = Path(session_path).expanduser().resolve()
        self._prepared = True
        self._frozen = False
        self._prepared_signature = signature
        self._session_capabilities = dict(payload.get("capabilities", {}))
        self.received_events = [str(item) for item in payload.get("events", self.received_events)]
        self.current_metrics = dict(metrics)
        self.training_curves = payload.get("training_curves")
        self._render_sequence_summary(payload.get("sequence_summary"))
        self._render_metrics()
        self._render_training_curves()
        self.finalize_button.setText("冻结模型设置")
        self.result_status_label.setText("训练与验证已完成。模型尚未冻结，最终测试集尚未读取。")
        self.state_label.setText("验证完成 · 等待冻结")
        self._append_log(f"持久验证 session 已保存：{self._session_file_path.name}；session_id={session_id}")

    def _apply_freeze_result(self, payload: dict[str, Any]) -> None:
        if (str(payload.get("state", "")) != "frozen"
                or str(payload.get("session_id", self._session_id)) != self._session_id
                or str(Path(str(payload.get("session_path", self._session_file_path))).resolve()) != str(self._session_file_path)
                or str(payload.get("frozen_config_sha256", self._prepared_signature)) != self._prepared_signature):
            self.last_error = "冻结响应没有确认当前验证 session"
            self.result_status_label.setText(self.last_error)
            self._finalize_requested = False
            return
        self._frozen = True
        self._finalize_requested = False
        self.finalize_button.setText("执行最终测试（一次）")
        self.result_status_label.setText("模型与参数已冻结；测试集仍未读取。确认后可执行一次最终测试。")
        self.state_label.setText("配置已冻结 · 等待最终测试")
        self._append_log("worker 确认冻结同一 session；最终测试等待用户操作。")

    def _apply_test_result(self, payload: dict[str, Any]) -> None:
        signature = str(payload.get("frozen_config_sha256", self._prepared_signature or ""))
        response_path = payload.get("session_path")
        try:
            path_matches = (
                self._session_file_path is not None
                and response_path is not None
                and Path(str(response_path)).expanduser().resolve() == self._session_file_path
            )
        except (OSError, TypeError, ValueError):
            path_matches = False
        if (not self._frozen or signature != self._prepared_signature
                or str(payload.get("session_id", self._session_id)) != self._session_id
                or not path_matches
                or str(payload.get("state", "")) != "tested"):
            self.last_error = "最终测试结果不属于当前已冻结的验证 session"
            self.result_status_label.setText(self.last_error)
            self.state_label.setText("结果未绑定")
            return
        audit = payload.get("audit", {})
        events = audit.get("events") if isinstance(audit, dict) else None
        if not isinstance(events, list):
            events = []
        events = [str(item) for item in events]
        freeze_indices = [index for index, item in enumerate(events) if item == "user_selection_frozen"]
        test_events = [(index, item) for index, item in enumerate(events) if item.startswith("test_")]
        if len(freeze_indices) != 1 or len(test_events) != 1 or test_events[0][0] <= freeze_indices[0]:
            self.last_error = "worker 审计没有证明用户冻结发生在最终测试之前"
            self.result_status_label.setText(self.last_error)
            self.state_label.setText("审计校验失败")
            return
        payload_count = payload.get("test_evaluation_count")
        audit_count = audit.get("test_evaluation_count") if isinstance(audit, dict) else None
        if (type(payload_count) is not int or type(audit_count) is not int
                or payload_count != audit_count):
            self.last_error = "worker 没有提供一致的最终测试评估计数"
            self.result_status_label.setText(self.last_error)
            self.state_label.setText("审计校验失败")
            return

        metrics = payload.get("metrics", {})
        test_metrics = metrics.get("test", {}) if isinstance(metrics, dict) else {}
        test_status = test_metrics.get("status") if isinstance(test_metrics, dict) else None
        test_event = test_events[0][1]
        guarded_capability = None
        task = self.task_combo.currentData()
        if task == "dimensionality reduction":
            guarded_capability = "transform"
        elif task in {"classification", "regression", "clustering", "anomaly detection", "sequence_modeling"}:
            guarded_capability = "predict"
        bound_capabilities = self._session_capabilities
        response_capabilities = payload.get("capabilities")
        response_capability_matches = (
            response_capabilities is None
            or (
                guarded_capability is not None
                and isinstance(response_capabilities, dict)
                and isinstance(bound_capabilities, dict)
                and response_capabilities.get(guarded_capability)
                == bound_capabilities.get(guarded_capability)
            )
        )
        bound_support = (
            bound_capabilities.get(guarded_capability)
            if guarded_capability is not None and isinstance(bound_capabilities, dict)
            else None
        )
        valid_guard = (
            payload_count == 0
            and test_event == "test_not_evaluated_capability_guard"
            and test_status == "not_supported"
            and guarded_capability is not None
            and bound_support is False
            and response_capability_matches
        )
        valid_scored_test = (
            payload_count == 1
            and test_event == "test_metrics_computed_once"
            and test_status != "not_supported"
            and bound_support is True
            and response_capability_matches
        )
        if not (valid_guard or valid_scored_test):
            self.last_error = "worker 审计未证明最终测试恰好评分一次，或由能力保护跳过"
            self.result_status_label.setText(self.last_error)
            self.state_label.setText("审计校验失败")
            return
        self._test_result_payload = dict(payload)
        if payload.get("training_curves") is not None:
            self.training_curves = payload.get("training_curves")
        self.current_metrics = dict(payload.get("metrics", self.current_metrics))
        self._render_metrics()
        self._render_training_curves()
        self._render_audit(payload.get("audit", {}))
        self._append_log(f"最终测试响应已缓存（cached={payload.get('cached', False)}）；准备导出产物。")
        self.finalize_button.setEnabled(False)
        self.result_status_label.setText("最终测试完成，指标已缓存；正在将同一 session 的产物导出。")
        self.state_label.setText("最终测试完成 · 导出中")

    def _apply_export_result(self, payload: dict[str, Any]) -> None:
        if self._test_result_payload is None:
            self.last_error = "收到导出结果，但当前 GUI 没有已缓存的最终测试结果"
            self.result_status_label.setText(self.last_error)
            return
        combined = dict(self._test_result_payload)
        combined.update(payload)
        combined["metrics"] = self._test_result_payload.get("metrics", {})
        combined["audit"] = self._test_result_payload.get("audit", {})
        combined["frozen_config_sha256"] = str(
            payload.get("frozen_config_sha256", self._prepared_signature or "")
        )
        combined["capabilities"] = dict(payload.get("capabilities", self._session_capabilities))
        combined["artifact_paths"] = dict(payload.get("artifact_paths", {}))
        self._apply_training_result(combined)
        if self.bound_result is not None:
            self._capture_session_plot_sources()
        self._discard_session(remove_config=True, preserve_result=True)
        self._update_worker_controls()

    def _apply_inference_result(self, payload: dict[str, Any]) -> None:
        path = str(payload.get("path", ""))
        operation = str(payload.get("operation", ""))
        if path:
            self._add_artifact("推理结果 CSV", path)
        self.result_status_label.setText(
            f"推理完成：{payload.get('rows', 0):,} 行，操作 {OPERATION_LABELS.get(operation, operation)}。"
        )
        self.state_label.setText("推理完成")

    def _apply_training_result(self, payload: dict[str, Any]) -> None:
        signature = str(payload.get("frozen_config_sha256", ""))
        if not self._prepared or signature != self._prepared_signature or signature != self._run_signature:
            self.last_error = "最终结果与此前冻结的验证会话不一致，结果未绑定到当前配置"
            self.result_status_label.setText(self.last_error)
            self.state_label.setText("结果未绑定")
            return
        self.current_metrics = dict(payload.get("metrics", {}))
        self.training_curves = payload.get("training_curves")
        self.artifact_paths = {str(k): str(v) for k, v in payload.get("artifact_paths", {}).items()}
        self.bound_result = {
            "frozen_config_sha256": signature,
            "metrics": self.current_metrics,
            "audit": payload.get("audit", {}),
            "artifact_paths": self.artifact_paths,
        }
        self._render_metrics()
        self._render_training_curves()
        self._render_audit(payload.get("audit", {}))
        self._populate_artifacts(self.artifact_paths)
        model_path = self.artifact_paths.get("model")
        if model_path:
            model_id = str(self.model_combo.currentData() or "unknown")
            self._apply_model_info(model_id, model_path, dict(payload.get("capabilities", {})), [], signature)
        self.finalize_button.setText("实验已完成")
        self.result_status_label.setText("最终测试完成；指标、审计信息和可复用模型已导出。")
        self.state_label.setText("最终测试完成")

    def _on_process_finished(self, exit_code: int, _exit_status: QProcess.ExitStatus) -> None:
        cancelled = self._cancel_requested
        self._read_worker_stdout()
        self._read_worker_stderr()
        self._flush_worker_stderr()
        action = self._worker_action
        if self._worker_output_buffer.strip():
            if cancelled:
                self._append_log("后台动作已取消；忽略进程结束时未完成的 worker 消息。")
            else:
                try:
                    payload = json.loads(self._worker_output_buffer.decode("utf-8"))
                    self._handle_worker_payload(payload)
                except (UnicodeDecodeError, json.JSONDecodeError):
                    self._append_log(self._worker_output_buffer.decode("utf-8", errors="replace"))
            self._worker_output_buffer = b""
        if exit_code != 0 and not cancelled and self.last_error is None:
            detail = self.stderr_text.strip().splitlines()[-1] if self.stderr_text.strip() else self.process.errorString()
            self.last_error = detail
            self.result_status_label.setText(f"后台进程退出代码 {exit_code}：{detail}")
            self.state_label.setText("任务失败")
        if not cancelled and (self.last_error or self.last_error_id or self.last_error_traceback):
            self._record_external_worker_error(
                error_type="WorkerError",
                message=self.last_error or "后台进程报告错误",
                traceback_text=self.stderr_text,
                action=action,
            )
        if cancelled:
            self.last_error = None
            self.last_error_id = None
            self.last_error_traceback = ""
            self.copy_error_button.setEnabled(False)
            self.state_label.setText("已取消")
            self.result_status_label.setText("后台动作已取消；未生成业务错误记录。")
            self._discard_session(remove_config=True)
        elif action == "train":
            if exit_code == 0 and self._session_id is not None:
                self._append_log("训练动作完成；持久 session 等待冻结。")
            else:
                if self.last_error is None:
                    self.result_status_label.setText("训练动作结束，但没有创建有效验证 session。")
                self._discard_session(remove_config=True)
        elif action == "freeze":
            self._finalize_requested = False
            if exit_code == 0 and self._frozen:
                self._append_log("冻结动作完成；可以明确启动最终测试。")
            elif exit_code == 0:
                self.result_status_label.setText("冻结动作未确认 session 状态；最终测试未启动。")
                self.finalize_button.setText("重试冻结")
        elif action == "test":
            if exit_code == 0 and self._test_result_payload is not None:
                self._worker_action = None
                self._finalize_requested = True
                self._start_session_action("export")
                return
            self._finalize_requested = False
            if exit_code == 0:
                self.result_status_label.setText("测试动作结束，但没有收到可用的测试结果。")
            self.finalize_button.setText("重试最终测试（同一 session）")
        elif action == "export":
            self._finalize_requested = False
            if exit_code == 0 and self.bound_result is not None:
                self._discard_session(remove_config=True, preserve_result=True)
            elif exit_code == 0:
                self.result_status_label.setText("导出动作结束，但没有绑定完整实验结果。")
                self.finalize_button.setText("重试导出")
        elif action in {"inspect", "inference"} and exit_code == 0:
            if action == "inspect":
                self.state_label.setText("模型已加载")
        if (action in {"inspect", "inference"} or action is None) and self._session_id is None:
            self._cleanup_config_file()
        self._worker_action = None
        self._cancel_requested = False
        controller = self.logging_controller
        if controller is not None:
            try:
                context = {"stage": action or "worker", "model": self.model_combo.currentData()}
                if cancelled:
                    controller.record_cancelled(context=context)
                else:
                    controller.record_process_exit(
                        exit_code,
                        context=context,
                        error_id=self.last_error_id,
                    )
            except Exception:
                pass
        self._set_worker_busy(False)

    def _record_external_worker_error(
        self,
        *,
        error_type: str,
        message: str,
        traceback_text: str,
        action: str | None,
    ) -> None:
        if self.last_error or traceback_text or self.last_error_id:
            self.copy_error_button.setEnabled(True)
        controller = self.logging_controller
        if controller is None:
            return
        try:
            record = controller.record_external_error(
                error_type,
                message,
                traceback_text or None,
                context={"stage": action or "worker", "model": self.model_combo.currentData()},
                error_id=self.last_error_id,
            )
        except Exception:
            return
        if record is not None:
            self.last_error_id = record.error_id
            if not self.last_error_traceback:
                self.last_error_traceback = record.traceback or traceback_text
            self.copy_error_button.setEnabled(True)

    def _on_process_error(self, _error: QProcess.ProcessError) -> None:
        if self.process.state() == QProcess.ProcessState.NotRunning:
            if self._cancel_requested:
                self._append_log("取消的后台进程已停止。")
                return
            message = self.process.errorString()
            self.last_error = message
            self.result_status_label.setText(f"无法启动后台进程：{message}")
            self.state_label.setText("进程启动失败")
            self._record_external_worker_error(
                error_type="WorkerProcessError",
                message=message,
                traceback_text="",
                action=self._worker_action,
            )
            if self._worker_action == "train":
                self._discard_session(remove_config=True)
            self._worker_action = None
            self._set_worker_busy(False)

    def _cleanup_config_file(self) -> None:
        if self._config_file_path is not None:
            try:
                self._config_file_path.unlink(missing_ok=True)
            except OSError as exc:
                self._append_log(f"清理临时配置文件失败：{exc}")
            self._config_file_path = None

    def _discard_session(self, *, remove_config: bool, preserve_result: bool = False) -> None:
        if self._session_file_path is not None:
            try:
                self._session_file_path.unlink(missing_ok=True)
            except OSError as exc:
                self._append_log(f"清理临时 session 失败：{exc}")
            self._session_file_path = None
        self._session_id = None
        self._prepared = False
        self._frozen = False
        self._prepared_signature = None
        self._session_capabilities = {}
        self._test_result_payload = None
        self._finalize_requested = False
        if remove_config:
            self._cleanup_config_file()
        if not preserve_result:
            self.finalize_button.setText("冻结模型设置")

    def _append_log(self, message: str) -> None:
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.events_text.appendPlainText(f"{timestamp}  {message}")

    def _render_metrics(self) -> None:
        self.metrics_text.setPlainText(json.dumps(self.current_metrics, ensure_ascii=False, indent=2, allow_nan=False))
        names: list[str] = []
        for split in ("train", "validation", "test"):
            values = self.current_metrics.get(split, {})
            if not isinstance(values, dict):
                continue
            for name, value in values.items():
                if name in {"sample_count", "cluster_count", "noise_count"}:
                    continue
                if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
                    if name not in names:
                        names.append(name)
        selected = self.metric_combo.currentData()
        self.metric_combo.blockSignals(True)
        self.metric_combo.clear()
        for name in names:
            self.metric_combo.addItem(name, name)
        if selected in names:
            self.metric_combo.setCurrentIndex(names.index(selected))
        self.metric_combo.blockSignals(False)
        self._render_metric_chart()

    def _render_sequence_summary(self, summary: Any) -> None:
        if summary is None:
            self.sequence_summary_text.clear()
            return
        if not isinstance(summary, dict) or summary.get("schema_version") != 1:
            self.sequence_summary_text.setPlainText("无法读取序列划分统计。")
            return
        splits = summary.get("splits")
        if not isinstance(splits, dict):
            self.sequence_summary_text.setPlainText("无法读取序列划分统计。")
            return
        fields = ("source_rows", "groups", "windows")
        counts: dict[str, dict[str, int]] = {}
        try:
            for split_name in ("train", "validation", "test"):
                part = splits.get(split_name)
                if not isinstance(part, dict):
                    raise ValueError
                counts[split_name] = {}
                for field in fields:
                    value = part.get(field)
                    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                        raise ValueError
                    counts[split_name][field] = value
        except ValueError:
            self.sequence_summary_text.setPlainText("序列划分统计格式无效。")
            return

        totals = {field: sum(counts[name][field] for name in counts) for field in fields}

        def ratio(split_name: str, field: str) -> str:
            total = totals[field]
            return "—" if not total else f"{counts[split_name][field] / total:.1%}"

        lines = [
            "实际序列划分（比例按三份的总量计算）：",
            f"总计：源行 {totals['source_rows']} · 组 {totals['groups']} · 窗口 {totals['windows']}",
        ]
        for split_name in ("train", "validation", "test"):
            part = counts[split_name]
            lines.append(
                f"{split_name}: 源行 {part['source_rows']} ({ratio(split_name, 'source_rows')}) · "
                f"组 {part['groups']} ({ratio(split_name, 'groups')}) · "
                f"窗口 {part['windows']} ({ratio(split_name, 'windows')})"
            )
        self.sequence_summary_text.setPlainText("\n".join(lines))

    def _render_metric_chart(self, *_args) -> None:
        self.metric_axes.clear()
        metric = self.metric_combo.currentData()
        splits: list[str] = []
        values: list[float] = []
        for split, color in (("train", "#2563EB"), ("validation", "#F59E0B"), ("test", "#0F766E")):
            entry = self.current_metrics.get(split, {})
            value = entry.get(metric) if isinstance(entry, dict) and metric else None
            if isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)):
                splits.append(split)
                values.append(float(value))
        if not values:
            self.metric_axes.text(0.5, 0.5, "完成验证后会显示指标图表", ha="center", va="center", color="#475569")
            self.metric_axes.set_axis_off()
        else:
            colors = ["#2563EB" if item == "train" else "#F59E0B" if item == "validation" else "#0F766E" for item in splits]
            bars = self.metric_axes.bar(splits, values, color=colors, width=0.58)
            self.metric_axes.set_title(str(metric), loc="left", color="#0F172A", fontsize=11, pad=10)
            self.metric_axes.set_ylabel("指标值", color="#334155")
            self.metric_axes.grid(axis="y", color="#CBD5E1", linewidth=0.7, alpha=0.75)
            self.metric_axes.set_axisbelow(True)
            self.metric_axes.spines["top"].set_visible(False)
            self.metric_axes.spines["right"].set_visible(False)
            for bar, value in zip(bars, values):
                self.metric_axes.annotate(f"{value:.4g}", (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                                          xytext=(0, 4), textcoords="offset points", ha="center", fontsize=8, color="#0F172A")
        self.metric_canvas.draw_idle()

    def _render_training_curves(self) -> None:
        self.curve_axes.clear()
        curves = self.training_curves
        if not isinstance(curves, dict):
            self.curve_status_label.setText("当前结果没有 skorch 逐 epoch 损失记录。")
            self.curve_axes.text(0.5, 0.5, "没有可绘制的训练曲线", ha="center", va="center", color="#475569")
            self.curve_axes.set_axis_off()
            self.curve_canvas.draw_idle()
            return
        epochs = curves.get("epochs")
        train_loss = curves.get("train_loss")
        validation_loss = curves.get("validation_loss")
        validation_available = curves.get("validation_available") is True
        valid = (
            isinstance(epochs, list) and isinstance(train_loss, list)
            and len(epochs) > 0 and len(epochs) == len(train_loss)
            and all(
                isinstance(value, (int, float)) and not isinstance(value, bool)
                and math.isfinite(float(value))
                for value in [*epochs, *train_loss]
            )
        )
        if validation_available:
            valid = valid and isinstance(validation_loss, list) and len(validation_loss) == len(epochs) and all(
                isinstance(value, (int, float)) and not isinstance(value, bool)
                and math.isfinite(float(value))
                for value in validation_loss
            )
        elif validation_loss is not None:
            valid = False
        if not valid:
            self.curve_status_label.setText("训练曲线数据不完整或不是有限数值。")
            self.curve_axes.text(0.5, 0.5, "训练曲线数据无效", ha="center", va="center", color="#B91C1C")
            self.curve_axes.set_axis_off()
            self.curve_canvas.draw_idle()
            return
        epochs = [int(value) for value in epochs]
        self.curve_axes.plot(epochs, train_loss, color="#2563EB", marker="o", markersize=3, label="训练损失")
        if validation_available:
            self.curve_axes.plot(epochs, validation_loss, color="#F59E0B", marker="o", markersize=3, label="验证损失")
        if curves.get("fit_scope") == "train_validation":
            self.curve_status_label.setText("最终重拟合曲线：使用训练集与验证集拟合；未计算验证损失。")
            self.curve_axes.set_title("训练 + 验证重拟合损失", loc="left", color="#0F172A", fontsize=11)
        else:
            self.curve_status_label.setText("曲线来自实际 skorch 逐 epoch history。")
            self.curve_axes.set_title("训练 / 验证损失", loc="left", color="#0F172A", fontsize=11)
        self.curve_axes.set_xlabel("Epoch")
        self.curve_axes.set_ylabel("Loss")
        self.curve_axes.grid(color="#CBD5E1", linewidth=0.7, alpha=0.75)
        self.curve_axes.legend(frameon=False)
        self.curve_axes.spines["top"].set_visible(False)
        self.curve_axes.spines["right"].set_visible(False)
        self.curve_canvas.draw_idle()

    def _render_audit(self, audit: dict[str, Any]) -> None:
        self.audit_text.setPlainText(json.dumps(audit, ensure_ascii=False, indent=2, allow_nan=False))

    def _populate_artifacts(self, paths: dict[str, str]) -> None:
        self.artifacts_list.clear()
        for name, path in paths.items():
            self._add_artifact(name, path)

    def _add_artifact(self, name: str, path: str) -> None:
        for index in range(self.artifacts_list.count()):
            if self.artifacts_list.item(index).data(Qt.ItemDataRole.UserRole) == path:
                return
        item = QListWidgetItem(f"{name}  ·  {path}")
        item.setData(Qt.ItemDataRole.UserRole, path)
        item.setToolTip(path)
        self.artifacts_list.addItem(item)

    def export_artifact(self, source: str | Path, destination: str | Path) -> Path:
        source_path = Path(source).expanduser().resolve()
        destination_path = Path(destination).expanduser().resolve()
        if not source_path.is_file():
            raise FileNotFoundError(f"导出源文件不存在：{source_path}")
        if destination_path.exists():
            raise FileExistsError(f"为防止覆盖现有文件，拒绝导出：{destination_path}")
        if source_path == destination_path:
            raise ValueError("导出目标不能与源文件相同")
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, destination_path)
        return destination_path

    def _selected_artifact_path(self) -> str | None:
        item = self.artifacts_list.currentItem()
        return str(item.data(Qt.ItemDataRole.UserRole)) if item is not None else None

    def _open_selected_artifact(self, *_args) -> None:
        path = self._selected_artifact_path()
        if path and Path(path).exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(Path(path).resolve())))
        elif path:
            self.result_status_label.setText(f"文件不存在：{path}")

    def _export_selected_artifact(self) -> None:
        source = self._selected_artifact_path()
        if not source or not Path(source).is_file():
            self.result_status_label.setText("请选择一个存在的导出文件。")
            return
        suggested = str(Path(source).with_name(f"{Path(source).stem}_copy{Path(source).suffix}"))
        destination, _ = QFileDialog.getSaveFileName(self, "导出文件副本", suggested, "所有文件 (*.*)")
        if not destination:
            return
        try:
            exported = self.export_artifact(source, destination)
        except Exception as exc:
            self.result_status_label.setText(f"导出失败：{exc}")
            return
        self.result_status_label.setText(f"文件副本已导出：{exported}")

    def reload_model(self, model_path: str | Path | None = None) -> bool:
        path = Path(model_path or self.model_path_edit.text().strip()).expanduser()
        if not path.is_file():
            self.result_status_label.setText(f"请选择存在的 model.joblib 文件：{path}")
            return False
        resolved_path = path.resolve()
        self.model_path_edit.setText(str(resolved_path))
        if not self._start_worker(["inspect", "--model", str(resolved_path)], action="inspect"):
            return False
        self.result_status_label.setText("正在后台加载模型并读取其真实能力。")
        self.state_label.setText("正在加载模型")
        self._set_worker_busy(True)
        return True

    def _apply_model_info(self, model_id: str, path: str, capabilities: dict[str, bool],
                          feature_columns: list[Any], frozen_hash: str | None = None) -> None:
        self.loaded_model_path = str(Path(path).expanduser().resolve())
        self.loaded_model_id = model_id
        self.loaded_model_capabilities = {str(key): bool(value) for key, value in capabilities.items()}
        self.model_path_edit.blockSignals(True)
        self.model_path_edit.setText(self.loaded_model_path)
        self.model_path_edit.blockSignals(False)
        supported = [label for op, label in OPERATION_LABELS.items() if self.loaded_model_capabilities.get(op)]
        columns_text = f"；需要特征列：{', '.join(map(str, feature_columns))}" if feature_columns else ""
        hash_text = f"；配置 SHA-256：{frozen_hash}" if frozen_hash else ""
        self.model_info_label.setText(f"已加载 {model_id}；可用操作：{', '.join(supported) or '无'}{columns_text}{hash_text}")
        self.operation_combo.blockSignals(True)
        self.operation_combo.clear()
        for operation, label in OPERATION_LABELS.items():
            if self.loaded_model_capabilities.get(operation):
                self.operation_combo.addItem(label, operation)
        self.operation_combo.setEnabled(self.operation_combo.count() > 0)
        self.operation_combo.blockSignals(False)
        self.inference_button.setEnabled(self.operation_combo.count() > 0)

    def _browse_source_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "选择训练数据", "", "表格文件 (*.csv *.xlsx *.xls)")
        if path:
            self.source_path_edit.setText(path)
            self.load_data(path)

    def _browse_output_directory(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, "选择结果目录", self.output_dir_edit.text().strip() or str(Path.cwd()))
        if directory:
            self.output_dir_edit.setText(directory)

    def _browse_model_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "加载模型", "", "Joblib 模型 (*.joblib);;所有文件 (*.*)")
        if path:
            self.model_path_edit.setText(path)

    def _browse_inference_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "选择新数据", "", "表格文件 (*.csv *.xlsx *.xls)")
        if path:
            self.inference_data_edit.setText(path)

    def start_inference(self, output_path: str | Path | None = None) -> bool:
        model_path = self.loaded_model_path
        if not model_path or not Path(model_path).is_file():
            self.result_status_label.setText("请先加载一个模型。")
            return False
        operation = self.operation_combo.currentData()
        if not operation or not self.loaded_model_capabilities.get(str(operation), False):
            self.result_status_label.setText("当前模型不支持所选操作。")
            return False
        data_path = Path(self.inference_data_edit.text().strip()).expanduser()
        if not data_path.is_file():
            self.result_status_label.setText("请选择存在的新数据文件。")
            return False
        destination = Path(output_path) if output_path is not None else data_path.with_name(
            f"{data_path.stem}_{operation}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        )
        if destination.exists():
            self.result_status_label.setText(f"为避免覆盖，请选择新的推理结果路径：{destination}")
            return False
        args = ["inference", "--operation", str(operation), "--model", model_path,
                "--data", str(data_path.resolve()), "--output", str(destination.resolve())]
        sheet = self.inference_sheet_edit.text().strip()
        if sheet:
            args.extend(["--sheet", sheet])
        if not self._start_worker(args, action="inference"):
            return False
        self._set_worker_busy(True)
        self.state_label.setText("正在运行推理")
        self.result_status_label.setText("推理正在后台运行；结果会写入新的 CSV 文件。")
        return True

    def _on_source_path_edited(self, text: str) -> None:
        if self.dataset is not None and text.strip() != self.dataset.source_path:
            self._invalidate_result("训练数据路径已编辑，需要重新加载")
            self.data_info_label.setText("文件路径已更改，请重新加载预览后再训练。")
            self.data_info_label.setStyleSheet("color: #92400E;")

    def _on_training_sheet_changed(self, _index: int) -> None:
        if self.dataset is None or not self.dataset.sheet_names:
            return
        selected = self.sheet_combo.currentText()
        if selected and selected != self.dataset.selected_sheet:
            self.load_data(self.dataset.source_path, selected)

    def _on_target_changed(self, _index: int) -> None:
        self._update_target_state()
        self._on_configuration_changed()

    def _on_task_changed(self, _index: int = -1) -> None:
        if not self._applying_configuration:
            self._clear_loaded_parameter_snapshot()
        self._refresh_model_choices()
        self._update_target_state()
        self._on_configuration_changed()

    def _on_model_changed(self, _index: int = -1) -> None:
        if not self._applying_configuration:
            self._clear_loaded_parameter_snapshot()
        self._refresh_parameters()
        self._update_sequence_config_state()
        self._update_target_state()
        self._on_configuration_changed()

    def _update_sequence_config_state(self) -> None:
        is_sequence = self.model_combo.currentData() in {"H01", "H02", "H03", "N04", "N06"}
        self.sequence_config_label.setVisible(is_sequence)
        self.sequence_config_edit.setVisible(is_sequence)
        self.sequence_config_help.setVisible(is_sequence)

    def _on_sequence_config_changed(self) -> None:
        self._update_target_state()
        self._on_configuration_changed()

    def _on_loaded_model_path_edited(self, text: str) -> None:
        if self.loaded_model_path is not None:
            try:
                changed = Path(text).expanduser().resolve() != Path(self.loaded_model_path).resolve()
            except OSError:
                changed = True
            if changed:
                self.loaded_model_path = None
                self.loaded_model_id = None
                self.loaded_model_capabilities.clear()
                self.model_info_label.setText("模型路径已更改；重新加载后才会启用推理操作。")
                self.operation_combo.clear()
                self.operation_combo.setEnabled(False)
                self.inference_button.setEnabled(False)

    def _on_feature_selection_changed(self, *_args) -> None:
        if not self._params_initializing:
            self._feature_selection_unspecified = False
            self._on_configuration_changed()

    def _on_parameter_changed(self, name: str) -> None:
        if self._params_initializing:
            return
        if self._loaded_parameters is not None:
            self._edited_parameter_names.add(name)
        self._on_configuration_changed()

    def _clear_loaded_parameter_snapshot(self) -> None:
        self._loaded_parameters = None
        self._loaded_advanced_text = None
        self._loaded_advanced_names.clear()
        self._edited_parameter_names.clear()

    def _on_configuration_changed(self, *_args) -> None:
        if self._params_initializing:
            return
        self._invalidate_result("配置已更改")

    def _invalidate_result(self, reason: str) -> None:
        self._session_plot_sources = {}
        has_session = self._session_id is not None
        if self.bound_result is None and not self.current_metrics and not self.artifact_paths and not has_session:
            return
        if has_session:
            self._discard_session(remove_config=True)
        self._clear_bound_result()
        self.result_status_label.setText(f"{reason}；先前结果已解除绑定，请重新训练与验证。")
        self.state_label.setText("结果已失效")
        self._update_worker_controls()

    def _clear_bound_result(self) -> None:
        self._session_plot_sources = {}
        self.bound_result = None
        self.current_metrics = {}
        self.training_curves = None
        self.artifact_paths = {}
        self.metrics_text.clear()
        self.sequence_summary_text.clear()
        self.audit_text.clear()
        self.artifacts_list.clear()
        self._render_metrics()
        self._render_training_curves()

    def _set_worker_busy(self, busy: bool) -> None:
        self.data_group.setEnabled(not busy)
        self.model_group.setEnabled(not busy)
        self.output_group.setEnabled(not busy)
        self.run_button.setEnabled(not busy)
        self.cancel_button.setEnabled(busy)
        self.finalize_button.setEnabled(busy and self._prepared and not self._finalize_requested)
        self.progress_bar.setVisible(busy and not self._prepared)
        self.inference_group.setEnabled(not busy)
        self._update_worker_controls()

    def _update_worker_controls(self) -> None:
        has_data = self.dataset is not None
        self.data_overview_action.setEnabled(has_data)
        self.basic_plot_action.setEnabled(has_data)
        self.result_plot_action.setEnabled(bool(self.bound_result is not None and self._session_plot_sources))
        process_running = self.process.state() != QProcess.ProcessState.NotRunning
        self.run_button.setEnabled(has_data and not process_running and self._worker_action is None and self._session_id is None)
        self.cancel_button.setEnabled(process_running or self._session_id is not None)
        self.finalize_button.setEnabled(
            not process_running and self._worker_action is None and self._prepared and not self._finalize_requested
        )

    def closeEvent(self, event) -> None:
        if self._guide_dialog is not None:
            self._guide_dialog.close()
        if self.process.state() != QProcess.ProcessState.NotRunning:
            self.process.kill()
            self.process.waitForFinished(1500)
        self._discard_session(remove_config=True, preserve_result=True)
        event.accept()


def main() -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    _configure_application_font(app)
    preferences_path = _default_preferences_path()
    try:
        startup_preferences = PreferenceStore(preferences_path).load()
    except PreferenceError:
        startup_preferences = Preferences()
    controller = _ApplicationLogController(startup_preferences.log_directory)
    try:
        with _application_exception_hooks(controller):
            window = WorkbenchWindow(
                preferences_path=preferences_path,
                logging_controller=controller,
            )
            if not controller.available:
                window.result_status_label.setText(
                    f"日志不可用；业务流程仍可继续：{controller.last_error or '初始化失败'}"
                )
            window.show()
            return app.exec()
    finally:
        controller.close()


def _configure_application_font(app: QApplication) -> str:
    """Load the installed CJK face into this Qt process and use it for the UI."""
    font_family = "Microsoft YaHei UI"
    try:
        font_path = find_matplotlib_font("Microsoft YaHei", fallback_to_default=False)
        font_id = QFontDatabase.addApplicationFont(font_path)
        if font_id >= 0:
            families = QFontDatabase.applicationFontFamilies(font_id)
            font_family = next(
                (name for name in families if name.casefold() == "microsoft yahei ui"),
                next((name for name in families if name.casefold() == "microsoft yahei"), font_family),
            )
    except (OSError, ValueError):
        pass
    app.setFont(QFont(font_family, 9))
    return font_family


__all__ = ["DataFrameTableModel", "WorkbenchWindow", "main"]
