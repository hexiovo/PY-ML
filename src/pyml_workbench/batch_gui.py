"""Responsive queue dialog for multi-dataset, multi-model search jobs."""
from __future__ import annotations

import codecs
import csv
import json
import math
import os
import re
from pathlib import Path
import sys
import tempfile
import uuid
from typing import Any

from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

from PySide6.QtCore import QProcess, QProcessEnvironment, Qt
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDialog,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .batch import (
    _finite_objective_score,
    _rank_validation_group_entries,
    request_job_control,
)
from .catalog import list_models
from .config import DatasetConfig, ExperimentConfig, SplitConfig
from .data import load_dataset
from .history import HistoryStore
from .objectives import ObjectiveSpec
from .search import SearchSpec
from .runtime import worker_command
from .search_space import SearchSpace, recommended_space
from .sequence import SequenceConfig
from .sequence_reporting import sequence_plan_summary_from_manifest
from .plot_cache import load_batch_plot_sources
from .plot_dialog import PlotDialog
from .plotting import build_plot_payload
from .gui import PlotSourceChooserDialog


TASKS = (
    ("分类", "classification"),
    ("回归", "regression"),
    ("聚类", "clustering"),
    ("降维", "dimensionality reduction"),
    ("异常检测", "anomaly detection"),
    ("序列建模 / HMM", "sequence_modeling"),
)

_WORKER_ERROR_ID_RE = re.compile(r"^[0-9a-f]{32}$")


def _nonnegative_worker_count(value: Any) -> bool:
    return type(value) is int and value >= 0


def _nonnegative_worker_duration(value: Any) -> bool:
    if type(value) not in {int, float} or value < 0:
        return False
    try:
        return math.isfinite(float(value))
    except (OverflowError, ValueError):
        return False


class BatchSearchDialog(QDialog):
    """Batch configuration, durable history, comparison and winner controls."""

    def __init__(self, parent=None, initial_config: ExperimentConfig | None = None):
        super().__init__(parent)
        self.setWindowTitle("PY-ML 批量搜索与队列")
        self.resize(1220, 860)
        self.setMinimumSize(1000, 720)
        self._worker_action: str | None = None
        self._temporary_request: Path | None = None
        self._stdout_buffer = b""
        self._stderr_text = ""
        self._stderr_decoder = None
        self._error_details: list[dict[str, str]] = []
        self.logging_controller = getattr(parent, "logging_controller", None)
        self._job_ids: list[str] = []
        self._last_error: str | None = None
        self._initial_config = initial_config
        self._plot_dialogs: list[PlotDialog] = []
        self.process = QProcess(self)
        self.process.setProcessChannelMode(QProcess.ProcessChannelMode.SeparateChannels)
        self.process.readyReadStandardOutput.connect(self._read_stdout)
        self.process.readyReadStandardError.connect(self._read_stderr)
        self.process.finished.connect(self._process_finished)
        self.process.errorOccurred.connect(self._process_error)
        self._build_ui()
        self._connect_ui()
        self._refresh_controls()
        self._update_preview()

    def _build_ui(self) -> None:
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(16, 14, 16, 14)
        root_layout.setSpacing(10)

        path_row = QHBoxLayout()
        path_row.addWidget(QLabel("批次目录"))
        self.root_edit = QLineEdit(str(Path(tempfile.gettempdir()) / "pyml_workbench_batch"))
        self.root_edit.setAccessibleName("批次与历史目录")
        path_row.addWidget(self.root_edit, 1)
        self.root_browse_button = QPushButton("选择目录")
        self.load_history_button = QPushButton("读取历史")
        path_row.addWidget(self.root_browse_button)
        path_row.addWidget(self.load_history_button)
        root_layout.addLayout(path_row)

        config_grid = QGridLayout()
        config_grid.setColumnStretch(0, 1)
        config_grid.setColumnStretch(1, 1)
        config_grid.addWidget(QLabel("独立数据配置 JSON（每项指定 task、target、features、models、seed）"), 0, 0)
        config_grid.addWidget(QLabel("每个模型的 typed search space JSON"), 0, 1)
        self.datasets_edit = QPlainTextEdit()
        self.datasets_edit.setAccessibleName("批量数据集与模型组合配置")
        self.datasets_edit.setPlaceholderText(
            '[{"source_path":"data.csv","task":"classification","target_column":"target",'
            '"feature_columns":["x1","x2"],"models":["C01","C02"],"seed":42}]'
        )
        self.datasets_edit.setMinimumHeight(190)
        self.spaces_edit = QPlainTextEdit()
        self.spaces_edit.setAccessibleName("按模型配置的类型化参数搜索空间")
        self.spaces_edit.setPlaceholderText(
            '{"C01":{"fields":{"C":{"type":"real","low":0.1,"high":10,"log":true,'
            '"values":[0.1,1,10]}}}}'
        )
        self.spaces_edit.setMinimumHeight(190)
        config_grid.addWidget(self.datasets_edit, 1, 0)
        config_grid.addWidget(self.spaces_edit, 1, 1)
        root_layout.addLayout(config_grid)

        form = QFormLayout()
        self.method_combo = QComboBox()
        for label, value in (
            ("网格搜索 Grid", "grid"),
            ("随机搜索 Random", "random"),
            ("TPE（Optuna）", "tpe"),
            ("遗传搜索（pymoo）", "genetic"),
            ("退火搜索（SciPy）", "annealing"),
        ):
            self.method_combo.addItem(label, value)
        self.max_fits_spin = QSpinBox()
        self.max_fits_spin.setRange(1, 50)
        self.max_fits_spin.setValue(50)
        self.max_proposals_spin = QSpinBox()
        self.max_proposals_spin.setRange(1, 250)
        self.max_proposals_spin.setValue(250)
        self.minutes_spin = QSpinBox()
        self.minutes_spin.setRange(1, 20)
        self.minutes_spin.setValue(20)
        self.parallel_spin = QSpinBox()
        self.parallel_spin.setRange(1, 2)
        self.parallel_spin.setValue(1)
        form.addRow("搜索方法", self.method_combo)
        form.addRow("每个组合最多真实拟合次数", self.max_fits_spin)
        form.addRow("每个组合最多 proposals", self.max_proposals_spin)
        form.addRow("每个组合活动时限（分钟）", self.minutes_spin)
        form.addRow("并行 worker（每个 worker 线程数固定为 1）", self.parallel_spin)
        root_layout.addLayout(form)

        self.preview_label = QLabel("请填写数据配置以查看组合数和预算上限。")
        self.preview_label.setWordWrap(True)
        self.preview_label.setObjectName("helperText")
        root_layout.addWidget(self.preview_label)

        tools_row = QHBoxLayout()
        self.add_current_button = QPushButton("添加当前单任务配置")
        self.add_files_button = QPushButton("添加多个数据文件")
        self.start_button = QPushButton("预检并启动队列")
        self.start_button.setObjectName("primaryButton")
        self.pause_button = QPushButton("暂停所选任务")
        self.resume_button = QPushButton("恢复所选任务")
        self.cancel_button = QPushButton("取消所选任务")
        self.freeze_button = QPushButton("冻结所选验证 winner")
        self.finalize_button = QPushButton("重拟合并执行一次最终测试")
        self.export_exploration_button = QPushButton("导出训练探索 winner")
        self.export_exploration_button.setToolTip(
            "仅导出已完成 train_exploratory 搜索的训练模型和结果；不重拟合、不进入验证榜或测试集。"
        )
        self.refresh_button = QPushButton("刷新队列")
        self.export_button = QPushButton("导出对比 CSV")
        self.plot_result_button = QPushButton("绘制结果图…")
        self.plot_result_button.setAccessibleName("绘制所选批量任务的结果图")
        for button in (
            self.add_current_button, self.add_files_button, self.start_button,
            self.pause_button, self.resume_button, self.cancel_button,
        ):
            tools_row.addWidget(button)
        root_layout.addLayout(tools_row)
        selection_row = QHBoxLayout()
        for button in (
            self.freeze_button, self.finalize_button, self.export_exploration_button,
            self.refresh_button, self.export_button, self.plot_result_button,
        ):
            selection_row.addWidget(button)
        root_layout.addLayout(selection_row)

        self.jobs_table = QTableWidget(0, 11)
        self.jobs_table.setAccessibleName("批量搜索历史与对比")
        self.jobs_table.setHorizontalHeaderLabels(
            ["任务 ID", "分组", "任务", "模型", "状态", "Fits", "Proposals", "活动秒", "Validation 名次", "目标分数（分区见分组）", "最佳参数"]
        )
        self.jobs_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.jobs_table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.jobs_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.jobs_table.horizontalHeader().setStretchLastSection(True)
        root_layout.addWidget(self.jobs_table, 1)

        details_row = QHBoxLayout()
        self.job_details_text = QPlainTextEdit()
        self.job_details_text.setReadOnly(True)
        self.job_details_text.setAccessibleName("所选批量任务与序列划分详情")
        self.job_details_text.setMinimumHeight(105)
        self.job_details_text.setMaximumHeight(150)
        details_row.addWidget(self.job_details_text, 1)
        curve_panel = QWidget()
        curve_layout = QVBoxLayout(curve_panel)
        curve_layout.setContentsMargins(0, 0, 0, 0)
        self.job_curve_status = QLabel("选择已完成的深度模型任务以查看 winner 损失曲线。")
        self.job_curve_status.setObjectName("helperText")
        curve_layout.addWidget(self.job_curve_status)
        self.job_curve_figure = Figure(figsize=(5.4, 1.65), tight_layout=True)
        self.job_curve_axes = self.job_curve_figure.add_subplot(111)
        self.job_curve_canvas = FigureCanvasQTAgg(self.job_curve_figure)
        details_row.addWidget(curve_panel, 1)
        curve_layout.addWidget(self.job_curve_canvas, 1)
        root_layout.addLayout(details_row)

        self.status_label = QLabel(
            "搜索只使用训练/验证数据；train_exploratory 只形成训练探索结果、不进入验证榜。"
            "冻结验证 winner 后才会读取一次测试集。"
        )
        self.status_label.setWordWrap(True)
        root_layout.addWidget(self.status_label)
        self.copy_error_button = QPushButton("复制错误详情")
        self.copy_error_button.setAccessibleName("复制批量错误 ID 和 traceback")
        self.copy_error_button.setEnabled(False)
        self.copy_error_button.clicked.connect(self.copy_error_details)
        root_layout.addWidget(self.copy_error_button)
        self.log_text = QPlainTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setAccessibleName("批量搜索事件与错误")
        self.log_text.setMaximumHeight(130)
        root_layout.addWidget(self.log_text)

        self.setStyleSheet(
            'QWidget { background: #EFF6FF; color: #0F172A; font-family: "Microsoft YaHei UI"; }'
            'QPlainTextEdit, QLineEdit, QComboBox, QSpinBox, QTableWidget { background: white; '
            'border: 1px solid #94A3B8; border-radius: 4px; padding: 4px; }'
            'QPushButton { background: #DBEAFE; border: 1px solid #93C5FD; border-radius: 5px; padding: 6px; }'
            'QPushButton#primaryButton { background: #2563EB; color: white; font-weight: 600; }'
            'QLabel#helperText { color: #475569; }'
        )

        if self._initial_config is not None:
            self._set_current_config(self._initial_config)

    def _connect_ui(self) -> None:
        self.root_browse_button.clicked.connect(self._browse_root)
        self.load_history_button.clicked.connect(self.refresh_history)
        self.add_current_button.clicked.connect(self._add_current_config)
        self.add_files_button.clicked.connect(self._add_files)
        self.start_button.clicked.connect(self.start_batch)
        self.pause_button.clicked.connect(lambda: self._control_selected("pause"))
        self.resume_button.clicked.connect(lambda: self._control_selected("resume"))
        self.cancel_button.clicked.connect(lambda: self._control_selected("cancel"))
        self.freeze_button.clicked.connect(self._freeze_selected)
        self.finalize_button.clicked.connect(self._finalize_selected)
        self.export_exploration_button.clicked.connect(self._export_exploration_selected)
        self.refresh_button.clicked.connect(self.refresh_history)
        self.export_button.clicked.connect(self.export_comparison)
        self.plot_result_button.clicked.connect(self._plot_selected)
        self.datasets_edit.textChanged.connect(self._update_preview)
        self.spaces_edit.textChanged.connect(self._update_preview)
        self.max_fits_spin.valueChanged.connect(self._update_preview)
        self.max_proposals_spin.valueChanged.connect(self._update_preview)
        self.minutes_spin.valueChanged.connect(self._update_preview)
        self.parallel_spin.valueChanged.connect(self._update_preview)
        self.jobs_table.itemSelectionChanged.connect(self._selected_job_changed)

    def _set_current_config(self, config: ExperimentConfig) -> None:
        dataset = config.dataset
        models = [config.model_id]
        row = {
            "source_path": dataset.source_path,
            "sheet_name": dataset.sheet_name,
            "task": config.task,
            "target_column": dataset.target_column,
            "feature_columns": list(dataset.feature_columns) if dataset.feature_columns is not None else None,
            "models": models,
            "parameters": config.parameters,
            "seed": config.split.seed,
            "sequence": config.sequence.to_dict() if config.sequence is not None else None,
        }
        self.datasets_edit.setPlainText(json.dumps([row], ensure_ascii=False, indent=2))
        try:
            self.spaces_edit.setPlainText(
                json.dumps({config.model_id: recommended_space(config.model_id).to_dict()}, ensure_ascii=False, indent=2)
            )
        except ValueError:
            self.spaces_edit.setPlainText("{}")

    def _add_current_config(self) -> None:
        config = self._initial_config
        if config is None and self.parent() is not None:
            builder = getattr(self.parent(), "build_experiment_config", None)
            if callable(builder):
                try:
                    config = builder()
                except Exception as exc:
                    self._show_error(f"当前配置无法读取：{exc}")
                    return
        if config is None:
            self._show_error("请先在主窗口加载数据并选择任务与模型，或直接编辑数据配置 JSON。")
            return
        if not self.datasets_edit.toPlainText().strip() or self.datasets_edit.toPlainText().strip() == "[]":
            self._set_current_config(config)
        else:
            self._set_current_config(config)
        self._update_preview()

    def _add_files(self) -> None:
        paths, _ = QFileDialog.getOpenFileNames(
            self, "添加数据文件", "", "表格文件 (*.csv *.xlsx *.xls)"
        )
        if not paths:
            return
        try:
            rows = self._parse_dataset_rows()
        except Exception:
            rows = []
        defaults = self._initial_config
        task = defaults.task if defaults else "classification"
        default_model = defaults.model_id if defaults else self._first_model(task)
        default_target = defaults.dataset.target_column if defaults else None
        default_features = defaults.dataset.feature_columns if defaults else None
        for raw_path in paths:
            frame = load_dataset(DatasetConfig(raw_path)).frame
            target = default_target if default_target in frame.columns else None
            if default_features and all(name in frame.columns for name in default_features):
                features = list(default_features)
            else:
                features = [name for name in frame.columns if name != target]
            row = {
                "source_path": str(Path(raw_path).expanduser().resolve()),
                "task": task,
                "target_column": target,
                "feature_columns": features,
                "models": [default_model],
                "seed": defaults.split.seed if defaults else 42,
            }
            if defaults is not None and defaults.sequence is not None:
                row["sequence"] = defaults.sequence.to_dict()
            rows.append(row)
        self.datasets_edit.setPlainText(json.dumps(rows, ensure_ascii=False, indent=2))

    @staticmethod
    def _first_model(task: str) -> str:
        models = list_models(task)
        if not models:
            raise ValueError(f"没有可运行的{task}模型")
        return models[0]["id"]

    def _parse_dataset_rows(self) -> list[dict[str, Any]]:
        try:
            rows = json.loads(self.datasets_edit.toPlainText())
        except json.JSONDecodeError as exc:
            raise ValueError(f"数据配置 JSON 第 {exc.lineno} 行第 {exc.colno} 列错误：{exc.msg}") from exc
        if not isinstance(rows, list) or not rows:
            raise ValueError("数据配置必须是至少包含一项的 JSON 数组")
        for index, row in enumerate(rows, 1):
            if not isinstance(row, dict):
                raise ValueError(f"第 {index} 项必须是 JSON 对象")
            if not row.get("source_path") or row.get("task") not in {task for _, task in TASKS}:
                raise ValueError(f"第 {index} 项必须包含有效 source_path 和 task")
            models = row.get("models")
            if not isinstance(models, list) or not models:
                raise ValueError(f"第 {index} 项的 models 必须是非空模型 ID 列表")
            if len(set(models)) != len(models):
                raise ValueError(f"第 {index} 项 models 不能包含重复 ID")
        return rows

    def _parse_spaces(self) -> dict[str, Any]:
        try:
            spaces = json.loads(self.spaces_edit.toPlainText() or "{}")
        except json.JSONDecodeError as exc:
            raise ValueError(f"搜索空间 JSON 第 {exc.lineno} 行第 {exc.colno} 列错误：{exc.msg}") from exc
        if not isinstance(spaces, dict):
            raise ValueError("搜索空间必须是以 model_id 为键的 JSON 对象")
        return spaces

    def _build_requests(self) -> list[tuple[ExperimentConfig, SearchSpec]]:
        rows = self._parse_dataset_rows()
        spaces = self._parse_spaces()
        requests: list[tuple[ExperimentConfig, SearchSpec]] = []
        method = self.method_combo.currentData()
        for row_number, row in enumerate(rows, 1):
            features = row.get("feature_columns")
            dataset = DatasetConfig(
                source_path=str(Path(row["source_path"]).expanduser().resolve()),
                sheet_name=row.get("sheet_name"),
                target_column=row.get("target_column"),
                feature_columns=tuple(features) if features is not None else None,
            )
            split = SplitConfig(seed=int(row.get("seed", 42)))
            objective = ObjectiveSpec.from_dict(row.get("objective"))
            for model_id in row["models"]:
                try:
                    raw_space = spaces.get(model_id)
                    space = SearchSpace.from_dict(raw_space) if raw_space is not None else recommended_space(model_id)
                except Exception as exc:
                    raise ValueError(
                        f"第 {row_number} 份数据的模型 {model_id} 没有有效搜索空间：{exc}。"
                        "请在右侧 JSON 中为该模型添加 fields/fixed。"
                    ) from exc
                config = ExperimentConfig(
                    dataset=dataset,
                    task=row["task"],
                    model_id=model_id,
                    parameters=dict(row.get("parameters", {})),
                    split=split,
                    output_dir=None,
                    sequence=SequenceConfig.from_dict(row["sequence"]) if row.get("sequence") is not None else None,
                )
                spec = SearchSpec(
                    method=method,
                    space=space,
                    objective=objective,
                    max_fits=self.max_fits_spin.value(),
                    timeout_seconds=float(self.minutes_spin.value() * 60),
                    max_proposals=self.max_proposals_spin.value(),
                    seed=int(row.get("search_seed", row.get("seed", 42))),
                )
                requests.append((config, spec))
        return requests

    def _update_preview(self, *_args) -> None:
        try:
            rows = self._parse_dataset_rows()
            combinations = sum(len(row["models"]) for row in rows)
            per_combo_fits = self.max_fits_spin.value()
            max_total_fits = combinations * per_combo_fits
            max_total_proposals = combinations * self.max_proposals_spin.value()
            parallel = self.parallel_spin.value()
            max_wall_minutes = combinations * self.minutes_spin.value() / parallel
            self.preview_label.setText(
                f"{len(rows)} 份独立数据配置 × 共 {combinations} 个兼容模型组合；"
                f"最多 {max_total_fits} 次真实搜索拟合、{max_total_proposals} 个 proposals，"
                f"活动时长上限约 {max_wall_minutes:.1f} worker 分钟（每组合 {self.minutes_spin.value()} 分钟，"
                f"最多 {parallel} 个并行 worker）。Grid/TPE/随机/遗传/退火均使用同一类型化空间。"
            )
        except Exception as exc:
            self.preview_label.setText(f"配置预览：{exc}")

    def start_batch(self) -> bool:
        if self.process.state() != QProcess.ProcessState.NotRunning:
            return False
        try:
            requests = self._build_requests()
            root = Path(self.root_edit.text().strip()).expanduser().resolve()
            root.mkdir(parents=True, exist_ok=True)
            if root.exists() and not root.is_dir():
                raise ValueError(f"批次路径不是目录：{root}")
            history = root / "history.sqlite3"
            payload = [
                {"config": config.to_dict(), "spec": spec.to_dict()}
                for config, spec in requests
            ]
            handle = tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", suffix=".json", prefix="pyml_batch_request_", delete=False
            )
            self._temporary_request = Path(handle.name)
            with handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2, allow_nan=False)
        except Exception as exc:
            self._show_error(f"批量预检配置失败：{exc}")
            return False
        self._job_ids = []
        self._last_error = None
        self.status_label.setText("后台正在核验数据、目标列、模型兼容性和搜索空间；通过后会创建队列并开始搜索。")
        self.log_text.clear()
        self._start_worker(
            [
                "batch-run-request", "--history", str(history), "--root", str(root),
                "--requests", str(self._temporary_request),
                "--parallel-workers", str(self.parallel_spin.value()),
            ],
            action="batch-run-request",
        )
        return True

    def _start_worker(self, arguments: list[str], *, action: str) -> None:
        self._worker_action = action
        self._stdout_buffer = b""
        self._stderr_text = ""
        self._stderr_decoder = codecs.getincrementaldecoder("utf-8")("replace")
        self._error_details = []
        self._last_error = None
        self.copy_error_button.setEnabled(False)
        environment = QProcessEnvironment.systemEnvironment()
        controller_root = getattr(self.logging_controller, "log_root", None)
        parent_preferences = getattr(self.parent(), "preferences", None)
        log_root = controller_root or getattr(parent_preferences, "log_directory", None)
        if log_root:
            environment.insert("PYML_LOG_ROOT", str(log_root))
        environment.insert("PYML_LOG_SESSION_ID", uuid.uuid4().hex)
        self.process.setProcessEnvironment(environment)
        self.process.setWorkingDirectory(str(Path.cwd()))
        executable, worker_arguments = worker_command(arguments)
        self.process.start(executable, worker_arguments)
        self._refresh_controls()

    def _read_stdout(self) -> None:
        self._stdout_buffer += bytes(self.process.readAllStandardOutput())
        while b"\n" in self._stdout_buffer:
            line, self._stdout_buffer = self._stdout_buffer.split(b"\n", 1)
            if not line.strip():
                continue
            try:
                payload = json.loads(line.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as exc:
                self._append_log(f"无法解析 worker 输出：{exc}")
                continue
            self._handle_worker_payload(payload)

    def _read_stderr(self) -> None:
        decoder = self._stderr_decoder
        if decoder is None:
            decoder = codecs.getincrementaldecoder("utf-8")("replace")
            self._stderr_decoder = decoder
        self._stderr_text += decoder.decode(
            bytes(self.process.readAllStandardError()), final=False
        )

    def _flush_stderr(self) -> None:
        self._read_stderr()
        decoder, self._stderr_decoder = self._stderr_decoder, None
        if decoder is not None:
            self._stderr_text += decoder.decode(b"", final=True)

    def _remember_error(
        self,
        *,
        error_id: str | None,
        message: str,
        traceback_text: str = "",
        error_type: str = "WorkerError",
    ) -> None:
        safe_error_id = (
            error_id
            if isinstance(error_id, str) and _WORKER_ERROR_ID_RE.fullmatch(error_id)
            else ""
        )
        item = {
            "error_id": safe_error_id,
            "message": message,
            "traceback": traceback_text,
            "error_type": error_type,
        }
        if not any(
            existing["error_id"] == item["error_id"]
            and existing["message"] == item["message"]
            for existing in self._error_details
        ):
            self._error_details.append(item)
        self.copy_error_button.setEnabled(True)

    def copy_error_details(self) -> bool:
        if not self._error_details and not self._stderr_text.strip() and not self._last_error:
            return False
        details = []
        for item in self._error_details:
            details.append(
                "\n".join(
                    part
                    for part in (
                        f"error_id: {item['error_id'] or 'unavailable'}",
                        item["message"],
                        item["traceback"].rstrip(),
                    )
                    if part
                )
            )
        if not details and self._last_error:
            details.append(f"error_id: unavailable\n{self._last_error}")
        if self._stderr_text.strip() and not any(item["traceback"] for item in self._error_details):
            details.append(self._stderr_text.strip())
        QApplication.clipboard().setText("\n\n".join(details))
        self.status_label.setText("错误 ID 与 traceback 已复制到剪贴板。")
        return True

    def _record_external_errors(self) -> None:
        controller = self.logging_controller
        if controller is None:
            return
        for item in self._error_details:
            try:
                record = controller.record_external_error(
                    item["error_type"],
                    item["message"],
                    item["traceback"] or self._stderr_text or None,
                    context={"stage": self._worker_action or "batch_worker"},
                    error_id=item["error_id"] or None,
                )
            except Exception:
                continue
            if record is not None and not item["error_id"]:
                item["error_id"] = record.error_id

    def _handle_worker_payload(self, payload: dict[str, Any]) -> None:
        if not isinstance(payload, dict):
            self._append_log("无法解析 worker 输出：JSON 顶层必须是对象。")
            return
        kind = payload.get("type")
        if kind not in {"progress", "result", "error"}:
            self._append_log("无法解析 worker 输出：消息类型无效。")
            return
        action_value = payload.get("action")
        if action_value is not None and not isinstance(action_value, str):
            self._append_log("无法解析 worker 输出：action 必须是文本。")
            return
        if kind == "progress":
            if payload.get("data") is not None and not isinstance(payload["data"], dict):
                self._append_log("无法解析 worker 输出：progress.data 必须是对象。")
                return
            if payload.get("job_id") is not None and not isinstance(payload["job_id"], str):
                self._append_log("无法解析 worker 输出：progress.job_id 必须是文本。")
                return
            if any(
                field in payload and not isinstance(payload[field], str)
                for field in ("name", "operation")
            ):
                self._append_log("无法解析 worker 输出：progress.name/operation 必须是文本。")
                return
            progress_data = payload.get("data") or {}
            progress_metrics = ("actual_fit_count", "proposal_count", "elapsed_seconds")
            if any(field in progress_data for field in progress_metrics) and not all(
                field in progress_data for field in progress_metrics
            ):
                self._append_log("无法解析 worker 输出：progress 搜索计数和耗时字段必须同时提供。")
                return
            if any(
                field in progress_data and not _nonnegative_worker_count(progress_data[field])
                for field in ("actual_fit_count", "proposal_count")
            ):
                self._append_log("无法解析 worker 输出：progress 拟合数和 proposal 数必须是非负整数。")
                return
            if (
                "elapsed_seconds" in progress_data
                and not _nonnegative_worker_duration(progress_data["elapsed_seconds"])
            ):
                self._append_log("无法解析 worker 输出：progress.elapsed_seconds 必须是有限的非负数。")
                return
        elif kind == "error":
            if not isinstance(payload.get("error_type"), str) or not isinstance(payload.get("message"), str):
                self._append_log("无法解析 worker 输出：error 消息缺少有效类型或说明。")
                return
            if payload.get("traceback") is not None and not isinstance(payload.get("traceback"), str):
                self._append_log("无法解析 worker 输出：traceback 必须是文本。")
                return
        elif kind == "result":
            phase = payload.get("phase")
            if phase == "queued":
                jobs = payload.get("jobs")
                if not isinstance(jobs, list) or any(
                    not isinstance(item, dict)
                    or not isinstance(item.get("job_id"), str)
                    or not item["job_id"]
                    for item in jobs
                ):
                    self._append_log("无法解析 worker 输出：queued.jobs 格式无效。")
                    return
            elif phase == "finished":
                outcomes = payload.get("outcomes")
                if not isinstance(outcomes, list) or any(not isinstance(item, dict) for item in outcomes):
                    self._append_log("无法解析 worker 输出：finished.outcomes 格式无效。")
                    return
                if any(
                    "diagnostic_errors" in item
                    and (
                        not isinstance(item["diagnostic_errors"], list)
                        or any(not isinstance(error, dict) for error in item["diagnostic_errors"])
                    )
                    for item in outcomes
                ):
                    self._append_log("无法解析 worker 输出：diagnostic_errors 格式无效。")
                    return
                if any(
                    ("error_id" in item and item["error_id"] is not None
                     and (not isinstance(item["error_id"], str)
                          or not _WORKER_ERROR_ID_RE.fullmatch(item["error_id"])))
                    or ("status" in item and not isinstance(item["status"], str))
                    or ("error" in item and item["error"] is not None
                        and not isinstance(item["error"], str))
                    or ("traceback" in item and item["traceback"] is not None
                        and not isinstance(item["traceback"], str))
                    for item in outcomes
                ):
                    self._append_log("无法解析 worker 输出：finished outcome 字段格式无效。")
                    return
                if any(
                    ("error_id" in error and error["error_id"] is not None
                     and (not isinstance(error["error_id"], str)
                          or not _WORKER_ERROR_ID_RE.fullmatch(error["error_id"])))
                    or any(
                        field in error and error[field] is not None
                        and not isinstance(error[field], str)
                        for field in ("error_type", "message", "traceback")
                    )
                    for item in outcomes
                    for error in item.get("diagnostic_errors", [])
                ):
                    self._append_log("无法解析 worker 输出：diagnostic_errors 字段格式无效。")
                    return
            elif phase == "selection_frozen":
                if not isinstance(payload.get("selection"), dict):
                    self._append_log("无法解析 worker 输出：selection 必须是对象。")
                    return
            elif phase == "finalized":
                final_selection = payload.get("final_selection")
                if not isinstance(final_selection, dict):
                    self._append_log("无法解析 worker 输出：final_selection 必须是对象。")
                    return
                if (
                    "state" in final_selection and not isinstance(final_selection["state"], str)
                ) or (
                    "test_evaluation_count" in final_selection
                    and not _nonnegative_worker_count(final_selection["test_evaluation_count"])
                ):
                    self._append_log("无法解析 worker 输出：final_selection 字段格式无效。")
                    return
            elif phase == "training_exploration_exported":
                result = payload.get("result")
                if not isinstance(result, dict):
                    self._append_log("无法解析 worker 输出：result 必须是对象。")
                    return
                artifact_paths = result.get("artifact_paths")
                if not isinstance(artifact_paths, dict) or not isinstance(artifact_paths.get("model"), str):
                    self._append_log("无法解析 worker 输出：result.artifact_paths.model 必须是文本。")
                    return
                if (
                    ("status" in result and not isinstance(result["status"], str))
                    or ("actual_fit_count" in result
                        and not _nonnegative_worker_count(result["actual_fit_count"]))
                    or ("test_evaluation_count" in result
                        and (
                            not _nonnegative_worker_count(result["test_evaluation_count"])
                            or result["test_evaluation_count"] != 0
                        ))
                ):
                    self._append_log("无法解析 worker 输出：training exploration result 字段格式无效。")
                    return
            elif phase == "control_requested":
                if not isinstance(payload.get("control"), str):
                    self._append_log("无法解析 worker 输出：control 必须是文本。")
                    return
        action = payload.get("action") or self._worker_action
        if kind == "progress":
            name = str(payload.get("name") or payload.get("operation") or "后台阶段")
            data = payload.get("data") or {}
            job_id = payload.get("job_id")
            fit_count = data.get("actual_fit_count")
            proposal_count = data.get("proposal_count")
            elapsed = data.get("elapsed_seconds")
            extra = f" · fit {fit_count} · proposals {proposal_count} · {float(elapsed):.1f}s" if fit_count is not None else ""
            self.status_label.setText(f"{job_id or ''} {name}{extra}".strip())
            self._append_log(f"{job_id or ''} {name}{extra}".strip())
            if job_id and job_id not in self._job_ids:
                self._job_ids.append(job_id)
            self.refresh_history()
            return
        if kind == "result":
            phase = payload.get("phase")
            if phase == "queued":
                self._job_ids = [str(item["job_id"]) for item in payload.get("jobs", [])]
                self.status_label.setText(f"已预检并入队 {len(self._job_ids)} 个模型/数据组合，搜索在后台运行。")
                self.refresh_history()
            elif phase == "finished":
                outcomes = payload.get("outcomes", [])
                failed = sum(item.get("status") == "failed" for item in outcomes)
                self.status_label.setText(f"队列搜索结束：{len(outcomes) - failed} 个完成，{failed} 个失败。")
                for outcome in outcomes:
                    if outcome.get("error_id"):
                        self._remember_error(
                            error_id=str(outcome.get("error_id")),
                            message=str(outcome.get("error", "批量任务失败")),
                            traceback_text=str(outcome.get("traceback", "")),
                        )
                    for item in outcome.get("diagnostic_errors", []):
                        self._remember_error(
                            error_id=str(item.get("error_id") or "") or None,
                            message=str(item.get("message") or item.get("error", "搜索 trial 失败")),
                            traceback_text=str(item.get("traceback", "")),
                            error_type=str(item.get("error_type") or "TrialError"),
                        )
                self.refresh_history()
            elif phase == "selection_frozen":
                self.status_label.setText("已冻结验证 winner；尚未重拟合或读取测试集。")
                self._append_log(f"冻结 selection：{payload.get('job_id')}")
                self.refresh_history()
            elif phase == "finalized":
                final = payload.get("final_selection", {})
                self.status_label.setText(
                    f"最终会话状态 {final.get('state')}；test_evaluation_count={final.get('test_evaluation_count')}。"
                )
                self._append_log(f"最终测试结果已缓存：{payload.get('job_id')}")
                self.refresh_history()
            elif phase == "training_exploration_exported":
                result = payload.get("result", {})
                paths = result.get("artifact_paths", {})
                verb = "复用" if result.get("status") == "cached" else "导出"
                self.status_label.setText(
                    f"训练探索结果已{verb}；fit={result.get('actual_fit_count')}，"
                    f"test_evaluation_count=0；模型：{paths.get('model', '')}"
                )
                self._append_log(f"训练探索模型已导出：{paths.get('model', '')}")
                self.refresh_history()
            elif phase == "control_requested":
                self.status_label.setText(f"已保存队列控制请求：{payload.get('control')}。")
                self.refresh_history()
            return
        if kind == "error":
            self._last_error = f"{payload.get('error_type', 'WorkerError')}: {payload.get('message', '')}"
            candidate_id = payload.get("error_id")
            safe_error_id = (
                candidate_id
                if isinstance(candidate_id, str) and _WORKER_ERROR_ID_RE.fullmatch(candidate_id)
                else None
            )
            self._remember_error(
                error_id=safe_error_id,
                message=self._last_error,
                traceback_text=str(payload.get("traceback") or ""),
                error_type=str(payload.get("error_type") or "WorkerError"),
            )
            self.status_label.setText(f"批量任务失败：{self._last_error}")
            error_id = safe_error_id
            suffix = f" · error_id={error_id}" if error_id else ""
            self._append_log(self._last_error + suffix)
            return

    def _process_finished(self, exit_code: int, _exit_status) -> None:
        self._read_stdout()
        self._flush_stderr()
        if self._stdout_buffer.strip():
            try:
                self._handle_worker_payload(json.loads(self._stdout_buffer.decode("utf-8")))
            except (UnicodeDecodeError, json.JSONDecodeError):
                self._append_log(self._stdout_buffer.decode("utf-8", errors="replace"))
            self._stdout_buffer = b""
        if exit_code != 0 and self._last_error is None:
            detail = self._stderr_text.strip().splitlines()[-1] if self._stderr_text.strip() else "未知 worker 错误"
            self._last_error = detail
            self._remember_error(
                error_id=None,
                message=detail,
                traceback_text=self._stderr_text,
                error_type="WorkerProcessError",
            )
            self.status_label.setText(f"worker 退出代码 {exit_code}：{detail}")
            self._append_log(self.status_label.text())
        self._record_external_errors()
        if self._temporary_request is not None:
            self._temporary_request.unlink(missing_ok=True)
            self._temporary_request = None
        if self.logging_controller is not None:
            try:
                error_id = next(
                    (
                        item["error_id"]
                        for item in reversed(self._error_details)
                        if item["error_id"]
                    ),
                    None,
                )
                self.logging_controller.record_process_exit(
                    exit_code,
                    context={"stage": self._worker_action or "batch_worker"},
                    error_id=error_id,
                )
            except Exception:
                pass
        self._worker_action = None
        self.refresh_history()
        self._refresh_controls()

    def _process_error(self, _error) -> None:
        if self.process.state() == QProcess.ProcessState.NotRunning:
            self._last_error = self.process.errorString()
            self._remember_error(
                error_id=None,
                message=self._last_error,
                traceback_text="",
                error_type="WorkerProcessError",
            )
            self.status_label.setText(f"无法启动批量 worker：{self._last_error}")
            self._append_log(self.status_label.text())
            self._record_external_errors()
            self._worker_action = None
            self._refresh_controls()

    @staticmethod
    def _comparison_group(job: dict[str, Any]) -> tuple[str, str, dict[str, Any]]:
        """Return a stable comparison key across every scoring-relevant dimension."""
        config = json.loads(job.get("config_json") or "{}")
        snapshot = json.loads(job.get("snapshot_json") or "{}")
        split_record = json.loads(job.get("split_json") or "{}")
        search_spec = json.loads(job.get("search_json") or "{}")
        dataset = config.get("dataset", {})
        objective = search_spec.get("objective", {})
        metric = objective.get("metric")
        direction = objective.get("direction") or ("min" if metric == "rmse" else "max")
        score_split = objective.get("split") or "validation"
        columns = {
            "features": snapshot.get("feature_columns", dataset.get("feature_columns")),
            "target": dataset.get("target_column"),
            "objective_labels": snapshot.get("objective_labels_column") or objective.get("objective_labels_column"),
        }
        group = {
            "dataset_id": job.get("dataset_id"),
            "source_sha256": snapshot.get("source_sha256", job.get("dataset_id")),
            "data_sha256": snapshot.get("data_sha256", job.get("dataset_id")),
            "sheet_name": dataset.get("sheet_name"),
            "split_sha256": snapshot.get("split_sha256") or split_record.get("split_sha256"),
            "split_seed": split_record.get("seed", config.get("split", {}).get("seed")),
            "task": job.get("task"),
            "columns": columns,
            "objective": objective,
            "metric": metric,
            "direction": direction,
            "score_split": score_split,
        }
        key = json.dumps(group, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
        data_id = str(group["data_sha256"] or group["dataset_id"] or "unknown")[:10]
        split_id = str(group["split_sha256"] or "unknown")[:10]
        columns_text = json.dumps(columns, ensure_ascii=False, separators=(",", ":"))
        label = (
            f"data={data_id} · split={split_id}/seed={group['split_seed']} · "
            f"task={group['task']} · columns={columns_text} · "
            f"objective={metric} {direction} · score={score_split}"
        )
        return label, key, group

    def refresh_history(self) -> None:
        current = self._selected_job()
        selected_job_id = current[0] if current else None
        root = Path(self.root_edit.text().strip()).expanduser().resolve()
        history_path = root / "history.sqlite3"
        if not history_path.is_file():
            self.jobs_table.blockSignals(True)
            self.jobs_table.setRowCount(0)
            self.jobs_table.blockSignals(False)
            self._render_selected_details()
            self._refresh_controls()
            return
        try:
            store = HistoryStore(history_path)
            jobs = store.list_jobs()
        except Exception as exc:
            self.status_label.setText(f"无法读取批量历史：{exc}")
            return
        grouped_jobs = []
        rank_entries = []
        for job in jobs:
            label, key, group = self._comparison_group(job)
            search_spec = json.loads(job["search_json"] or "{}")
            objective = search_spec.get("objective", {})
            direction = objective.get("direction") or ("min" if objective.get("metric") == "rmse" else "max")
            score_split = objective.get("split", "validation")
            trial_rows = store.list_trials(job["job_id"])
            scored = [
                (trial, _finite_objective_score(trial.get("objective_value")))
                for trial in trial_rows
                if trial["status"] in {"succeeded", "cached"}
                and _finite_objective_score(trial.get("objective_value")) is not None
            ]
            best = None
            best_score = None
            if scored:
                best, best_score = (min if direction == "min" else max)(scored, key=lambda item: item[1])
            best_result = json.loads(best["result_json"]) if best and best["result_json"] else {}
            row = {
                "label": label,
                "group_key": key,
                "group": group,
                "job": job,
                "direction": direction,
                "score_split": score_split,
                "best": best,
                "best_result": best_result,
                "score": best_score,
            }
            grouped_jobs.append(row)
            rank_entries.append({
                "job_id": job["job_id"],
                "comparison_group_key": key,
                "direction": direction,
                "score_split": score_split,
                "score": best_score,
                "created_at": job.get("created_at", ""),
            })
        ranks = _rank_validation_group_entries(rank_entries)
        grouped_jobs.sort(key=lambda row: (
            row["group_key"],
            ranks[row["job"]["job_id"]] is None,
            ranks[row["job"]["job_id"]] or 0,
            row["job"].get("created_at", ""),
            row["job"]["job_id"],
        ))
        self.jobs_table.blockSignals(True)
        self.jobs_table.setRowCount(len(grouped_jobs))
        for row_index, row in enumerate(grouped_jobs):
            job = row["job"]
            group_label, group_key, group_data = row["label"], row["group_key"], row["group"]
            score_split = row["score_split"]
            best, best_result, best_score = row["best"], row["best_result"], row["score"]
            score_label = "训练探索" if score_split == "train_exploratory" else "验证"
            rank = ranks[job["job_id"]]
            rank_label = (
                "不排名·训练探索" if score_split == "train_exploratory"
                else "无分数" if best_score is None
                else str(rank)
            )
            values = (
                job["job_id"], group_label, job["task"], job["model_id"], job["status"],
                f"{job['actual_fit_count']}/{json.loads(job['budget_json']).get('max_actual_fits', '')}",
                f"{job['proposal_count']}/{json.loads(job['budget_json']).get('max_proposals', '')}",
                f"{float(job.get('elapsed_seconds', 0.0)):.1f}",
                rank_label,
                "" if best_score is None else f"{score_label}：{best_score}",
                json.dumps(best_result.get("parameters", {}), ensure_ascii=False, sort_keys=True),
            )
            for column, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                if column == 1:
                    item.setToolTip(group_key)
                    item.setData(Qt.ItemDataRole.UserRole, group_data)
                if column == 8:
                    item.setToolTip(
                        "仅在同一比较组内对有限 validation 分数排名；同分共享 competition rank。"
                        "训练探索与无分数任务不排名。"
                    )
                if column == 9:
                    item.setToolTip(f"{score_label}分数；分组身份包含评分分区。")
                if column == 0:
                    item.setData(Qt.ItemDataRole.UserRole, job["artifact_dir"])
                self.jobs_table.setItem(row_index, column, item)
        self.jobs_table.resizeColumnsToContents()
        if grouped_jobs:
            target_row = next(
                (index for index, row in enumerate(grouped_jobs) if row["job"]["job_id"] == selected_job_id),
                0,
            )
            self.jobs_table.selectRow(target_row)
        self.jobs_table.blockSignals(False)
        self._refresh_controls()
        self._render_selected_details()

    def _selected_job_changed(self) -> None:
        self._refresh_controls()
        self._render_selected_details()

    def _render_selected_details(self) -> None:
        selected = self._selected_job()
        self.job_curve_axes.clear()
        if selected is None:
            self.job_details_text.clear()
            self.job_curve_status.setText("选择已完成的深度模型任务以查看 winner 损失曲线。")
            self.job_curve_axes.text(0.5, 0.5, "未选择任务", ha="center", va="center", color="#475569")
            self.job_curve_axes.set_axis_off()
            self.job_curve_canvas.draw_idle()
            return
        job_id, _artifact_dir = selected
        try:
            store = HistoryStore(self._history_path())
            job = store.get_job(job_id)
            trials = store.list_trials(job_id)
            events = store.list_events(job_id)
        except Exception as exc:
            self.job_details_text.setPlainText(f"无法读取所选任务详情：{exc}")
            self.job_curve_status.setText("无法读取损失曲线。")
            self.job_curve_axes.text(0.5, 0.5, "详情读取失败", ha="center", va="center", color="#B91C1C")
            self.job_curve_axes.set_axis_off()
            self.job_curve_canvas.draw_idle()
            return
        if job is None:
            self.job_details_text.setPlainText("所选任务已从历史中移除。")
            self.job_curve_status.setText("所选任务不存在。")
            self.job_curve_axes.text(0.5, 0.5, "任务不存在", ha="center", va="center", color="#475569")
            self.job_curve_axes.set_axis_off()
            self.job_curve_canvas.draw_idle()
            return

        search_spec = json.loads(job.get("search_json") or "{}")
        objective = search_spec.get("objective", {})
        direction = objective.get("direction") or ("min" if objective.get("metric") == "rmse" else "max")
        scored = [
            (trial, _finite_objective_score(trial.get("objective_value")))
            for trial in trials
            if trial.get("status") in {"succeeded", "cached"}
            and _finite_objective_score(trial.get("objective_value")) is not None
        ]
        best = (min if direction == "min" else max)(scored, key=lambda item: item[1]) if scored else None
        best_result = json.loads(best[0].get("result_json") or "{}") if best else {}
        training_curves = best_result.get("training_curves")
        curve_source = "搜索 winner"
        for event in reversed(events):
            if event.get("event_type") != "test_completed":
                continue
            try:
                payload = json.loads(event.get("payload_json") or "{}")
            except json.JSONDecodeError:
                continue
            result = (payload.get("data") or {}).get("result") or {}
            refit_curves = result.get("training_curves")
            if isinstance(refit_curves, dict) and refit_curves.get("fit_scope") == "train_validation":
                training_curves = refit_curves
                curve_source = "最终重拟合"
                break

        lines = [
            f"任务：{job_id}  ·  {job['model_id']}  ·  {job['status']}",
            f"评分：{objective.get('metric', '未设置')} / {objective.get('split', 'validation')}；"
            + (f"winner={best[1]:.6g}" if best else "尚无有限 winner 分数"),
        ]
        if best:
            lines.append("winner 参数：" + json.dumps(best_result.get("parameters", {}), ensure_ascii=False, sort_keys=True))
        receipt_text = job.get("sequence_plan_receipt_json")
        if receipt_text:
            try:
                receipt = json.loads(receipt_text)
                summary = sequence_plan_summary_from_manifest(receipt.get("manifest"))
            except (json.JSONDecodeError, ValueError, TypeError):
                summary = None
            if summary is not None:
                lines.append("序列划分（各比例按三份的总量计算）：")
                for split_name in ("train", "validation", "test"):
                    part = summary["splits"][split_name]
                    def ratio(field: str) -> str:
                        value = part.get(f"{field}_proportion")
                        return "—" if value is None else f"{value:.1%}"
                    lines.append(
                        f"  {split_name}: 源行 {part['source_rows']} ({ratio('source_rows')}) · "
                        f"组 {part['groups']} ({ratio('groups')}) · "
                        f"窗口 {part['windows']} ({ratio('windows')})"
                    )
        self.job_details_text.setPlainText("\n".join(lines))
        self._draw_batch_training_curves(training_curves, curve_source)

    def _draw_batch_training_curves(self, curves: Any, source: str) -> None:
        self.job_curve_axes.clear()
        if not isinstance(curves, dict):
            self.job_curve_status.setText("所选任务没有 skorch 逐 epoch 损失曲线。")
            self.job_curve_axes.text(0.5, 0.5, "没有可绘制的训练曲线", ha="center", va="center", color="#475569")
            self.job_curve_axes.set_axis_off()
            self.job_curve_canvas.draw_idle()
            return
        epochs = curves.get("epochs")
        train_loss = curves.get("train_loss")
        validation_loss = curves.get("validation_loss")
        validation_available = curves.get("validation_available") is True
        valid = (
            isinstance(epochs, list) and isinstance(train_loss, list)
            and bool(epochs) and len(epochs) == len(train_loss)
            and all(
                isinstance(value, (int, float)) and not isinstance(value, bool)
                and math.isfinite(float(value))
                for value in [*epochs, *train_loss]
            )
        )
        if validation_available:
            valid = valid and isinstance(validation_loss, list) and len(validation_loss) == len(epochs) and all(
                isinstance(value, (int, float)) and not isinstance(value, bool)
                and math.isfinite(float(value)) for value in validation_loss
            )
        elif validation_loss is not None:
            valid = False
        if not valid:
            self.job_curve_status.setText("训练曲线记录格式无效。")
            self.job_curve_axes.text(0.5, 0.5, "训练曲线数据无效", ha="center", va="center", color="#B91C1C")
            self.job_curve_axes.set_axis_off()
            self.job_curve_canvas.draw_idle()
            return
        epochs = [int(value) for value in epochs]
        self.job_curve_axes.plot(epochs, train_loss, color="#2563EB", marker="o", markersize=3, label="训练损失")
        if validation_available:
            self.job_curve_axes.plot(epochs, validation_loss, color="#F59E0B", marker="o", markersize=3, label="验证损失")
        if curves.get("fit_scope") == "train_validation":
            self.job_curve_status.setText(f"{source}曲线：train+validation 重拟合；未计算验证损失。")
            self.job_curve_axes.set_title("训练 + 验证重拟合损失", loc="left", fontsize=10)
        else:
            self.job_curve_status.setText(f"{source}逐 epoch 实际训练曲线。")
            self.job_curve_axes.set_title("训练 / 验证损失", loc="left", fontsize=10)
        self.job_curve_axes.set_xlabel("Epoch")
        self.job_curve_axes.set_ylabel("Loss")
        self.job_curve_axes.grid(color="#CBD5E1", linewidth=0.7, alpha=0.75)
        self.job_curve_axes.legend(frameon=False, fontsize=8)
        self.job_curve_axes.spines["top"].set_visible(False)
        self.job_curve_axes.spines["right"].set_visible(False)
        self.job_curve_canvas.draw_idle()

    def _control_selected(self, action: str) -> None:
        selected = self._selected_job()
        if selected is None:
            self._show_error("请先在历史表中选择一个队列任务。")
            return
        job_id, artifact_dir = selected
        history = self._history_path()
        active = self.process.state() != QProcess.ProcessState.NotRunning
        try:
            job = HistoryStore(history).get_job(job_id)
            process_active = bool(job and (Path(job["artifact_dir"]) / "runner.lock").exists())
            request_job_control(history, job_id, action, worker_active=process_active)
            if action == "resume" and not process_active:
                self._start_worker(
                    ["batch-run", "--history", str(history), "--job-id", job_id, "--parallel-workers", "1"],
                    action="batch-run",
                )
            else:
                self.status_label.setText(f"已向 {job_id} 保存 {action} 请求。")
                self.refresh_history()
        except Exception as exc:
            self._show_error(f"无法执行队列控制：{exc}")

    def _freeze_selected(self) -> None:
        selected = self._selected_job()
        if selected is None:
            self._show_error("请先选择已完成且有 validation winner 的任务。")
            return
        job_id, _artifact_dir = selected
        self._start_worker(
            ["batch-freeze", "--history", str(self._history_path()), "--job-id", job_id],
            action="batch-freeze",
        )

    def _export_exploration_selected(self) -> None:
        selected = self._selected_job()
        if selected is None:
            self._show_error("请先选择已完成的 train_exploratory 任务。")
            return
        job_id, _artifact_dir = selected
        root = Path(self.root_edit.text().strip()).expanduser().resolve()
        output_dir = root / "exports" / f"training-exploration-{job_id}"
        self._start_worker(
            [
                "batch-export-exploration", "--history", str(self._history_path()),
                "--job-id", job_id, "--output", str(output_dir),
            ],
            action="batch-export-exploration",
        )

    def _finalize_selected(self) -> None:
        selected = self._selected_job()
        if selected is None:
            self._show_error("请先选择已冻结 validation winner 的任务。")
            return
        job_id, artifact_dir = selected
        root = Path(self.root_edit.text().strip()).expanduser().resolve()
        export_dir = root / "exports" / job_id
        arguments = ["batch-finalize", "--history", str(self._history_path()), "--job-id", job_id]
        if not export_dir.exists():
            arguments.extend(["--output", str(export_dir)])
        self._start_worker(arguments, action="batch-finalize")

    def export_comparison(self) -> None:
        root = Path(self.root_edit.text().strip()).expanduser().resolve()
        history_path = root / "history.sqlite3"
        if not history_path.is_file():
            self._show_error("当前目录还没有可导出的队列历史。")
            return
        destination, _ = QFileDialog.getSaveFileName(
            self, "导出批量结果对比", str(root / "comparison.csv"), "CSV 文件 (*.csv)"
        )
        if not destination:
            return
        try:
            store = HistoryStore(history_path)
            rows: list[dict[str, Any]] = []
            rank_entries = []
            created_at_by_job: dict[str, str] = {}
            for job in store.list_jobs():
                config = json.loads(job["config_json"] or "{}")
                spec = json.loads(job["search_json"] or "{}")
                objective = spec.get("objective", {})
                direction = objective.get("direction") or ("min" if objective.get("metric") == "rmse" else "max")
                score_split = objective.get("split", "validation")
                group_label, group_key, group = self._comparison_group(job)
                trials = [
                    (item, _finite_objective_score(item.get("objective_value")))
                    for item in store.list_trials(job["job_id"])
                    if item["status"] in {"succeeded", "cached"}
                    and _finite_objective_score(item.get("objective_value")) is not None
                ]
                winner, winner_score = (
                    (min if direction == "min" else max)(trials, key=lambda item: item[1])
                    if trials else (None, None)
                )
                permission = store.get_test_permission(job["job_id"])
                rows.append({
                    "job_id": job["job_id"], "dataset_id": job["dataset_id"],
                    "comparison_group": group_label, "comparison_group_key": group_key,
                    "source_sha256": group["source_sha256"], "data_sha256": group["data_sha256"],
                    "dataset_path": config.get("dataset", {}).get("source_path"),
                    "sheet_name": group["sheet_name"], "split_sha256": group["split_sha256"],
                    "split_seed": group["split_seed"],
                    "task": job["task"], "model_id": job["model_id"],
                    "columns": json.dumps(group["columns"], ensure_ascii=False),
                    "objective": objective.get("metric"), "direction": direction,
                    "score_split": score_split,
                    "group_rank": None,
                    "objective_settings": json.dumps(objective, ensure_ascii=False, sort_keys=True),
                    "status": job["status"], "actual_fit_count": job["actual_fit_count"],
                    "proposal_count": job["proposal_count"],
                    "elapsed_seconds": job.get("elapsed_seconds", 0),
                    "winner_trial_id": winner["trial_id"] if winner else None,
                    "winner_score": winner_score,
                    "winner_parameters": json.dumps(
                        json.loads(winner["result_json"]).get("parameters", {})
                        if winner and winner["result_json"] else {},
                        ensure_ascii=False,
                    ),
                    "test_permission_state": permission["state"] if permission else "not used",
                })
                rank_entries.append({
                    "job_id": job["job_id"],
                    "comparison_group_key": group_key,
                    "direction": direction,
                    "score_split": score_split,
                    "score": winner_score,
                    "created_at": job.get("created_at", ""),
                })
                created_at_by_job[job["job_id"]] = job.get("created_at", "")
            ranks = _rank_validation_group_entries(rank_entries)
            for row in rows:
                row["group_rank"] = ranks[row["job_id"]]
            rows.sort(key=lambda row: (
                row["comparison_group_key"],
                row["group_rank"] is None,
                row["group_rank"] or 0,
                created_at_by_job[row["job_id"]],
                row["job_id"],
            ))
            fields = list(rows[0]) if rows else ["job_id", "dataset_id", "task", "model_id", "status"]
            with Path(destination).open("w", encoding="utf-8-sig", newline="") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(rows)
            self.status_label.setText(f"对比 CSV 已导出：{destination}")
        except Exception as exc:
            self._show_error(f"导出对比失败：{exc}")

    def _history_path(self) -> Path:
        return Path(self.root_edit.text().strip()).expanduser().resolve() / "history.sqlite3"

    def _selected_job(self) -> tuple[str, Path] | None:
        row = self.jobs_table.currentRow()
        if row < 0:
            return None
        id_item = self.jobs_table.item(row, 0)
        if id_item is None:
            return None
        return id_item.text(), Path(id_item.data(Qt.ItemDataRole.UserRole))

    def _plot_selected(self) -> None:
        selected = self._selected_job()
        if selected is None:
            self.status_label.setText("请先选择一条已完成的批量任务。")
            return
        job_id, _artifact_dir = selected
        try:
            collection = load_batch_plot_sources(self._history_path(), job_id)
        except Exception as exc:
            self._show_error(f"无法构造所选任务的可信绘图来源：{exc}")
            return
        if not collection.sources:
            details = "；".join(reason for _partition, reason in collection.unavailable)
            self.status_label.setText(details or "所选任务没有可绘制的缓存分区。")
            return
        chooser = PlotSourceChooserDialog(
            collection.sources,
            unavailable=collection.unavailable_by_partition(),
            parent=self,
        )
        if chooser.exec() != QDialog.DialogCode.Accepted:
            return
        try:
            payload = build_plot_payload(chooser.selected_source, chooser.selected_spec)
            dialog = PlotDialog([payload], parent=self)
        except Exception as exc:
            self.status_label.setText(f"无法生成所选结果图：{exc}")
            return
        self._plot_dialogs.append(dialog)
        dialog.destroyed.connect(
            lambda _obj=None, target=dialog: self._plot_dialogs.remove(target)
            if target in self._plot_dialogs else None
        )
        dialog.show()
        dialog.raise_()
        dialog.activateWindow()

    def _refresh_controls(self, *_args) -> None:
        running = self.process.state() != QProcess.ProcessState.NotRunning
        selected = self._selected_job()
        job = None
        if selected:
            try:
                job = HistoryStore(self._history_path()).get_job(selected[0])
            except Exception:
                job = None
        self.start_button.setEnabled(not running)
        self.add_current_button.setEnabled(not running)
        self.add_files_button.setEnabled(not running)
        self.pause_button.setEnabled(bool(job and job["status"] == "running"))
        self.resume_button.setEnabled(bool(job and job["status"] == "paused"))
        self.cancel_button.setEnabled(bool(job and job["status"] in {"queued", "running", "pausing", "paused"}))
        try:
            training_exploratory = bool(
                job and json.loads(job["search_json"] or "{}").get("objective", {}).get("split") == "train_exploratory"
            )
        except (TypeError, json.JSONDecodeError):
            training_exploratory = False
        self.freeze_button.setEnabled(
            bool(job and job["status"] == "completed" and not training_exploratory and not running)
        )
        self.freeze_button.setToolTip(
            "训练探索结果没有独立 validation winner，不能冻结或进入最终测试。"
            if training_exploratory else "冻结独立验证集 winner；此操作不会读取测试集。"
        )
        self.export_exploration_button.setEnabled(
            bool(job and job["status"] == "completed" and training_exploratory and not running)
        )
        self.export_exploration_button.setToolTip(
            "导出此已完成任务的现有训练 session，不重拟合、不读取 validation/test。"
            if training_exploratory else "仅适用于评分分区为 train_exploratory 的已完成搜索。"
        )
        self.plot_result_button.setEnabled(bool(job and job["status"] == "completed" and not running))
        self.plot_result_button.setToolTip(
            "从当前 job 所有的结果缓存、快照收据与分区行位置构造图表；无效的 test 凭据不会解锁测试图。"
        )
        artifact_dir = Path(job["artifact_dir"]) if job else None
        selection_saved = bool(artifact_dir and (artifact_dir / "frozen-selection.joblib").is_file())
        self.finalize_button.setEnabled(bool(job and selection_saved and not training_exploratory and not running))
        self.finalize_button.setToolTip(
            "需要先冻结独立 validation winner；train_exploratory 结果不能进入最终重拟合/测试。"
        )

    def _browse_root(self) -> None:
        directory = QFileDialog.getExistingDirectory(self, "选择批次与历史目录", self.root_edit.text())
        if directory:
            self.root_edit.setText(directory)
            self.refresh_history()

    def _show_error(self, message: str) -> None:
        self.status_label.setText(message)
        self._append_log(f"错误：{message}")
        QMessageBox.warning(self, "批量搜索", message)

    def _append_log(self, message: str) -> None:
        self.log_text.appendPlainText(message)
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
