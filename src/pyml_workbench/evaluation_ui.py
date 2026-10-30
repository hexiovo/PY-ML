"""Reusable evaluation, chart-page, and export UI components."""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

# Load dateutil before PySide activates Shiboken's lazy import hook; in some
# Windows Python 3.12 environments, otherwise importing six.moves fails.
from dateutil import rrule as _dateutil_rrule  # noqa: F401

from PySide6.QtCore import QTimer, QUrl, Qt
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
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
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg

from .plotting import available_plot_specs, build_plot_payload, render_plot_payload


EVALUATION_HOLDOUT = "holdout"
EVALUATION_KFOLD = "kfold"
DEFAULT_FOLD_COUNT = 5
DEFAULT_TEST_FRACTION = 0.2

# max_fits follows the existing candidate-count contract. Present it in the UI
# as a candidate-model limit, never as the total number of underlying fit calls.
NEW_TASK_RESOURCE_DEFAULTS = {
    "workers": 1,
    "numeric_threads_per_worker": 1,
    "max_fits": 10,
    "proposal_cap": 30,
    "time_limit_seconds": 600,
}

EVALUATION_METHOD_CHOICES = (
    ("留出法", EVALUATION_HOLDOUT),
    ("K 折交叉验证", EVALUATION_KFOLD),
)

PARTITION_LABELS = {
    "oof": "交叉验证折外预测（OOF）",
    "validation": "留出验证集",
    "test": "最终测试集",
    "train": "训练集",
}


def normalize_evaluation_settings(raw: Mapping[str, Any] | None) -> dict[str, Any]:
    """Keep old configs/caches on holdout unless they explicitly select K-fold."""
    settings = dict(raw or {})
    method = settings.get("evaluation_method", EVALUATION_HOLDOUT)
    if method not in (EVALUATION_HOLDOUT, EVALUATION_KFOLD):
        method = EVALUATION_HOLDOUT
    folds_value = settings.get("n_splits", DEFAULT_FOLD_COUNT)
    try:
        folds = int(folds_value)
    except (TypeError, ValueError, OverflowError):
        folds = DEFAULT_FOLD_COUNT
    if isinstance(folds_value, bool) or not 2 <= folds <= 10:
        folds = DEFAULT_FOLD_COUNT
    settings["evaluation_method"] = method
    settings["n_splits"] = folds
    return settings


def new_task_resource_defaults(
    existing_values: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Fill missing new-task budgets without replacing explicit user choices."""
    values: dict[str, Any] = dict(NEW_TASK_RESOURCE_DEFAULTS)
    if existing_values is not None:
        values.update(
            {
                key: existing_values[key]
                for key in NEW_TASK_RESOURCE_DEFAULTS
                if key in existing_values
            }
        )
    return values


def resource_explanation(n_splits: int = DEFAULT_FOLD_COUNT) -> str:
    """Chinese UI copy for evaluation and search-budget controls."""
    try:
        folds = int(n_splits)
    except (TypeError, ValueError, OverflowError):
        folds = DEFAULT_FOLD_COUNT
    if not 2 <= folds <= 10:
        folds = DEFAULT_FOLD_COUNT
    return (
        "K 折交叉验证先保留 20% 独立测试集，仅在其余开发集上比较候选；"
        f"{folds} 折时，一个候选通常需要训练 {folds} 次，折数越多通常增加总 CPU 时间。"
        "并行 worker 增多会同时运行更多算法，增加 CPU 和内存占用；"
        "独立 K 折的训练预算按候选计数；高级评价配置的交叉验证按拟合次数计数，每个候选占用 1 + 折数。"
        "训练预算、提案上限或时限调高允许更多搜索，但不保证结果更好，"
        "达到任一预算后停止。旧任务和旧缓存继续按留出法解释。"
    )


def create_evaluation_controls(
    parent: QWidget,
    initial_settings: Mapping[str, Any] | None = None,
) -> tuple[QGroupBox, dict[str, QWidget]]:
    """Create method/K controls, preserving any explicit saved selection."""
    panel = QGroupBox("评估方式", parent)
    form = QFormLayout(panel)
    settings = normalize_evaluation_settings(initial_settings)

    method = QComboBox(panel)
    for label, value in EVALUATION_METHOD_CHOICES:
        method.addItem(label, value)
    method.setCurrentIndex(method.findData(settings["evaluation_method"]))

    folds_label = QLabel("折数 K", panel)
    folds = QSpinBox(panel)
    folds.setRange(2, 10)
    folds.setValue(settings["n_splits"])

    help_label = QLabel(resource_explanation(DEFAULT_FOLD_COUNT), panel)
    help_label.setWordWrap(True)

    form.addRow("方法", method)
    form.addRow(folds_label, folds)
    form.addRow(help_label)

    def update_fold_visibility(_index: int = -1) -> None:
        visible = method.currentData() == EVALUATION_KFOLD
        folds_label.setVisible(visible)
        folds.setVisible(visible)
        help_label.setText(resource_explanation(folds.value()))

    method.currentIndexChanged.connect(update_fold_visibility)
    folds.valueChanged.connect(lambda _value: help_label.setText(resource_explanation(folds.value())))
    update_fold_visibility()

    return panel, {
        "evaluation_method": method,
        "n_splits": folds,
        "resource_help": help_label,
    }


def set_kfold_availability(
    controls: Mapping[str, QWidget],
    *,
    supported: bool,
    reason: str | None = None,
) -> None:
    """Disable K-fold for deep/sequence workflows using the caller's support check."""
    method = controls["evaluation_method"]
    folds = controls["n_splits"]
    if not isinstance(method, QComboBox) or not isinstance(folds, QSpinBox):
        raise TypeError("Evaluation controls were not created by create_evaluation_controls().")

    kfold_index = method.findData(EVALUATION_KFOLD)
    item = method.model().item(kfold_index) if kfold_index >= 0 else None
    if item is not None:
        item.setEnabled(supported)
    method.setProperty("kfold_supported", supported)
    method.setToolTip(reason or "")
    help_label = controls.get("resource_help")
    if isinstance(help_label, QLabel):
        help_label.setToolTip(reason or "")
    folds.setEnabled(supported and method.currentData() == EVALUATION_KFOLD)


@dataclass(frozen=True)
class ExportRequest:
    format_id: str
    output_directory: Path
    include_sample_results: bool
    include_model: bool
    include_comparison_png: bool
    timestamp_subdirectory: bool
    format_ids: tuple[str, ...] = ()

    @property
    def selected_formats(self) -> tuple[str, ...]:
        """Return all selected formats while keeping the single-format API usable."""
        return self.format_ids or ((self.format_id,) if self.format_id else ())


class ExportOptionsDialog(QDialog):
    """Collect export choices and preview a target path without creating files."""

    FORMATS = (
        ("CSV", "csv"),
        ("Excel 工作簿（XLSX）", "xlsx"),
        ("JSON", "json"),
        ("HTML", "html"),
    )

    def __init__(
        self,
        parent: QWidget,
        initial: ExportRequest,
        preview_path: Callable[[ExportRequest], str | Path],
        *,
        allow_sample_results: bool = True,
        sample_results_reason: str = "",
        allow_model: bool = True,
        model_reason: str = "",
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("导出结果")
        self.setModal(True)
        self._preview_path = preview_path
        self._request: ExportRequest | None = None

        root = QVBoxLayout(self)
        form = QFormLayout()

        initial_formats = set(initial.selected_formats)
        self.format_checkboxes: dict[str, QCheckBox] = {}
        format_row = QWidget(self)
        format_layout = QHBoxLayout(format_row)
        format_layout.setContentsMargins(0, 0, 0, 0)
        for label, value in self.FORMATS:
            checkbox = QCheckBox(label, self)
            checkbox.setChecked(value in initial_formats)
            checkbox.stateChanged.connect(self._update_preview)
            self.format_checkboxes[value] = checkbox
            format_layout.addWidget(checkbox)

        self.directory_edit = QLineEdit(str(initial.output_directory), self)
        self.browse_button = QPushButton("浏览…", self)
        directory_row = QWidget(self)
        directory_layout = QHBoxLayout(directory_row)
        directory_layout.setContentsMargins(0, 0, 0, 0)
        directory_layout.addWidget(self.directory_edit, 1)
        directory_layout.addWidget(self.browse_button)

        form.addRow("格式", format_row)
        form.addRow("保存位置", directory_row)
        root.addLayout(form)

        self.sample_checkbox = QCheckBox("逐样本结果", self)
        self.sample_checkbox.setChecked(initial.include_sample_results)
        self.sample_checkbox.setEnabled(allow_sample_results)
        if not allow_sample_results:
            self.sample_checkbox.setChecked(False)
            self.sample_checkbox.setToolTip(sample_results_reason)
        self.model_checkbox = QCheckBox("模型文件", self)
        self.model_checkbox.setChecked(initial.include_model)
        self.model_checkbox.setEnabled(allow_model)
        if not allow_model:
            self.model_checkbox.setChecked(False)
            self.model_checkbox.setToolTip(model_reason)
        self.png_checkbox = QCheckBox("比较图 PNG", self)
        self.png_checkbox.setChecked(initial.include_comparison_png)
        self.timestamp_checkbox = QCheckBox("创建唯一时间戳子目录", self)
        self.timestamp_checkbox.setChecked(initial.timestamp_subdirectory)
        root.addWidget(self.sample_checkbox)
        root.addWidget(self.model_checkbox)
        root.addWidget(self.png_checkbox)
        root.addWidget(self.timestamp_checkbox)

        self.preview_label = QLabel(self)
        self.preview_label.setWordWrap(True)
        root.addWidget(self.preview_label)

        self.buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok
            | QDialogButtonBox.StandardButton.Cancel,
            parent=self,
        )
        self.buttons.accepted.connect(self._accept_if_valid)
        self.buttons.rejected.connect(self.reject)
        root.addWidget(self.buttons)

        self.browse_button.clicked.connect(self._choose_directory)
        self.directory_edit.textChanged.connect(self._update_preview)
        self.sample_checkbox.stateChanged.connect(self._update_preview)
        self.model_checkbox.stateChanged.connect(self._update_preview)
        self.png_checkbox.stateChanged.connect(self._update_preview)
        self.timestamp_checkbox.stateChanged.connect(self._update_preview)
        self._update_preview()

    @property
    def request(self) -> ExportRequest | None:
        return self._request

    def _current_request(self) -> ExportRequest:
        selected_formats = tuple(
            format_id for format_id, checkbox in self.format_checkboxes.items()
            if checkbox.isChecked()
        )
        return ExportRequest(
            format_id=selected_formats[0] if selected_formats else "",
            output_directory=Path(self.directory_edit.text()).expanduser(),
            include_sample_results=self.sample_checkbox.isChecked(),
            include_model=self.model_checkbox.isChecked(),
            include_comparison_png=self.png_checkbox.isChecked(),
            timestamp_subdirectory=self.timestamp_checkbox.isChecked(),
            format_ids=selected_formats,
        )

    def _choose_directory(self) -> None:
        selected = QFileDialog.getExistingDirectory(
            self,
            "选择结果保存位置",
            self.directory_edit.text(),
        )
        if selected:
            self.directory_edit.setText(selected)

    def _update_preview(self, *_args: Any) -> None:
        directory = self.directory_edit.text().strip()
        ok_button = self.buttons.button(QDialogButtonBox.StandardButton.Ok)
        if not directory:
            self.preview_label.setText("请选择保存位置。")
            ok_button.setEnabled(False)
            return
        if not any(checkbox.isChecked() for checkbox in self.format_checkboxes.values()):
            self.preview_label.setText("至少选择一种导出格式。")
            ok_button.setEnabled(False)
            return
        try:
            target = self._preview_path(self._current_request())
        except Exception as exc:
            self.preview_label.setText(f"无法预览目标路径：{exc}")
            ok_button.setEnabled(False)
            return
        self.preview_label.setText(f"确认后将导出到：{target}")
        ok_button.setEnabled(True)

    def _accept_if_valid(self) -> None:
        self._update_preview()
        ok_button = self.buttons.button(QDialogButtonBox.StandardButton.Ok)
        if ok_button.isEnabled():
            self._request = self._current_request()
            self.accept()


def show_export_complete(parent: QWidget, actual_path: str | Path) -> None:
    """Show the exact exporter-returned path and offer, but do not force, folder opening."""
    exported_path = Path(actual_path).expanduser().resolve()
    message = QMessageBox(parent)
    message.setIcon(QMessageBox.Icon.Information)
    message.setWindowTitle("导出完成")
    message.setText(f"结果已导出到：\n{exported_path}")
    open_button = message.addButton("打开文件夹", QMessageBox.ButtonRole.ActionRole)
    close_button = message.addButton("关闭", QMessageBox.ButtonRole.RejectRole)
    message.setDefaultButton(close_button)
    message.exec()
    if message.clickedButton() is open_button:
        folder = exported_path if exported_path.is_dir() else exported_path.parent
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))


def run_export_dialog(
    parent: QWidget,
    initial: ExportRequest,
    preview_path: Callable[[ExportRequest], str | Path],
    exporter: Callable[[ExportRequest], str | Path],
    *,
    allow_sample_results: bool = True,
    sample_results_reason: str = "",
    allow_model: bool = True,
    model_reason: str = "",
) -> Path | None:
    """Return None on cancel; call the existing exporter only after confirmation."""
    dialog = ExportOptionsDialog(
        parent,
        initial,
        preview_path,
        allow_sample_results=allow_sample_results,
        sample_results_reason=sample_results_reason,
        allow_model=allow_model,
        model_reason=model_reason,
    )
    if dialog.exec() != QDialog.DialogCode.Accepted or dialog.request is None:
        return None
    actual_path = Path(exporter(dialog.request)).expanduser()
    show_export_complete(parent, actual_path)
    return actual_path


@dataclass(frozen=True)
class ChartSourceChoice:
    """Adapter output for one algorithm/partition pair from the result cache."""

    algorithm_key: str
    algorithm_label: str
    source: Any


class ResultChartPage(QWidget):
    """Responsive chart tab backed by the plotting.py public cache API."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._choices: tuple[ChartSourceChoice, ...] = ()
        self._chart_title_label: QLabel | None = None
        self._chart_note_label: QLabel | None = None
        self._chart_canvas: FigureCanvasQTAgg | None = None

        root = QVBoxLayout(self)
        filters = QHBoxLayout()

        self.algorithm_combo = QComboBox(self)
        self.partition_combo = QComboBox(self)
        self.plot_combo = QComboBox(self)
        filters.addWidget(self.algorithm_combo, 2)
        filters.addWidget(self.partition_combo, 1)
        filters.addWidget(self.plot_combo, 2)
        root.addLayout(filters)

        self.status_label = QLabel("暂无可用缓存结果。", self)
        self.status_label.setWordWrap(True)
        root.addWidget(self.status_label)

        self.chart_scroll_area = QScrollArea(self)
        self.chart_scroll_area.setObjectName("resultChartScrollArea")
        self.chart_scroll_area.setWidgetResizable(True)
        self.chart_scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.chart_scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.chart_host = QWidget()
        self.chart_layout = QVBoxLayout(self.chart_host)
        self.chart_layout.setContentsMargins(0, 0, 0, 0)
        self.chart_layout.setSpacing(4)
        self.chart_scroll_area.setWidget(self.chart_host)
        root.addWidget(self.chart_scroll_area, 1)

        self.algorithm_combo.currentIndexChanged.connect(self._refresh_partitions)
        self.partition_combo.currentIndexChanged.connect(self._refresh_specs)
        self.plot_combo.currentIndexChanged.connect(self._render_current)

    def refresh_sources(self, choices: Sequence[ChartSourceChoice]) -> None:
        """Refresh after result-cache changes; the caller supplies the cache adapter."""
        self._choices = tuple(choices)
        selected_algorithm = self.algorithm_combo.currentData()

        self.algorithm_combo.blockSignals(True)
        self.algorithm_combo.clear()
        algorithms: dict[str, str] = {}
        for choice in self._choices:
            algorithms.setdefault(choice.algorithm_key, choice.algorithm_label)
        for key, label in algorithms.items():
            self.algorithm_combo.addItem(label, key)
        index = self.algorithm_combo.findData(selected_algorithm)
        if index >= 0:
            self.algorithm_combo.setCurrentIndex(index)
        self.algorithm_combo.blockSignals(False)
        self._refresh_partitions()

    def _choices_for_algorithm(self) -> tuple[ChartSourceChoice, ...]:
        key = self.algorithm_combo.currentData()
        return tuple(choice for choice in self._choices if choice.algorithm_key == key)

    def _selected_choice(self) -> ChartSourceChoice | None:
        partition = self.partition_combo.currentData()
        return next(
            (
                choice
                for choice in self._choices_for_algorithm()
                if getattr(choice.source, "partition", None) == partition
            ),
            None,
        )

    def _refresh_partitions(self, *_args: Any) -> None:
        selected_partition = self.partition_combo.currentData()
        choices = self._choices_for_algorithm()
        partitions = list(dict.fromkeys(getattr(choice.source, "partition", "") for choice in choices))

        self.partition_combo.blockSignals(True)
        self.partition_combo.clear()
        for partition in partitions:
            label = PARTITION_LABELS.get(str(partition), str(partition))
            self.partition_combo.addItem(label, partition)
        index = self.partition_combo.findData(selected_partition)
        if index >= 0:
            self.partition_combo.setCurrentIndex(index)
        self.partition_combo.blockSignals(False)
        self._refresh_specs()

    def _refresh_specs(self, *_args: Any) -> None:
        choice = self._selected_choice()
        previous_key = self._spec_key(self.plot_combo.currentData())
        self.plot_combo.blockSignals(True)
        self.plot_combo.clear()
        if choice is None:
            self.plot_combo.blockSignals(False)
            self._show_status("暂无可用缓存结果。")
            return
        try:
            specs = available_plot_specs(choice.source)
        except Exception as exc:
            self.plot_combo.blockSignals(False)
            self._show_status(f"无法读取当前结果的图表：{exc}")
            return
        for spec in specs:
            title = str(spec.display_title)
            self.plot_combo.addItem(title, spec)
            self.plot_combo.setItemData(
                self.plot_combo.count() - 1, title, Qt.ItemDataRole.ToolTipRole
            )
        index = next(
            (
                position
                for position in range(self.plot_combo.count())
                if self._spec_key(self.plot_combo.itemData(position)) == previous_key
            ),
            -1,
        )
        if index >= 0:
            self.plot_combo.setCurrentIndex(index)
        self.plot_combo.blockSignals(False)
        if not specs:
            reason = getattr(choice.source, "score_unavailable_reason", None)
            self._show_status(reason or "当前缓存没有可用图表数据。")
            return
        self._render_current()

    @staticmethod
    def _spec_key(spec: Any) -> tuple[Any, Any, str] | None:
        if spec is None:
            return None
        return (
            getattr(spec, "kind", None),
            getattr(spec, "class_label", None),
            str(getattr(spec, "display_title", "")),
        )

    def _show_status(self, text: str) -> None:
        self._clear_chart()
        self.status_label.setText(text)
        self.status_label.show()

    def _clear_chart(self) -> None:
        while self.chart_layout.count():
            item = self.chart_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                figure = getattr(widget, "figure", None)
                if figure is not None:
                    figure.clear()
                widget.deleteLater()
        self._chart_title_label = None
        self._chart_note_label = None
        self._chart_canvas = None

    def _render_current(self, *_args: Any) -> None:
        choice = self._selected_choice()
        spec = self.plot_combo.currentData()
        if choice is None or spec is None:
            return
        self._clear_chart()
        try:
            payload = build_plot_payload(choice.source, spec)
            count_parts = [
                f"分区 {payload.partition_count} 行",
                f"有效 {payload.effective_count}",
                f"排除 {payload.excluded_count}",
                f"抽样 {payload.sampled_count}",
            ]
            if payload.missing_count:
                count_parts.append(f"缺失类别 {payload.missing_count}")
            if payload.other_count:
                count_parts.append(f"其他类别 {payload.other_count}")
            details_text = " · ".join(count_parts)
            if payload.note:
                details_text += f" · {payload.note}"
            figure = render_plot_payload(
                replace(payload, title=str(payload.spec.display_title), note=""),
                include_count_caption=False,
            )
        except Exception as exc:
            self._show_status(str(exc))
            return
        self.status_label.clear()
        self.status_label.hide()
        full_title = str(payload.title or payload.spec.display_title)
        title_label = QLabel(full_title, self.chart_host)
        title_label.setObjectName("resultChartTitle")
        title_label.setAccessibleName("图表完整标题")
        title_label.setTextFormat(Qt.TextFormat.PlainText)
        title_label.setWordWrap(True)
        title_label.setToolTip(full_title)
        self.chart_layout.addWidget(title_label)
        self._chart_title_label = title_label
        details_label = QLabel(details_text, self.chart_host)
        details_label.setObjectName("resultChartNote")
        details_label.setAccessibleName("图表说明和有效样本数量")
        details_label.setTextFormat(Qt.TextFormat.PlainText)
        details_label.setWordWrap(True)
        details_label.setToolTip(details_text)
        self.chart_layout.addWidget(details_label)
        self._chart_note_label = details_label
        canvas = FigureCanvasQTAgg(figure)
        canvas.setToolTip(f"{full_title}\n{details_text}")
        self.chart_layout.addWidget(canvas, 1)
        self._chart_canvas = canvas
        QTimer.singleShot(0, self._fit_canvas_to_viewport)

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt event name
        super().resizeEvent(event)
        QTimer.singleShot(0, self._fit_canvas_to_viewport)

    def _fit_canvas_to_viewport(self) -> None:
        """Use available height by default; leave scrolling for constrained layouts."""
        canvas = self._chart_canvas
        if canvas is None:
            return
        viewport = self.chart_scroll_area.viewport()
        viewport_height = max(1, viewport.height())
        viewport_width = max(1, viewport.width())
        labels = [
            label for label in (self._chart_title_label, self._chart_note_label)
            if label is not None and not label.isHidden()
        ]
        fixed_height = sum(
            max(0, label.heightForWidth(viewport_width))
            if label.hasHeightForWidth()
            else label.sizeHint().height()
            for label in labels
        )
        layout = self.chart_layout
        fixed_height += layout.contentsMargins().top() + layout.contentsMargins().bottom()
        fixed_height += max(0, len(labels)) * max(0, layout.spacing())
        desired_height = viewport_height - fixed_height
        minimum_height = 180
        maximum_height = 460
        target_height = min(maximum_height, max(minimum_height, desired_height))
        if canvas.minimumHeight() != target_height or canvas.maximumHeight() != target_height:
            # A fixed widget height is needed as well as a smaller Figure:
            # Qt otherwise keeps the original FigureCanvas size hint in the
            # scroll area's content size and clips the bottom at small windows.
            canvas.setFixedHeight(target_height)
            canvas.updateGeometry()


def add_or_select_result_chart_tab(
    tabs: QTabWidget,
    chart_page: QWidget,
    *,
    has_cached_results: bool,
) -> int:
    """Add a visible 图表 tab and select it when results become available."""
    index = tabs.indexOf(chart_page)
    if index < 0:
        index = tabs.addTab(chart_page, "图表")
    if has_cached_results:
        tabs.setCurrentIndex(index)
    return index
