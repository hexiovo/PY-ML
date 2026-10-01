"""Non-modal Qt window for selecting, rendering, and saving plot payloads."""
from __future__ import annotations

from pathlib import Path
import re
from typing import Sequence

# Prime dateutil before PySide installs its import hook.  Matplotlib's Qt
# backend also imports it, but this lightweight module does not import Matplotlib.
import dateutil.rrule  # noqa: F401

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from .plotting import PlotPayload, render_plot_payload


class PlotDialog(QDialog):
    """Display one or more payloads without entering a nested modal event loop."""

    def __init__(self, payloads: Sequence[PlotPayload], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._payloads = tuple(payloads)
        if not self._payloads:
            raise ValueError("PlotDialog requires at least one PlotPayload")
        if any(not isinstance(payload, PlotPayload) for payload in self._payloads):
            raise TypeError("payloads must contain PlotPayload values")

        self.setWindowTitle("基础图表")
        self.setModal(False)
        self.setWindowModality(Qt.WindowModality.NonModal)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self.resize(820, 600)
        self._selected_index = 0
        self._figures: list[object | None] = [None] * len(self._payloads)
        self._last_saved_path: Path | None = None

        outer_layout = QVBoxLayout(self)
        controls = QHBoxLayout()
        controls.addWidget(QLabel("图表"))
        self.chart_selector = QComboBox(self)
        self.chart_selector.setObjectName("plotChartSelector")
        for payload in self._payloads:
            self.chart_selector.addItem(payload.title or payload.spec.display_title)
        self.chart_selector.setVisible(len(self._payloads) > 1)
        controls.addWidget(self.chart_selector, 1)
        outer_layout.addLayout(controls)

        # Import the Qt canvas only when a user opens a plot window.  This is
        # the first point where Matplotlib and its GUI backend are needed.
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.figure import Figure

        self._figure_type = Figure
        self._canvas = FigureCanvasQTAgg(Figure(figsize=(7.2, 4.6)))
        self._figures[0] = self._canvas.figure
        outer_layout.addWidget(self._canvas, 1)

        self.effective_count_label = QLabel(self)
        self.effective_count_label.setObjectName("plotCountsLabel")
        outer_layout.addWidget(self.effective_count_label)

        actions = QHBoxLayout()
        actions.addStretch(1)
        self.save_png_button = QPushButton("保存 PNG…", self)
        self.save_png_button.setObjectName("savePlotPngButton")
        self.save_svg_button = QPushButton("保存 SVG…", self)
        self.save_svg_button.setObjectName("savePlotSvgButton")
        self.close_button = QPushButton("关闭", self)
        self.close_button.setObjectName("closePlotButton")
        actions.addWidget(self.save_png_button)
        actions.addWidget(self.save_svg_button)
        actions.addWidget(self.close_button)
        outer_layout.addLayout(actions)

        self.chart_selector.currentIndexChanged.connect(self._select_chart)
        self.save_png_button.clicked.connect(lambda: self.save_selected("png"))
        self.save_svg_button.clicked.connect(lambda: self.save_selected("svg"))
        self.close_button.clicked.connect(self.close)
        self._render_selected()

    @property
    def payloads(self) -> tuple[PlotPayload, ...]:
        return self._payloads

    @property
    def selected_index(self) -> int:
        return self._selected_index

    @property
    def current_figure(self):
        """Return the Figure currently displayed by the chart canvas."""
        return self._figures[self._selected_index]

    @property
    def last_saved_path(self) -> Path | None:
        return self._last_saved_path

    def _select_chart(self, index: int) -> None:
        if not 0 <= index < len(self._payloads):
            return
        self._selected_index = index
        figure = self._figures[index]
        if figure is None:
            figure = self._figure_type(figsize=(7.2, 4.6))
            self._figures[index] = figure
        self._canvas.figure = figure
        figure.set_canvas(self._canvas)
        self._render_selected()

    def _render_selected(self) -> None:
        payload = self._payloads[self._selected_index]
        figure = self._figures[self._selected_index]
        if figure is None:
            return
        render_plot_payload(payload, figure=figure)
        self.effective_count_label.setText(
            f"{payload.partition}：总计 {payload.partition_count}，有效 {payload.effective_count}，"
            f"排除 {payload.excluded_count}，抽样 {payload.sampled_count}"
        )
        self._canvas.draw_idle()

    def save_selected(self, format_name: str) -> bool:
        """Open a save dialog and export exactly the Figure visible in this window."""
        if format_name not in {"png", "svg"}:
            raise ValueError("format_name must be 'png' or 'svg'")
        figure = self.current_figure
        if figure is None:
            return False
        payload = self._payloads[self._selected_index]
        extension = ".png" if format_name == "png" else ".svg"
        default_name = _safe_filename(payload.title or payload.spec.display_title)
        path_text, _selected_filter = QFileDialog.getSaveFileName(
            self,
            f"保存 {format_name.upper()} 图表",
            f"{default_name}{extension}",
            "PNG 图像 (*.png)" if format_name == "png" else "SVG 图像 (*.svg)",
        )
        if not path_text:
            return False
        target = Path(path_text).expanduser()
        if target.suffix.lower() != extension:
            target = target.with_suffix(extension)
        try:
            # Figure.savefig is deliberately invoked on the selected Figure;
            # pyplot's current-figure state is never consulted.
            figure.savefig(target, format=format_name, bbox_inches="tight")
        except (OSError, ValueError) as exc:
            QMessageBox.critical(self, "保存图表失败", str(exc))
            return False
        self._last_saved_path = target
        return True


def _safe_filename(value: str) -> str:
    cleaned = re.sub(r"[<>:\\|?*\"/\x00-\x1f]", "_", value).strip(" ._")
    return (cleaned[:80] or "plot")


__all__ = ["PlotDialog"]
