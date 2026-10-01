from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pyml_workbench.plot_dialog import PlotDialog
from pyml_workbench.plotting import (
    PlotColumn,
    PlotKind,
    PlotSource,
    PlotSpec,
    build_plot_payload,
)

from PySide6.QtCore import Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QMainWindow, QPushButton


def _payloads():
    source = PlotSource(
        schema_version=1,
        source_kind="loaded_eda",
        source_id="eda-source",
        owner_kind="loaded_dataset",
        owner_id="loaded-data",
        partition="all",
        row_positions=(0, 1, 2, 3),
        columns=(
            PlotColumn("measurement", "measurement", "float64", (1.0, 2.0, 3.0, 4.0)),
            PlotColumn("kind", "kind", "object", ("a", "b", "a", "c")),
        ),
        partition_count=4,
    )
    return (
        build_plot_payload(source, PlotSpec(PlotKind.HISTOGRAM, ("measurement",))),
        build_plot_payload(source, PlotSpec(PlotKind.CATEGORY_COUNTS, ("kind",))),
    )


class PlotDialogTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.application = QApplication.instance() or QApplication([])

    def test_chart_selection_is_nonmodal_and_png_svg_buttons_export_selected_figure(self):
        from matplotlib.figure import Figure

        dialog = PlotDialog(_payloads())
        dialog.show()
        self.application.processEvents()
        self.assertFalse(dialog.isModal())
        self.assertEqual(dialog.chart_selector.count(), 2)
        first_figure = dialog.current_figure
        self.assertIsInstance(first_figure, Figure)

        dialog.chart_selector.setCurrentIndex(1)
        self.application.processEvents()
        selected_figure = dialog.current_figure
        self.assertIsInstance(selected_figure, Figure)
        self.assertIsNot(selected_figure, first_figure)

        with tempfile.TemporaryDirectory() as directory:
            png_path = Path(directory) / "selected.png"
            svg_path = Path(directory) / "selected.svg"
            with patch(
                "pyml_workbench.plot_dialog.QFileDialog.getSaveFileName",
                return_value=(str(png_path), "PNG 图像 (*.png)"),
            ), patch.object(first_figure, "savefig", wraps=first_figure.savefig) as first_save, patch.object(
                selected_figure, "savefig", wraps=selected_figure.savefig
            ) as selected_save:
                QTest.mouseClick(dialog.save_png_button, Qt.MouseButton.LeftButton)
                self.application.processEvents()
                first_save.assert_not_called()
                selected_save.assert_called_once()
            self.assertTrue(png_path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n"))
            self.assertEqual(dialog.last_saved_path, png_path)

            with patch(
                "pyml_workbench.plot_dialog.QFileDialog.getSaveFileName",
                return_value=(str(svg_path), "SVG 图像 (*.svg)"),
            ):
                QTest.mouseClick(dialog.save_svg_button, Qt.MouseButton.LeftButton)
                self.application.processEvents()
            self.assertIn(b"<svg", svg_path.read_bytes()[:1000])
            self.assertEqual(dialog.last_saved_path, svg_path)

        self.assertTrue(dialog.close())
        self.application.processEvents()

    def test_empty_payload_list_is_rejected_before_opening_a_window(self):
        with self.assertRaises(ValueError):
            PlotDialog(())

    def test_main_window_event_loop_remains_responsive_while_plot_is_open(self):
        main = QMainWindow()
        button = QPushButton("main action", main)
        main.setCentralWidget(button)
        events = []
        button.clicked.connect(lambda: events.append("clicked"))
        main.show()
        dialog = PlotDialog(_payloads(), parent=main)
        dialog.show()
        self.application.processEvents()
        QTest.mouseClick(button, Qt.MouseButton.LeftButton)
        self.application.processEvents()
        self.assertFalse(dialog.isModal())
        self.assertTrue(main.isVisible())
        self.assertEqual(events, ["clicked"])
        dialog.close()
        main.close()
        self.application.processEvents()


if __name__ == "__main__":
    unittest.main()
