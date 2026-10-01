"""Offline guide viewer; the Markdown document is a packaged resource."""
from __future__ import annotations

from importlib.resources import files
from pathlib import Path

# Preserve the GUI's dateutil-before-Qt import contract when this module is
# imported directly (Qt's Shiboken hook cannot inspect six's lazy module).
import dateutil.rrule  # noqa: F401

from PySide6.QtCore import Qt
from PySide6.QtGui import QTextCursor, QTextDocument
from PySide6.QtWidgets import (
    QDialog, QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem,
    QPushButton, QSplitter, QTextBrowser, QVBoxLayout,
)


def load_overall_guide() -> str:
    """Read the installed/frozen resource or the single source in a checkout."""
    resource = files("pyml_workbench").joinpath("resources", "overall-guide.md")
    if resource.is_file():
        return resource.read_text(encoding="utf-8")
    package = Path(__file__).resolve().parent
    if package.parent.name == "src":
        return (package.parent.parent / "docs" / "overall-guide.md").read_text(
            encoding="utf-8"
        )
    raise FileNotFoundError("缺少随包发布的 resources/overall-guide.md")


class OverallGuideDialog(QDialog):
    """Non-modal, reusable help with chapter navigation and wrapping search."""

    def __init__(self, markdown: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("整体指南 · PY-ML 工作台")
        self.setObjectName("overallGuideDialog")
        self.setModal(False)
        self.resize(1080, 780)
        self.setMinimumSize(760, 520)
        self._search_query = ""
        layout = QVBoxLayout(self)
        tools = QHBoxLayout()
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("搜索指南，例如：自动调参、错误编号、HMM")
        self.search_edit.setAccessibleName("搜索整体指南")
        self.previous_button = QPushButton("上一个")
        self.next_button = QPushButton("下一个")
        self.previous_button.setAutoDefault(False)
        self.next_button.setAutoDefault(False)
        tools.addWidget(self.search_edit, 1)
        tools.addWidget(self.previous_button)
        tools.addWidget(self.next_button)
        layout.addLayout(tools)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        self.chapters = QListWidget()
        self.chapters.setAccessibleName("整体指南章节目录")
        self.chapters.setMinimumWidth(170)
        self.browser = QTextBrowser()
        self.browser.setObjectName("overallGuideContent")
        self.browser.setAccessibleName("整体指南正文")
        self.browser.setOpenLinks(False)
        self.browser.setOpenExternalLinks(False)
        self.browser.document().setDefaultStyleSheet(
            "h1, h2 { color: #173e69; } "
            "td, th { padding: 5px; } pre { background-color: #f2f5fa; }"
        )
        self.browser.setMarkdown(markdown)
        # QTextDocument positions work with Chinese headings and tables without
        # relying on renderer-specific Markdown anchor generation.
        block = self.browser.document().begin()
        while block.isValid():
            if block.blockFormat().headingLevel() == 2:
                item = QListWidgetItem(block.text(), self.chapters)
                item.setData(Qt.ItemDataRole.UserRole, block.position())
                item.setToolTip(block.text())
            block = block.next()
        splitter.addWidget(self.chapters)
        splitter.addWidget(self.browser)
        splitter.setSizes([245, 790])
        splitter.setStretchFactor(1, 1)
        layout.addWidget(splitter, 1)
        footer = QHBoxLayout()
        self.search_status = QLabel("离线指南 · 左侧选择章节，上方输入关键词查找")
        self.search_status.setWordWrap(True)
        close_button = QPushButton("关闭")
        close_button.setAutoDefault(False)
        close_button.clicked.connect(self.close)
        footer.addWidget(self.search_status, 1)
        footer.addWidget(close_button)
        layout.addLayout(footer)
        self.chapters.currentItemChanged.connect(self._go_to_chapter)
        self.next_button.clicked.connect(lambda: self.find_text())
        self.previous_button.clicked.connect(lambda: self.find_text(backward=True))
        self.search_edit.returnPressed.connect(lambda: self.find_text())

    def _go_to_chapter(self, current, _previous) -> None:
        if current is None:
            return
        cursor = QTextCursor(self.browser.document())
        cursor.setPosition(current.data(Qt.ItemDataRole.UserRole))
        self.browser.setTextCursor(cursor)
        self.browser.ensureCursorVisible()

    def find_text(self, *, backward: bool = False) -> bool:
        query = self.search_edit.text().strip()
        if not query:
            self.search_status.setText("请输入要查找的关键词。")
            self._search_query = ""
            return False
        flags = QTextDocument.FindFlag.FindBackward if backward else QTextDocument.FindFlag(0)
        cursor = self.browser.textCursor()
        if query != self._search_query:
            cursor.movePosition(
                QTextCursor.MoveOperation.End if backward else QTextCursor.MoveOperation.Start
            )
            self.browser.setTextCursor(cursor)
        self._search_query = query
        found = self.browser.find(query, flags)
        if not found:
            cursor.movePosition(
                QTextCursor.MoveOperation.End if backward else QTextCursor.MoveOperation.Start
            )
            self.browser.setTextCursor(cursor)
            found = self.browser.find(query, flags)
        self.search_status.setText(f"已找到：{query}" if found else f"未找到：{query}")
        return found
