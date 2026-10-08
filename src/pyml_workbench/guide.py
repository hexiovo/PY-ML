"""Offline guide viewer; the Markdown document is a packaged resource."""
from __future__ import annotations

from importlib.resources import files
from pathlib import Path
import re

# Preserve the GUI's dateutil-before-Qt import contract when this module is
# imported directly (Qt's Shiboken hook cannot inspect six's lazy module).
import dateutil.rrule  # noqa: F401

from PySide6.QtCore import Qt, QTimer, QUrl
from PySide6.QtGui import (
    QColor, QFont, QImage, QPixmap, QTextCursor, QTextDocument, QTextFrameFormat, QTextLength,
    QTextOption, QTextTable,
)
from PySide6.QtWidgets import (
    QDialog, QFrame, QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem,
    QPushButton, QSplitter, QTextBrowser, QVBoxLayout,
)


def _guide_directory() -> Path:
    """Resolve the document and its relative images from the same location."""
    resource = files("pyml_workbench").joinpath("resources", "overall-guide.md")
    if resource.is_file():
        return Path(str(resource)).parent
    package = Path(__file__).resolve().parent
    if package.parent.name == "src":
        return package.parent.parent / "docs"
    raise FileNotFoundError("缺少随包发布的 resources/overall-guide.md")


def load_overall_guide() -> str:
    """Read the installed/frozen resource or the single source in a checkout."""
    return (_guide_directory() / "overall-guide.md").read_text(encoding="utf-8")


class _GuideBrowser(QTextBrowser):
    """Keep screenshot proportions and fit images to the reading pane."""

    def fit_images(self) -> None:
        document = self.document()
        available_width = max(120, self.viewport().width() - 2 * document.documentMargin() - 12)
        changed = False
        block = document.begin()
        while block.isValid():
            iterator = block.begin()
            while not iterator.atEnd():
                fragment = iterator.fragment()
                if fragment.isValid() and fragment.charFormat().isImageFormat():
                    image_format = fragment.charFormat().toImageFormat()
                    url = document.baseUrl().resolved(QUrl(image_format.name()))
                    picture = document.resource(QTextDocument.ResourceType.ImageResource, url)
                    if isinstance(picture, (QImage, QPixmap)) and not picture.isNull():
                        width = min(available_width, picture.width())
                        height = width * picture.height() / picture.width()
                        if abs(image_format.width() - width) > 1 or abs(image_format.height() - height) > 1:
                            image_format.setWidth(width)
                            image_format.setHeight(height)
                            cursor = QTextCursor(document)
                            cursor.setPosition(fragment.position())
                            cursor.setPosition(fragment.position() + fragment.length(), QTextCursor.MoveMode.KeepAnchor)
                            cursor.setCharFormat(image_format)
                            changed = True
                iterator += 1
            block = block.next()
        if changed:
            # Qt may otherwise keep zero-height text layouts after resizing an
            # image in a document containing tables. Reflow the complete guide.
            document.markContentsDirty(0, document.characterCount())

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self.fit_images()


class OverallGuideDialog(QDialog):
    """Non-modal, reusable help with chapter navigation and wrapping search."""

    def __init__(self, markdown: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("操作指南 · PY-ML 工作台")
        self.setObjectName("overallGuideDialog")
        self.setModal(False)
        screen = self.screen()
        available = screen.availableGeometry() if screen is not None else None
        width = min(1160, max(1, available.width() - 32)) if available is not None else 1160
        height = min(820, max(1, available.height() - 64)) if available is not None else 820
        self.resize(width, height)
        self.setMinimumSize(min(760, width), min(520, height))
        self._search_query = ""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 16)
        layout.setSpacing(14)
        self.setStyleSheet(self._style_sheet())
        banner = QFrame()
        banner.setObjectName("guideBanner")
        banner_layout = QHBoxLayout(banner)
        banner_layout.setContentsMargins(22, 18, 22, 18)
        banner_text = QVBoxLayout()
        title = QLabel("操作指南")
        title.setObjectName("guideTitle")
        description = QLabel("从导入数据到导出结果，一步一步完成你的实验。")
        description.setObjectName("guideDescription")
        description.setWordWrap(True)
        banner_text.addWidget(title)
        banner_text.addWidget(description)
        banner_layout.addLayout(banner_text, 1)
        badge = QLabel("离线阅读  /  F1 随时打开")
        badge.setObjectName("guideBadge")
        banner_layout.addWidget(badge)
        layout.addWidget(banner)
        tools = QHBoxLayout()
        tools.setSpacing(8)
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("输入关键词，例如：导入数据、自动调参、HMM")
        self.search_edit.setAccessibleName("搜索操作指南")
        self.previous_button = QPushButton("上一个")
        self.next_button = QPushButton("下一个")
        self.search_edit.setClearButtonEnabled(True)
        self.previous_button.setAutoDefault(False)
        self.next_button.setAutoDefault(False)
        tools.addWidget(self.search_edit, 1)
        tools.addWidget(self.previous_button)
        tools.addWidget(self.next_button)
        layout.addLayout(tools)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.setHandleWidth(12)
        sidebar = QFrame()
        sidebar.setObjectName("guideSidebar")
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(12, 16, 12, 12)
        contents_label = QLabel("章节目录")
        contents_label.setObjectName("guideContentsTitle")
        sidebar_layout.addWidget(contents_label)
        self.chapters = QListWidget()
        self.chapters.setObjectName("guideChapters")
        self.chapters.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.chapters.setTextElideMode(Qt.TextElideMode.ElideRight)
        self.chapters.setAccessibleName("操作指南章节目录")
        sidebar.setMinimumWidth(200)
        sidebar_layout.addWidget(self.chapters, 1)
        self.browser = _GuideBrowser()
        self.browser.setObjectName("overallGuideContent")
        self.browser.setAccessibleName("操作指南正文")
        self.browser.setOpenLinks(False)
        self.browser.setOpenExternalLinks(False)
        self.browser.document().setBaseUrl(QUrl.fromLocalFile(str(_guide_directory()) + "/"))
        self.browser.document().setDocumentMargin(24)
        self.browser.document().setDefaultFont(QFont("Microsoft YaHei UI", 11))
        self.browser.document().setDefaultStyleSheet(
            "h1 { color: #16365c; font-size: 24px; margin-bottom: 16px; } "
            "h2 { color: #1d4ed8; font-size: 20px; margin-top: 28px; margin-bottom: 12px; } "
            "p, li { line-height: 150%; } "
            "td, th { padding: 8px; border: 1px solid #dbe3ee; } "
            "th { background-color: #eff6ff; color: #16365c; } "
            "pre { background-color: #f1f5f9; color: #334155; } "
            "a { color: #1d4ed8; }"
        )
        self.browser.setMarkdown(markdown)
        text_option = self.browser.document().defaultTextOption()
        text_option.setWrapMode(QTextOption.WrapMode.WrapAtWordBoundaryOrAnywhere)
        self.browser.document().setDefaultTextOption(text_option)
        # Markdown tables and code blocks must also fit a narrow reading pane.
        def fit_tables(frame):
            for child in frame.childFrames():
                if isinstance(child, QTextTable):
                    table_format = child.format()
                    table_format.setWidth(QTextLength(QTextLength.Type.PercentageLength, 100))
                    table_format.setColumnWidthConstraints([
                        QTextLength(QTextLength.Type.PercentageLength, 100 / child.columns())
                        for _ in range(child.columns())
                    ])
                    table_format.setCellPadding(6)
                    child.setFormat(table_format)
                fit_tables(child)
        fit_tables(self.browser.document().rootFrame())
        # QTextDocument's Markdown importer ignores CSS for fenced code. Put
        # each contiguous code block in a real frame so multi-line commands
        # share one padded panel instead of separate highlighted text rows.
        code_ranges = []
        block = self.browser.document().begin()
        code_start = None
        code_end = None
        while block.isValid():
            if block.blockFormat().nonBreakableLines():
                if code_start is None:
                    code_start = block.position()
                code_end = block.position() + block.length() - 1
            elif code_start is not None:
                code_ranges.append((code_start, code_end))
                code_start = None
            block = block.next()
        if code_start is not None:
            code_ranges.append((code_start, code_end))
        for start, end in reversed(code_ranges):
            cursor = QTextCursor(self.browser.document())
            cursor.setPosition(start)
            cursor.setPosition(end, QTextCursor.MoveMode.KeepAnchor)
            code_format = cursor.charFormat()
            code_format.setFontFamilies(["Cascadia Code", "Consolas"])
            code_format.setFontPointSize(10)
            code_format.setForeground(QColor("#0F172A"))
            cursor.mergeCharFormat(code_format)
            panel = QTextFrameFormat()
            panel.setBackground(QColor("#F1F5F9"))
            panel.setBorder(1)
            panel.setBorderBrush(QColor("#DBE3EE"))
            panel.setPadding(12)
            panel.setTopMargin(8)
            panel.setBottomMargin(8)
            panel.setWidth(QTextLength(QTextLength.Type.PercentageLength, 100))
            cursor.insertFrame(panel)
        # QTextDocument positions work with Chinese headings and tables without
        # relying on renderer-specific Markdown anchor generation.
        block = self.browser.document().begin()
        heading_titles = iter(re.findall(r"^##\s+(.+)$", markdown, re.MULTILINE))
        while block.isValid():
            if block.blockFormat().headingLevel() == 2:
                heading = next(heading_titles, block.text())
                item = QListWidgetItem(heading, self.chapters)
                item.setData(Qt.ItemDataRole.UserRole, block.position())
                item.setToolTip(heading)
            if block.blockFormat().nonBreakableLines():
                block_format = block.blockFormat()
                block_format.setNonBreakableLines(False)
                block_format.setTopMargin(0)
                block_format.setBottomMargin(0)
                QTextCursor(block).setBlockFormat(block_format)
            block = block.next()
        sidebar_note = QLabel(f"{self.chapters.count()} 个章节 · 按操作顺序阅读")
        sidebar_note.setObjectName("guideMutedText")
        sidebar_layout.addWidget(sidebar_note)
        splitter.addWidget(sidebar)
        splitter.addWidget(self.browser)
        splitter.setSizes([270, 850])
        splitter.setStretchFactor(1, 1)
        layout.addWidget(splitter, 1)
        footer = QHBoxLayout()
        self.search_status = QLabel("左侧选择章节 · Enter 查找下一处 · 图片随阅读区域自动缩放")
        self.search_status.setObjectName("guideMutedText")
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
        self.browser.fit_images()

    @staticmethod
    def _style_sheet() -> str:
        return """
        QWidget { font-family: "Microsoft YaHei UI"; font-size: 10pt; color: #1E293B; background: #F3F6FB; }
        QFrame#guideBanner { background: #E8F0FE; border: 1px solid #CCDCF5; border-radius: 14px; }
        QFrame#guideBanner QLabel { background: transparent; }
        QLabel#guideTitle { color: #16365C; font-size: 23pt; font-weight: 700; }
        QLabel#guideDescription { color: #475569; font-size: 11pt; }
        QLabel#guideBadge { background: #FFFFFF; color: #1D4ED8; padding: 8px 12px; border-radius: 8px; }
        QFrame#guideSidebar { background: #FFFFFF; border: 1px solid #DBE3EE; border-radius: 12px; }
        QLabel#guideContentsTitle { background: transparent; font-size: 11pt; font-weight: 650; padding-left: 8px; }
        QLabel#guideMutedText { background: transparent; color: #64748B; font-size: 9pt; }
        QListWidget#guideChapters { background: #FFFFFF; border: none; outline: 0; padding: 0; }
        QListWidget#guideChapters::item { padding: 11px 8px; margin: 2px 0; border-radius: 7px; }
        QListWidget#guideChapters::item:hover { background: #F1F5F9; }
        QListWidget#guideChapters::item:selected { background: #E8F0FE; color: #1D4ED8; font-weight: 600; }
        QListWidget#guideChapters:focus { border: 2px solid #2563EB; }
        QTextBrowser#overallGuideContent { background: #FFFFFF; border: 1px solid #DBE3EE; border-radius: 12px; selection-background-color: #BFDBFE; selection-color: #0F172A; }
        QLineEdit { background: #FFFFFF; border: 1px solid #CBD5E1; border-radius: 8px; padding: 10px 12px; }
        QLineEdit:focus { border: 2px solid #2563EB; }
        QPushButton { background: #FFFFFF; border: 1px solid #CBD5E1; border-radius: 8px; padding: 9px 16px; font-weight: 550; }
        QPushButton:hover { background: #E8F0FE; border-color: #2563EB; }
        QPushButton:pressed { background: #DBEAFE; }
        QPushButton:focus { border: 2px solid #2563EB; }
        QSplitter::handle { background: #F3F6FB; }
        """

    def _go_to_chapter(self, current, _previous) -> None:
        if current is None:
            return
        cursor = QTextCursor(self.browser.document())
        cursor.setPosition(current.data(Qt.ItemDataRole.UserRole))
        self.browser.setTextCursor(cursor)
        def align_heading():
            if self.chapters.currentRow() == self.chapters.row(current):
                scrollbar = self.browser.verticalScrollBar()
                scrollbar.setValue(scrollbar.value() + self.browser.cursorRect().top() - 18)
        # Let the image sizes and text reflow settle before aligning the chapter.
        QTimer.singleShot(0, self, align_heading)

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
