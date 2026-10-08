"""Build the full Chinese PDF guide from Markdown and actual Qt captures.

Run with the bundled document Python (ReportLab, Pillow, pypdf).
Application dependencies are used only by capture_pdf_guide.py.
"""
from __future__ import annotations

import argparse
import hashlib
from html import escape
import json
from pathlib import Path
import re
from datetime import date

from PIL import Image as PILImage
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate, Frame, Image, KeepTogether, LongTable, NextPageTemplate,
    PageBreak, PageTemplate, Paragraph, Spacer, Table, TableStyle,
)
from reportlab.platypus.tableofcontents import TableOfContents
from pypdf import PdfReader


INK = colors.HexColor('#172B4D')
BLUE = colors.HexColor('#2457A6')
MUTED = colors.HexColor('#52647C')
PALE = colors.HexColor('#F2F6FC')
RULE = colors.HexColor('#CCD8E8')
P_WIDTH, P_HEIGHT = A4
L_WIDTH, L_HEIGHT = landscape(A4)
BODY_WIDTH = P_WIDTH - 88


def register_fonts():
    for name, file in (('GuideRegular', 'msyh.ttc'), ('GuideBold', 'msyhbd.ttc')):
        pdfmetrics.registerFont(TTFont(name, str(Path('C:/Windows/Fonts') / file), subfontIndex=0))
    pdfmetrics.registerFontFamily('GuideRegular', normal='GuideRegular', bold='GuideBold',
                                 italic='GuideRegular', boldItalic='GuideBold')


def styles():
    body = ParagraphStyle('Body', fontName='GuideRegular', fontSize=11,
                          leading=18, textColor=INK, wordWrap='CJK', spaceAfter=9,
                          allowWidows=0, allowOrphans=0, splitLongWords=True)
    return {
        'body': body,
        'title': ParagraphStyle('CoverTitle', parent=body, fontName='GuideBold', fontSize=29,
                                leading=41, textColor=BLUE, spaceAfter=18),
        'subtitle': ParagraphStyle('CoverSubtitle', parent=body, fontSize=16, leading=25),
        'heading': ParagraphStyle('ChapterHeading', parent=body, fontName='GuideBold', fontSize=17,
                                 leading=27, textColor=BLUE, spaceAfter=14, keepWithNext=True),
        'smallheading': ParagraphStyle('SmallHeading', parent=body, fontName='GuideBold', fontSize=12,
                                      leading=19, spaceBefore=10, keepWithNext=True),
        'figureheading': ParagraphStyle('FigureHeading', parent=body, fontName='GuideBold',
                                        fontSize=12, leading=19, alignment=TA_CENTER,
                                        spaceBefore=0, spaceAfter=7),
        'caption': ParagraphStyle('FigureCaption', parent=body, fontName='GuideBold', fontSize=11,
                                 leading=17, alignment=TA_CENTER, spaceBefore=0, spaceAfter=5),
        'figurenote': ParagraphStyle('FigureNote', parent=body, fontSize=9.5, leading=14.5,
                                     textColor=MUTED, alignment=TA_CENTER, spaceAfter=10),
        'tablecaption': ParagraphStyle('TableCaption', parent=body, fontName='GuideBold',
                                       fontSize=11, leading=17, alignment=TA_CENTER,
                                       spaceBefore=10, spaceAfter=6, keepWithNext=True),
        'note': ParagraphStyle('Note', parent=body, fontSize=9.5, leading=14.5,
                               textColor=MUTED, spaceAfter=10),
        'table': ParagraphStyle('TableCell', parent=body, fontSize=9.5, leading=14.8, spaceAfter=0),
        'tableheader': ParagraphStyle('TableHeader', parent=body, fontName='GuideBold', fontSize=9.8,
                                     leading=15.2, textColor=colors.white, spaceAfter=0),
        'code': ParagraphStyle('Code', parent=body, fontSize=9.5, leading=15,
                               backColor=PALE, borderColor=RULE, borderWidth=0.5,
                               borderPadding=9, spaceBefore=2, spaceAfter=12),
        'bullet': ParagraphStyle('Bullet', parent=body, leftIndent=13, firstLineIndent=-10, spaceAfter=6),
        'toc': ParagraphStyle('ContentsEntry', parent=body, fontSize=10.5, leading=18,
                              leftIndent=0, firstLineIndent=0, spaceBefore=5, spaceAfter=5),
    }


def inline(text):
    # Protect inline code before applying emphasis; source never contains HTML.
    protected = []

    def code(match):
        protected.append('<font color="#2457A6">' + escape(match.group(1)) + '</font>')
        return f'@@CODE{len(protected) - 1}@@'

    text = re.sub(r'`([^`]+)`', code, text)
    text = escape(text)
    text = re.sub(r'\*\*(.+?)\*\*', r'<b>\1</b>', text)
    for index, value in enumerate(protected):
        text = text.replace(f'@@CODE{index}@@', value)
    return text


class GuideDocument(BaseDocTemplate):
    def __init__(self, filename, *, version, revision_date, guide_styles):
        super().__init__(str(filename), pagesize=A4, title='PY-ML 工作台整体图文指南',
                         author='PY-ML Workbench',
                         subject=f'中文版 {version}：13 章操作指南与实际界面截图')
        self.version = version
        self.revision_date = revision_date
        self.guide_styles = guide_styles
        self.active_title = '整体图文指南'
        portrait_frame = Frame(44, 45, BODY_WIDTH, P_HEIGHT - 112,
                               leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
        landscape_frame = Frame(38, 34, L_WIDTH - 76, L_HEIGHT - 69,
                                leftPadding=0, rightPadding=0, topPadding=0, bottomPadding=0)
        self.addPageTemplates([
            PageTemplate('portrait', frames=[portrait_frame], pagesize=A4, onPage=self.draw_page),
            PageTemplate('landscape', frames=[landscape_frame], pagesize=landscape(A4), onPage=self.draw_page),
        ])

    def draw_page(self, canvas, document):
        width, height = canvas._pagesize
        canvas.saveState()
        canvas.setFont('GuideRegular', 8)
        canvas.setFillColor(MUTED)
        canvas.drawString(38 if width > height else 44, height - 25, 'PY-ML 工作台  |  整体图文指南')
        canvas.setStrokeColor(RULE)
        canvas.setLineWidth(0.5)
        canvas.line(38 if width > height else 44, 28, width - (38 if width > height else 44), 28)
        canvas.drawString(38 if width > height else 44, 16,
                          f'{self.version} · 中文图文指南 · {self.revision_date}')
        canvas.drawRightString(width - (38 if width > height else 44), 16, f'第 {document.page} 页')
        canvas.restoreState()

    def afterFlowable(self, flowable):
        if isinstance(flowable, Paragraph) and hasattr(flowable, 'chapter_key'):
            self.active_title = flowable.getPlainText()
            key = flowable.chapter_key
            self.canv.bookmarkPage(key)
            self.canv.addOutlineEntry(self.active_title, key, level=0, closed=False)
            self.notify('TOCEntry', (0, self.active_title, self.page, key))


def parse_markdown(text, guide_styles):
    """Preserve paragraphs, numbered steps, typed tables, and code blocks."""
    lines = text.splitlines()
    result = []
    index = 0
    while index < len(lines):
        line = lines[index].strip()
        if not line:
            index += 1
            continue
        table_caption = re.fullmatch(r'<p align="center"><strong>(.+)</strong></p>', line)
        if table_caption:
            result.append(Paragraph(escape(table_caption.group(1)), guide_styles['tablecaption']))
            index += 1
            continue
        if line.startswith('```'):
            code = []
            index += 1
            while index < len(lines) and not lines[index].strip().startswith('```'):
                code.append(lines[index])
                index += 1
            html = '<br/>'.join(escape(value).replace(' ', '&#160;') if value else '&#160;' for value in code)
            result.append(Paragraph(html, guide_styles['code']))
            index += 1
            continue
        if line.startswith('|'):
            rows = []
            while index < len(lines) and lines[index].strip().startswith('|'):
                cells = [value.strip() for value in lines[index].strip().strip('|').split('|')]
                if not all(re.fullmatch(r':?-+:?', cell) for cell in cells):
                    rows.append(cells)
                index += 1
            columns = len(rows[0])
            if columns == 2:
                widths = [BODY_WIDTH * 0.31, BODY_WIDTH * 0.69]
            elif columns == 3:
                widths = [BODY_WIDTH * 0.22, BODY_WIDTH * 0.23, BODY_WIDTH * 0.55]
            else:
                widths = [BODY_WIDTH / columns] * columns
            data = [[Paragraph(inline(value), guide_styles['tableheader' if row_index == 0 else 'table'])
                     for value in row] for row_index, row in enumerate(rows)]
            table = LongTable(data, colWidths=widths, repeatRows=1, hAlign='CENTER')
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), BLUE),
                ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, PALE]),
                ('GRID', (0, 0), (-1, -1), 0.4, RULE),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('TOPPADDING', (0, 0), (-1, -1), 7),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 7),
                ('LEFTPADDING', (0, 0), (-1, -1), 8),
                ('RIGHTPADDING', (0, 0), (-1, -1), 8),
            ]))
            result.extend([table, Spacer(1, 12)])
            continue
        if line.startswith('### '):
            result.append(Paragraph(inline(line[4:]), guide_styles['smallheading']))
            index += 1
            continue
        if line.startswith('- '):
            result.append(Paragraph('• ' + inline(line[2:]), guide_styles['bullet']))
            index += 1
            continue
        if re.match(r'^\d+\. ', line):
            result.append(Paragraph(inline(line), guide_styles['bullet']))
            index += 1
            continue
        paragraph = [line]
        index += 1
        while index < len(lines) and lines[index].strip() and not re.match(r'^(?:```|\||### |\- |\d+\. )', lines[index].strip()):
            paragraph.append(lines[index].strip())
            index += 1
        result.append(Paragraph(inline(' '.join(paragraph)), guide_styles['body']))
    return result


def picture_flowables(item, assets, number, guide_styles, *, wide=False):
    with PILImage.open(assets / item['file']) as image:
        width, height = image.size
    maximum_width = L_WIDTH - 90 if wide else BODY_WIDTH
    maximum_height = 400 if wide else 540
    ratio = min(maximum_width / width, maximum_height / height, 1.0)
    image = Image(str(assets / item['file']), width=width * ratio, height=height * ratio)
    image.hAlign = 'CENTER'
    caption = Paragraph(f'图 {number} · {escape(item["title"])}', guide_styles['caption'])
    note = Paragraph(escape(item['note']), guide_styles['figurenote'])
    return [image, Spacer(1, 9), caption, note]


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, default=root / 'docs/overall-guide.md')
    parser.add_argument('--assets', type=Path, default=root / 'docs/assets/pdf-guide')
    parser.add_argument('--output', type=Path,
                        help='PDF output path (defaults to the screenshot manifest version)')
    args = parser.parse_args()
    report = json.loads((args.assets / 'screenshots.json').read_text(encoding='utf-8'))
    if args.output is None:
        args.output = root / f'output/pdf/PYML-Workbench-{report["version"]}-图文指南.pdf'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    names = [item['name'] for item in report['screenshots']]
    assert len(names) == len(set(names)), 'screenshot names must be unique'
    for item in report['screenshots']:
        assert hashlib.sha256((args.assets / item['file']).read_bytes()).hexdigest() == item['sha256']
    register_fonts()
    guide_styles = styles()
    text = args.source.read_text(encoding='utf-8')
    sections = re.split(r'^## ', text, flags=re.MULTILINE)
    assert len(sections) == 14, 'expected 13 chapters'
    revision_date = str(report.get('revision_date') or date.today().isoformat())
    doc = GuideDocument(args.output, version=report['version'], revision_date=revision_date,
                        guide_styles=guide_styles)
    story = [Spacer(1, 46),
             Paragraph('PY-ML 工作台<br/>整体图文指南', guide_styles['title']),
             Paragraph('从数据导入到模型推理', guide_styles['subtitle']),
             Paragraph(f'版本 {escape(str(report["version"]))}  ·  {escape(revision_date)}', guide_styles['note']),
             Spacer(1, 25)]
    cover = [
        ('完整流程', '13 章：安装、数据、模型参数、单任务、批量调参、序列与深度、图表、推理和排错'),
        ('操作截图', f'{len(report["screenshots"])} 张真实工作台界面截图，关键设置、结果和入口均有示例'),
        ('演示数据', '180 行确定性合成数据；单任务与 Grid 搜索实际运行，图片中的分数仅用于说明操作'),
    ]
    table = Table([[Paragraph(escape(left), guide_styles['tableheader']),
                    Paragraph(escape(right), guide_styles['body'])] for left, right in cover],
                  colWidths=[95, BODY_WIDTH - 95])
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), BLUE), ('BACKGROUND', (1, 0), (1, -1), PALE),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('TOPPADDING', (0, 0), (-1, -1), 12), ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
        ('LEFTPADDING', (0, 0), (-1, -1), 10), ('RIGHTPADDING', (0, 0), (-1, -1), 10),
    ]))
    story.extend([table, Spacer(1, 26),
                  Paragraph('推荐阅读顺序', guide_styles['smallheading']),
                  Paragraph('第一次使用：第 1-6 章；自动比较模型：第 7 章；HMM / LSTM / GRU：第 8 章；保存、推理和排错：第 9-13 章。', guide_styles['body']),
                  Paragraph('截图均来自本机 Qt 工作台的直接界面采集。当前向导页面使用源码运行采集，仍适用的功能截图保留此前已核验的安装版图片；每张图片的来源与 SHA-256 见截图记录。PDF 目录可点击跳转，也可用阅读器搜索。', guide_styles['note']),
                  Paragraph('配套合成 CSV：docs/assets/pdf-guide/demo-data.csv。完整 Markdown、API 文档与示例随项目提供。', guide_styles['note']),
                  PageBreak(), Paragraph('目录', guide_styles['heading'])])
    toc = TableOfContents()
    toc.levelStyles = [guide_styles['toc']]
    story.extend([toc, Spacer(1, 18),
                  Paragraph('每章操作截图位于该章正文后。图中目录、任务编号和模型路径是本次合成演示的本机路径，使用时替换为自己的位置。', guide_styles['note'])])
    intro = '\n'.join(sections[0].splitlines()[1:]).strip()
    story.extend([Paragraph('工作台与模型范围', guide_styles['smallheading'])])
    story.extend(parse_markdown(intro, guide_styles))
    figure_number = 0
    for chapter_number, section in enumerate(sections[1:], 1):
        title, _, body = section.partition('\n')
        story.extend([NextPageTemplate('portrait'), PageBreak()])
        heading = Paragraph(escape(title.strip()), guide_styles['heading'])
        heading.chapter_key = f'chapter-{chapter_number}'
        story.append(heading)
        story.extend(parse_markdown(body, guide_styles))
        pictures = [item for item in report['screenshots'] if item['chapter'] == chapter_number]
        for item in pictures:
            figure_number += 1
            wide = item['orientation'] == 'landscape'
            story.extend([NextPageTemplate('landscape' if wide else 'portrait'), PageBreak()])
            story.append(KeepTogether([
                Paragraph(escape(title.strip()) + ' · 界面示例', guide_styles['figureheading']),
                *picture_flowables(item, args.assets, figure_number, guide_styles, wide=wide),
            ]))
    story.extend([Spacer(1, 14),
                  Paragraph('截图与复现说明', guide_styles['smallheading']),
                  Paragraph('本 PDF 未使用真实用户数据。C01 单任务实际完成训练、冻结、一次最终测试和导出；批量示例实际执行 3 个 Grid 候选。HMM 截图仅说明配置入口。截图采集脚本与 PDF 构建脚本位于 packaging，生成来源和哈希记录位于 docs/assets/pdf-guide/screenshots.json。', guide_styles['note'])])
    doc.multiBuild(story)
    reader = PdfReader(args.output)
    extracted = '\n'.join(page.extract_text() or '' for page in reader.pages)
    for word in ('整体图文指南', '遗传', '退火', '隐马尔可夫', '诊断', 'Python API', '执行最终测试'):
        assert word in extracted, word
    for number in range(1, figure_number + 1):
        assert f'图 {number} ·' in extracted, f'missing figure caption {number}'
    table_numbers = [int(number) for number in re.findall(r'<p align="center"><strong>表 (\d+) ·', text)]
    assert table_numbers == list(range(1, len(table_numbers) + 1)), table_numbers
    for number in table_numbers:
        assert f'表 {number} ·' in extracted, f'missing table caption {number}'
    outline_titles = [item.title for item in reader.outline if hasattr(item, 'title')]
    assert len(outline_titles) == 13, outline_titles
    assert all(len((page.extract_text() or '').strip()) > 65 for page in reader.pages), 'empty content page'
    image_pages = [page for page in reader.pages if len(page.images)]
    assert figure_number == len(report['screenshots']), 'every manifest screenshot must produce one figure'
    assert len(image_pages) == figure_number, 'one complete figure per image page'
    expected_landscape = sum(item['orientation'] == 'landscape' for item in report['screenshots'])
    assert sum(float(p.mediabox.width) > float(p.mediabox.height) for p in reader.pages) == expected_landscape
    links = [a.get_object() for p in reader.pages for a in p.get('/Annots', [])
             if a.get_object().get('/Subtype') == '/Link']
    destinations = {a['/Dest'][0].idnum for a in links}
    expected = {reader.pages[reader.get_destination_page_number(item)].indirect_reference.idnum
                for item in reader.outline if hasattr(item, 'title')}
    assert destinations == expected and len(destinations) == 13, 'all chapter links must resolve'
    embedded_fonts = {}
    for page in reader.pages:
        for font in page['/Resources'].get('/Font', {}).values():
            font = font.get_object()
            descriptor = font.get('/FontDescriptor')
            if descriptor:
                descriptor = descriptor.get_object()
                embedded_fonts[str(font['/BaseFont'])] = any(
                    key in descriptor for key in ('/FontFile', '/FontFile2', '/FontFile3'))
    assert embedded_fonts and all(embedded_fonts.values()), 'Chinese fonts must be embedded'
    output_report = {
        'version': report['version'], 'PDF': str(args.output.resolve()),
        'pages': len(reader.pages), 'chapters': len(outline_titles), 'screenshots': figure_number,
        'manifest_screenshots': len(report['screenshots']),
        'bytes': args.output.stat().st_size, 'sha256': hashlib.sha256(args.output.read_bytes()).hexdigest(),
        'source_sha256': hashlib.sha256(args.source.read_bytes()).hexdigest(),
        'text_and_bookmark_checks': 'PASS',
        'directory_links': len(links), 'linked_chapters': len(destinations), 'image_pages': len(image_pages),
        'screenshot_source': report['source_kind'], 'data_source': report['data_kind'],
        'embedded_fonts': embedded_fonts,
        'screenshot_manifest_sha256': hashlib.sha256((args.assets / 'screenshots.json').read_bytes()).hexdigest(),
        'portrait_pages': sum(float(page.mediabox.width) < float(page.mediabox.height) for page in reader.pages),
        'landscape_pages': sum(float(page.mediabox.width) > float(page.mediabox.height) for page in reader.pages),
    }
    (args.output.parent / 'PDF-guide-checks.json').write_text(
        json.dumps(output_report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(output_report, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
