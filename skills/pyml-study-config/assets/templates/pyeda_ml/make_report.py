"""Render an APA-style PYEDA report from the recorded run and supplied study metadata."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.font_manager import fontManager
from docx import Document
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_TAB_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor


BODY_PT = 13
SMALL_PT = 11
FIGURE_WIDTH = 6.2
REFERENCE = ("Cawley, G. C., & Talbot, N. L. C. (2010). On over-fitting in model selection and subsequent "
             "selection bias in performance evaluation. *Journal of Machine Learning Research, 11*, 2079–2107. "
             "https://www.jmlr.org/papers/v11/cawley10a.html")
BIB = """@article{cawley2010model,
  author = {Cawley, Gavin C. and Talbot, Nicola L. C.},
  title = {On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation},
  journal = {Journal of Machine Learning Research},
  year = {2010},
  volume = {11},
  pages = {2079--2107},
  url = {https://www.jmlr.org/papers/v11/cawley10a.html}
}
"""
METRICS = {"regression": ("r2", "mae", "rmse"),
           "classification": ("balanced_accuracy", "roc_auc", "average_precision")}


def read_json(path):
    with path.open(encoding="utf-8-sig") as stream:
        return json.load(stream)


def write_csv(path, headers, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(headers)
        writer.writerows(rows)


def chart_font():
    available = {font.name for font in fontManager.ttflist}
    for name in ("Microsoft YaHei", "SimSun", "Noto Sans CJK SC", "DejaVu Sans"):
        if name in available:
            return name
    return "DejaVu Sans"


def render_figures(metrics, split, figures, figure_data):
    blocks = []
    for (outcome, metric), models in sorted(metrics.items()):
        if metric == "roc_auc" and not models:
            continue
        items = sorted(models.items())
        for offset in range(0, len(items), 12):
            batch = items[offset:offset + 12]
            path = f"metric-{len(blocks) + 1:03}"
            means = [statistics.mean(values) for _, values in batch]
            fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, max(2.5, 1.5 + .26 * len(batch))), layout="constrained")
            positions = list(range(len(batch)))
            ax.scatter(means, positions, color="#174F78")
            ax.set_yticks(positions, [name for name, _ in batch])
            ax.set_xlabel(metric, fontsize=SMALL_PT)
            ax.tick_params(labelsize=SMALL_PT)
            ax.grid(axis="x", alpha=.25)
            fig.savefig(figures / f"{path}.png", dpi=200)
            fig.savefig(figures / f"{path}.pdf")
            plt.close(fig)
            write_csv(figure_data / f"{path}.csv", ["outcome", "model_id", "metric", "split", "mean", "iterations"],
                      [(outcome, name, metric, split, value, len(values)) for (name, values), value in zip(batch, means, strict=True)])
            blocks.append(("figure", (str(len(blocks) + 1), f"{outcome} 的 {metric}", path,
                                      "点表示可估计轮次的算术平均。分区及轮次数见图表数据。")))
    return blocks


def load_run(run, out, split):
    matrix = run / "machine-learning" / "method-matrix.csv"
    with matrix.open(encoding="utf-8-sig", newline="") as stream:
        slots = list(csv.DictReader(stream))
    metrics = {}
    evidence, long_rows, params, class_rows, confusion_rows = [], [], [], [], []
    for slot in slots:
        if slot["status"] != "completed":
            continue
        path = Path(slot["result_path"]).resolve()
        if not path.is_relative_to(run) or not path.is_file():
            raise ValueError("结果矩阵指向运行目录外或缺少结果文件")
        payload = read_json(path)
        identity = (slot["outcome"], slot["model_id"], int(slot["iteration"]))
        if identity != (payload["outcome"], payload["model_id"], payload["iteration"]):
            raise ValueError("结果矩阵与运行槽标识不一致")
        config = payload["config"]
        params.append([*identity, config["task"], config["split"]["strategy"], slot["seed"],
                       json.dumps(config.get("parameters", {}), ensure_ascii=False, sort_keys=True)])
        with path.open("rb") as stream:
            evidence.append([*identity, path.relative_to(run).as_posix(), hashlib.file_digest(stream, "sha256").hexdigest()])
        for part, values in payload["metrics"].items():
            if part not in {"train", "validation", split}:
                continue
            notes = values.get("metric_notes", {})
            class_order = values.get("score_class_order", [])
            if class_order:
                class_rows.append([*identity, part, json.dumps(class_order, ensure_ascii=False), class_order[-1] if len(class_order) == 2 else ""])
            for key, value in values.items():
                if not isinstance(value, (dict, list)):
                    long_rows.append([*identity, part, key, value, notes.get(key, "")])
                if part == split and key in METRICS[config["task"]] and isinstance(value, (int, float)) and math.isfinite(value):
                    metrics.setdefault((identity[0], key), {}).setdefault(identity[1], []).append(value)
        if config["task"] == "classification":
            prediction_file = path.with_name("results.csv")
            with prediction_file.open(encoding="utf-8-sig", newline="") as stream:
                predictions = csv.DictReader(stream)
                counts = {}
                for row in predictions:
                    if row["split"] in {"train", "validation", split}:
                        key = (row["split"], row["y_true"], row["y_pred"])
                        counts[key] = counts.get(key, 0) + 1
                confusion_rows.extend([*identity, *key, count] for key, count in sorted(counts.items()))
    table = out / "tables"
    identity = ["outcome", "model_id", "iteration"]
    write_csv(out / "evidence.csv", [*identity, "result_file", "sha256"], evidence)
    write_csv(table / "table_all_metrics_long.csv", [*identity, "split", "metric", "value", "reason"], long_rows)
    write_csv(table / "table_model_parameters.csv", [*identity, "task", "split_strategy", "seed", "parameters"], params)
    write_csv(table / "table_class_orders.csv", [*identity, "split", "class_order", "positive_class"], class_rows)
    write_csv(table / "table_classification_confusion.csv", [*identity, "split", "actual", "predicted", "count"], confusion_rows)
    (table / "table_method_results.csv").write_bytes(matrix.read_bytes())
    summary = [(outcome, model, metric, split, statistics.mean(values), min(values), max(values), len(values))
               for (outcome, metric), models in sorted(metrics.items()) for model, values in sorted(models.items())]
    write_csv(table / "table_model_aggregate.csv", ["outcome", "model_id", "metric", "split", "mean", "min", "max", "iterations"], summary)
    return slots, metrics


def report_pages(config, prep, summary, figures, split):
    study = config.get("study") or {}
    title = study.get("title") or "PYEDA 特征模型评估"
    mode = "独立最终测试" if split == "test" else "开发集验证"
    outcomes = prep["outcomes"]
    rows = [[name, prep["outcome_tasks"][name], str(prep["outcome_rows"][name]),
             str(prep["outcome_group_counts"][name])] for name in outcomes]
    counts = summary["machine_learning"]["counts"]
    coverage = "，".join(f"{key} {value} 项" for key, value in sorted(counts.items()))
    pages = [
        ("title", title, [("author", study.get("author") or "［作者姓名待补］"),
                         ("author", study.get("affiliation") or "［机构与部门待补］")]),
        ("section", "摘要", [("abstract", f"本次运行使用 PYEDA 转换表评估 {len(outcomes)} 个结局。"
                             f"当前阶段为{mode}。{coverage}。"),
                           ("keywords", "皮电活动，模型评估，验证")]),
        ("section", title, [("paragraph", study.get("research_question") or "研究问题及实验设计由研究者补充。")]),
        ("section", "方法", [
            ("paragraph", study.get("methods") or "本轮配置未提供信号采集、处理窗口或评分者细节。"),
            ("paragraph", f"转换来源为 {config['conversion']['source']}。使用 {len(prep['requested_features'])} 个已登记特征，"
                          f"对 {len(prep['all_missing_features'])} 个全缺失特征记录了处理策略。"
                          "逐模型参数、种子和切分策略见附录。"),
            ("table", ("1", "参与建模的观测及分组", ["结局", "任务", "观测", "组数"], rows, [.45, .23, .16, .16])),
        ]),
        ("section", "结果", [("paragraph", f"{mode}的模型指标见下图及附录。{coverage}。"), *figures]),
        ("section", "讨论", [
            ("paragraph", "这些数值用于描述已记录分区的预测表现。按验证结果选模产生的选择偏差需要由独立测试评估"
                          "（Cawley & Talbot, 2010）。"),
            ("paragraph", "所列观测次数与轮次数不代表独立参与者数。因果解释和外推范围需结合实际研究设计。"),
        ]),
        ("section", "参考文献", [("reference", REFERENCE)]),
        ("section", "附录 A", [("appendix_title", "运行状态及可复核结果"),
                              ("paragraph", "tables/ 保存逐项状态、参数、指标、类别顺序和混淆计数。evidence.csv 保存结果文件及其 SHA-256。")]),
    ]
    if prep.get("ioc"):
        audit = prep["ioc"]
        pages[3][2].append(("paragraph", f"IOC 匹配 {audit['matched_events']} 条合格事件，"
                                       f"{audit['unmatched_labeled_events']} 条有标签事件无匹配特征。"
                                       f"{audit['feature_timing']}。"))
    return pages


def font_run(run, size, *, bold=False):
    run.font.name = "Times New Roman"
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = RGBColor(0, 0, 0)
    fonts = run._element.get_or_add_rPr().rFonts
    fonts.set(qn("w:eastAsia"), "黑体" if bold else "宋体")
    return run


def doc_text(doc, text, size=BODY_PT, *, bold=False, style=None, keep=False, italic=False, markup=False):
    paragraph = doc.add_paragraph(style=style)
    pieces = text.split("*") if markup else [text]
    for index, piece in enumerate(pieces):
        run = font_run(paragraph.add_run(piece), size, bold=bold)
        run.italic = italic or bool(index % 2)
    paragraph.paragraph_format.keep_with_next = keep
    return paragraph


def build_docx(pages):
    doc = Document()
    section = doc.sections[0]
    section.page_width, section.page_height = Cm(21), Cm(29.7)
    section.top_margin = section.bottom_margin = Cm(2.54)
    section.left_margin = section.right_margin = Cm(2.54)
    normal = doc.styles["Normal"]
    normal.font.name = "Times New Roman"
    normal.font.size = Pt(BODY_PT)
    normal.paragraph_format.line_spacing = 1.5
    normal.paragraph_format.space_after = Pt(0)
    normal.paragraph_format.widow_control = True
    for style_name in ("Title", "Heading 1", "Heading 2", "Heading 3"):
        style = doc.styles[style_name]
        style.paragraph_format.line_spacing = 1.5
        style.paragraph_format.space_before = Pt(0 if style_name == "Title" else 9)
        style.paragraph_format.space_after = Pt(0)
        style.paragraph_format.left_indent = Pt(0)
        style.paragraph_format.first_line_indent = Pt(0)
        for border in style._element.xpath("./w:pPr/w:pBdr"):
            border.getparent().remove(border)
    zoom = doc.settings.element.find(qn("w:zoom"))
    if zoom is not None:
        zoom.set(qn("w:percent"), "100")
    header = section.header.paragraphs[0]
    doc.styles["Header"].paragraph_format.tab_stops.clear_all()
    header.paragraph_format.tab_stops.add_tab_stop(Cm(15.92), WD_TAB_ALIGNMENT.RIGHT)
    font_run(header.add_run("PYEDA 模型评估\t"), SMALL_PT)
    field = OxmlElement("w:fldSimple")
    field.set(qn("w:instr"), "PAGE")
    header._p.append(field)
    appendix_page_sections = False
    for role, title, blocks in pages:
        level = "Title" if role == "title" else "Heading 2" if role == "subsection" else "Heading 1"
        p = doc_text(doc, title, 21 if role == "title" else 14 if role == "subsection" else 17,
                     bold=True, style=level, keep=True)
        if role != "subsection":
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        if role == "title":
            p.paragraph_format.space_before = Pt(66)
            p.paragraph_format.space_after = Pt(16)
        if title.startswith("附录 "):
            appendix_page_sections = title in {"附录 A", "附录 B", "附录 C", "附录 D"}
        front_or_references = title in {"摘要", pages[0][1], "参考文献"} and role != "title"
        if front_or_references or title.startswith("附录 ") or (appendix_page_sections and role == "subsection"):
            p.paragraph_format.page_break_before = True
        for block_index, (kind, content) in enumerate(blocks):
            if kind == "heading":
                level = "Heading 3" if role == "subsection" else "Heading 2"
                doc_text(doc, content, BODY_PT, bold=True, style=level, keep=True, italic=level == "Heading 3")
            elif kind == "appendix_title":
                p = doc_text(doc, content, 17, bold=True, style="Heading 1", keep=True)
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            elif kind == "author":
                p = doc_text(doc, content)
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            elif kind in {"keywords", "note"}:
                label = "关键词。" if kind == "keywords" else "注。"
                p = doc_text(doc, f"*{label}* {content}", BODY_PT if kind == "keywords" else SMALL_PT, markup=True)
                if kind == "keywords":
                    p.paragraph_format.first_line_indent = Cm(1.27)
            elif kind == "figure":
                number, label, name, note = content
                p = doc_text(doc, f"图 {number}", BODY_PT, bold=True, keep=True)
                p.paragraph_format.space_before = Pt(9)
                doc_text(doc, label, BODY_PT, keep=True, italic=True)
                p = doc.add_paragraph()
                p.paragraph_format.line_spacing = 1
                p.paragraph_format.keep_with_next = True
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                p.add_run().add_picture(str(FIG / f"{name}.png"), width=Pt(72*FIGURE_WIDTH))
                p = doc_text(doc, f"*注。* {note}", SMALL_PT, markup=True)
                p.paragraph_format.keep_together = True
            elif kind == "table":
                number, label, headers, rows, widths = content
                p = doc_text(doc, f"表 {number}", BODY_PT, bold=True, keep=True)
                p.paragraph_format.space_before = Pt(9)
                doc_text(doc, label, BODY_PT, keep=True, italic=True)
                table = doc.add_table(rows=0, cols=len(headers))
                has_note = block_index + 1 < len(blocks) and blocks[block_index + 1][0] == "note"
                table.alignment = WD_TABLE_ALIGNMENT.CENTER
                table.autofit = False
                borders = OxmlElement("w:tblBorders")
                for side in ("top", "bottom"):
                    rule = OxmlElement(f"w:{side}")
                    rule.set(qn("w:val"), "single")
                    rule.set(qn("w:sz"), "6")
                    borders.append(rule)
                table._tbl.tblPr.insert_element_before(borders, "w:tblLayout", "w:tblLook")
                for column, width in zip(table.columns, widths):
                    column.width = Cm(15.92*width)
                for row_index, values in enumerate([headers, *rows]):
                    row = table.add_row()
                    row._tr.get_or_add_trPr().append(OxmlElement("w:cantSplit"))
                    if row_index == 0:
                        row._tr.get_or_add_trPr().append(OxmlElement("w:tblHeader"))
                    for cell, value, width in zip(row.cells, values, widths):
                        cell.width = Cm(15.92*width)
                        cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
                        p = cell.paragraphs[0]
                        p.paragraph_format.space_after = Pt(3)
                        p.paragraph_format.space_before = Pt(3)
                        p.paragraph_format.line_spacing = 1.15
                        p.paragraph_format.keep_with_next = row_index < len(rows) or has_note
                        font_run(p.add_run(str(value)), SMALL_PT, bold=row_index == 0)
                        if row_index == 0:
                            properties = cell._tc.get_or_add_tcPr()
                            cell_borders = OxmlElement("w:tcBorders")
                            rule = OxmlElement("w:bottom")
                            rule.set(qn("w:val"), "single")
                            rule.set(qn("w:sz"), "4")
                            cell_borders.append(rule)
                            properties.insert_element_before(cell_borders, "w:vAlign")
            else:
                p = doc_text(doc, content, markup=kind == "reference")
                if kind == "reference":
                    p.paragraph_format.left_indent = Cm(1.27)
                    p.paragraph_format.first_line_indent = Cm(-1.27)
                    p.paragraph_format.keep_together = True
                elif kind == "paragraph":
                    p.paragraph_format.first_line_indent = Cm(1.27)
                    p.paragraph_format.space_before = Pt(4)
    doc.core_properties.title = pages[0][1]
    doc.core_properties.subject = "AUT 与 PIT 皮电特征的探索性预测研究"
    doc.save(OUT / "report.docx")


def build_markdown(pages):
    lines = []
    for role, title, blocks in pages:
        prefix = "#" if role == "title" else "###" if role == "subsection" else "##"
        lines += [f"{prefix} {title}", ""]
        for kind, content in blocks:
            if kind == "figure":
                number, label, name, note = content
                lines += [f"**图 {number}**", "", f"*{label}*", "", f"![{label}](figures/{name}.png)", "", f"*注。* {note}", ""]
            elif kind == "table":
                number, label, headers, rows, _ = content
                lines += [f"**表 {number}**", "", f"*{label}*", "", "| " + " | ".join(headers) + " |",
                          "| " + " | ".join(["---"] * len(headers)) + " |"]
                lines += ["| " + " | ".join(map(str, row)) + " |" for row in rows]
                lines.append("")
            elif kind in {"heading", "appendix_title"}:
                lines += [f"{'####' if role == 'subsection' else '###'} {content}", ""]
            elif kind in {"keywords", "note"}:
                lines += [f"*{'关键词。' if kind == 'keywords' else '注。'}* {content}", ""]
            else:
                lines += [content, ""]
    (OUT / "report.md").write_text("\n".join(lines), encoding="utf-8")


def build_pdf(pages):
    # Use physical units so figures and type retain the same size as the Word copy.
    from matplotlib.font_manager import FontProperties
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    width, height = 595.28, 841.89
    margin = 72
    line_width = width - 2*margin
    with PdfPages(OUT / "report.pdf") as pdf:
        page_number = 0

        def new_page():
            nonlocal page_number
            page_number += 1
            fig = plt.figure(figsize=(width/72, height/72), dpi=72, facecolor="white")
            FigureCanvasAgg(fig)
            renderer = fig.canvas.get_renderer()
            fig.text(1-margin/width, 1-37/height, str(page_number), ha="right", fontsize=SMALL_PT, color="black")
            return fig, renderer, height-margin

        for role, title, blocks in pages:
            fig, renderer, y = new_page()
            families = [chart_font(), "Times New Roman", "DejaVu Sans"]
            heading_families = ["SimHei", "Times New Roman", chart_font()]

            def measure(text, size, bold):
                props = FontProperties(family=heading_families if bold else families, size=size)
                return renderer.get_text_width_height_descent(text, props, ismath=False)[0]

            def wrap_lines(text, size, bold, max_width):
                tokens = re.findall(r"[A-Za-z0-9_.:/×–-]+|.", text)
                line = ""
                output = []
                for token in tokens:
                    if line and measure(line + token, size, bold) > max_width:
                        output.append(line.rstrip())
                        line = token.lstrip()
                    else:
                        line += token
                if line:
                    output.append(line.rstrip())
                return output

            def ensure(space):
                nonlocal fig, renderer, y
                if y-space < margin:
                    pdf.savefig(fig)
                    plt.close(fig)
                    fig, renderer, y = new_page()

            def text(text, size=BODY_PT, bold=False, after=8, italic=False, center=False, before=0):
                nonlocal y
                lines = wrap_lines(text, size, bold, line_width)
                ensure(size*1.5*len(lines)+after+before)
                y -= before
                for line in lines:
                    fig.text(.5 if center else margin/width, y/height, line,
                             ha="center" if center else "left", va="top", fontsize=size,
                             color="black", fontfamily=heading_families if bold else families,
                             style="italic" if italic else "normal")
                    y -= size*1.5
                y -= after

            text(title, 21 if role == "title" else 14 if role == "subsection" else 17, True, 11,
                 center=role != "subsection", before=66 if role == "title" else 0)
            for kind, content in blocks:
                if kind == "figure":
                    number, label, name, note = content
                    pixels = plt.imread(FIG / f"{name}.png")
                    plot_height = FIGURE_WIDTH*72*pixels.shape[0]/pixels.shape[1]
                    note_lines = wrap_lines(note, SMALL_PT, False, line_width)
                    ensure(plot_height + 75 + len(note_lines)*SMALL_PT*1.5)
                    text(f"图 {number}", SMALL_PT, True, 1)
                    text(label, BODY_PT, False, 6, italic=True)
                    plot = fig.add_axes([(width-FIGURE_WIDTH*72)/(2*width), (y-plot_height)/height,
                                         FIGURE_WIDTH*72/width, plot_height/height])
                    plot.imshow(pixels)
                    plot.axis("off")
                    y -= plot_height + 5
                    text(f"注。{note}", SMALL_PT)
                elif kind == "table":
                    number, label, headers, rows, fractions = content
                    text(f"表 {number}", SMALL_PT, True, 1)
                    text(label, BODY_PT, italic=True)
                    for row_index, values in enumerate([headers, *rows]):
                        columns = [wrap_lines(str(value), SMALL_PT, row_index == 0, line_width*fraction-9)
                                   for value, fraction in zip(values, fractions)]
                        row_height = max(map(len, columns))*14+7
                        ensure(row_height+1)
                        if row_index == 0:
                            for rule_y in (y, y-row_height):
                                fig.add_artist(plt.Line2D([margin/width, 1-margin/width],
                                                          [rule_y/height]*2, color="black", linewidth=.6))
                        x = margin
                        for fraction, lines in zip(fractions, columns):
                            for i, line in enumerate(lines):
                                fig.text((x+4)/width, (y-4-i*14)/height, line, va="top", fontsize=SMALL_PT,
                                         color="black", fontfamily=heading_families if row_index == 0 else families)
                            x += line_width*fraction
                        y -= row_height
                    fig.add_artist(plt.Line2D([margin/width, 1-margin/width], [y/height]*2,
                                              color="black", linewidth=.6))
                    y -= 7
                else:
                    label = {"note": "注。", "keywords": "关键词。"}.get(kind, "")
                    text(label + (content.replace("*", "") if kind == "reference" else content),
                         SMALL_PT if kind == "note" else BODY_PT, kind in {"heading", "appendix_title"},
                         center=kind in {"author", "appendix_title"})
            pdf.savefig(fig)
            plt.close(fig)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--source", type=Path, help="兼容参数；证据来自已保存的运行回执")
    parser.add_argument("--docx-only", action="store_true")
    args = parser.parse_args(argv)
    run = args.run.resolve()
    global OUT, FIG
    OUT = args.out.resolve()
    FIG = OUT / "figures"
    data = OUT / "figure-data"
    for directory in (FIG, data, OUT / "tables"):
        directory.mkdir(parents=True, exist_ok=True)
    config = read_json(run / "config.json")
    summary = read_json(run / "oneclick-summary.json")
    prep = read_json(run / "machine-learning" / "data-preparation.json")
    evaluation = config["machine_learning"].get("evaluation_mode", "validation")
    split = "test" if evaluation == "final" else "validation"
    slots, metrics = load_run(run, OUT, split)
    figures = render_figures(metrics, split, FIG, data)
    pages = report_pages(config, prep, summary, figures, split)
    build_docx(pages)
    build_markdown(pages)
    if not args.docx_only:
        build_pdf(pages)
    (OUT / "references.bib").write_text(BIB, encoding="utf-8")
    check = {"status": "completed", "evaluation_mode": evaluation, "reported_split": split,
             "planned_slots": len(slots), "figures": len(figures), "outputs": ["report.docx", "report.md", "evidence.csv"],
             "visual_review": "pending"}
    if not args.docx_only:
        check["outputs"].append("report.pdf")
    (OUT / "report-check.json").write_text(json.dumps(check, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
