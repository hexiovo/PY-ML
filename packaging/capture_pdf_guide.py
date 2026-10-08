"""Capture real Qt widgets and computed synthetic demo results for the PDF."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as metadata
import json
import os
from pathlib import Path
import sqlite3
import time
from datetime import date

os.environ['QT_QPA_PLATFORM'] = 'offscreen'
os.environ.setdefault('QT_SCALE_FACTOR', '1.5')

from pyml_workbench.gui import WorkbenchWindow, _configure_application_font
import pyml_workbench.gui as gui_module
from pyml_workbench.batch_gui import BatchSearchDialog
from pyml_workbench.plot_cache import loaded_dataset_plot_source
from pyml_workbench.plot_dialog import PlotDialog
from pyml_workbench.plotting import PlotKind, available_plot_specs, build_plot_payload
from PySide6.QtCore import QProcess, QPoint, QRect, Qt
from PySide6.QtWidgets import QApplication, QGroupBox, QScrollArea
import pandas as pd
from sklearn.datasets import make_classification


def main():
    root = Path(__file__).resolve().parents[1]
    gui_path = Path(gui_module.__file__).resolve()
    source_kind = (
        'Source checkout Qt application, direct QWidget captures; offscreen platform'
        if gui_path.is_relative_to(root / 'src')
        else 'Installed Qt application, direct QWidget captures; offscreen platform'
    )
    parser = argparse.ArgumentParser()
    parser.add_argument('--assets', type=Path, default=root / 'docs/assets/pdf-guide')
    parser.add_argument('--work', type=Path, default=root / 'tmp/pdfs/demo')
    parser.add_argument('--wizard-only', action='store_true',
                        help='refresh source-sensitive wizard screenshots and merge them with the existing guide manifest')
    parser.add_argument('--window-size', default='1420x930', metavar='WIDTHxHEIGHT',
                        help='logical Qt main-window size used for screenshots (default: 1420x930)')
    args = parser.parse_args()
    try:
        window_width, window_height = (int(value) for value in args.window_size.lower().split('x', 1))
    except ValueError as exc:
        raise SystemExit('--window-size must be WIDTHxHEIGHT, for example 1420x930') from exc
    if window_width < 1060 or window_height < 720:
        raise SystemExit('--window-size must be at least 1060x720')
    assets = args.assets.resolve()
    work = args.work.resolve()
    assets.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    os.environ['PYML_LOG_ROOT'] = str(work / 'logs')
    app = QApplication.instance() or QApplication([])
    _configure_application_font(app)
    captures = []

    def events():
        for _ in range(3):
            app.processEvents()

    def capture(widget, name, title, chapter, note, orientation='portrait', rectangle=None,
                scroll_area=None, scroll_to_top=False, event_cycles=3):
        widget.ensurePolished()
        for _ in range(event_cycles):
            app.processEvents()
        if scroll_area is not None and scroll_to_top:
            scroll_area.verticalScrollBar().setValue(0)
            events()
        if rectangle is None:
            pixmap = widget.grab()
        else:
            visible_rectangle = rectangle.intersected(widget.rect())
            assert visible_rectangle == rectangle, f'capture rectangle exceeds visible widget: {name}'
            pixmap = widget.grab(visible_rectangle)
        destination = assets / (name + '.png')
        assert pixmap.save(str(destination)), destination
        captures.append({
            'name': name, 'file': destination.name, 'title': title,
            'chapter': chapter, 'note': note, 'orientation': orientation,
            'pixels': [pixmap.width(), pixmap.height()],
            'sha256': hashlib.sha256(destination.read_bytes()).hexdigest(),
            'source_kind': source_kind,
        })
        print('CAPTURE ' + name, flush=True)

    def wait(owner):
        started = time.monotonic()
        while time.monotonic() - started < 90:
            app.processEvents()
            if owner.process.state() == QProcess.ProcessState.NotRunning and owner._worker_action is None:
                break
            time.sleep(0.02)
        assert owner.process.state() == QProcess.ProcessState.NotRunning, 'worker timeout'
        error = getattr(owner, 'last_error', None) or getattr(owner, '_last_error', None)
        assert not error, error

    x, y = make_classification(n_samples=180, n_features=2, n_redundant=0,
                               class_sep=1.3, flip_y=0.04, random_state=42)
    frame = pd.DataFrame({'temperature': (20 + x[:, 0] * 4).round(3),
                          'pressure': (100 + x[:, 1] * 7).round(3), 'label': y})
    sample = (work if args.wizard_only else assets) / 'demo-data.csv'
    frame.to_csv(sample, index=False, encoding='utf-8')
    window = WorkbenchWindow(preferences_path=work / 'preferences.json')
    batch = None
    sequence = None
    dialogs = []
    try:
        window.resize(window_width, window_height)
        window.show()
        assert window.load_data(sample)
        if args.wizard_only:
            # The embedded guide text changes with each documented release.
            # Refresh its screenshot along with the source-sensitive wizard views.
            window.overall_guide_button.click()
            window._guide_dialog.resize(800, 600)
            capture(window._guide_dialog, 'offline-guide', '应用内操作指南', 1,
                    '左侧目录跳转章节，上方按关键词搜索；Enter 向前查找，F1 可再次打开指南。')
            window._guide_dialog.close()
        left_scroll = next(
            scroll for scroll in window.findChildren(QScrollArea)
            if scroll.widget() is not None and scroll.widget().isAncestorOf(window.data_group)
        )
        if args.wizard_only:
            manifest_path = assets / 'screenshots.json'
            assert manifest_path.is_file(), 'refresh mode requires an existing guide manifest'
            report = json.loads(manifest_path.read_text(encoding='utf-8'))
            original_source = report.get('source_kind', 'Existing Qt screenshot source')
            sample_hash = hashlib.sha256(sample.read_bytes()).hexdigest()
            assert report.get('sample_sha256') == sample_hash, (
                'refreshed screenshots must use the same deterministic demo data'
            )

            window.target_combo.setCurrentIndex(window.target_combo.findData('label'))
            window.task_combo.setCurrentIndex(window.task_combo.findData('classification'))
            window.model_combo.setCurrentIndex(window.model_combo.findData('C01'))
            window.output_dir_edit.setText(str(work / 'single-output'))
            window.source_path_edit.setCursorPosition(0)
            window.source_path_edit.deselect()
            window.output_dir_edit.setCursorPosition(0)
            window.output_dir_edit.deselect()
            window.search_fits_spin.setValue(1)
            window.search_proposals_spin.setValue(1)
            window.search_minutes_spin.setValue(10)
            window.parallel_workers_spin.setValue(1)
            selected_models = {'C01', 'C02'}
            for index in range(window.algorithm_list.count()):
                item = window.algorithm_list.item(index)
                model_id = str(item.data(Qt.ItemDataRole.UserRole))
                item.setCheckState(
                    Qt.CheckState.Checked if model_id in selected_models else Qt.CheckState.Unchecked
                )
            assert selected_models.issubset(set(window._checked_batch_models()))

            window._set_wizard_step(0)
            events()
            capture(window.data_group, 'data-selection', '数据预览与特征 / 目标选择', 4,
                    '演示表格共 180 行。label 是目标，temperature 和 pressure 是特征。')

            setup_pages = [
                (1, 'wizard-preprocessing-split', '预处理与划分', 4,
                 '检查训练 / 验证 / 测试划分和 seed；预处理只在训练分区拟合。'),
                (2, 'wizard-algorithm-metrics', '算法与评价', 5,
                 '选择两个兼容分类模型，并勾选需要查看和导出的 validation 指标。'),
                (3, 'wizard-resources', '资源与输出', 5,
                 '设置 CPU worker、每模型拟合 / 提案预算、活动时限和新结果目录。GPU 仅在可用时为 N01 提供。'),
            ]
            for step, name, title, chapter, note in setup_pages:
                window._set_wizard_step(step)
                events()
                point = left_scroll.mapTo(window, QPoint(0, 0))
                right_edge = point.x() + left_scroll.width()
                rectangle = QRect(0, 0, right_edge, window.height()).intersected(window.rect())
                capture(window, name, title, chapter, note, 'portrait', rectangle,
                        scroll_area=left_scroll)
                if step == 2:
                    capture(window.model_group, 'model-parameters', '单任务模型与常用参数', 5,
                            '算法比较页同时保留单任务模型选择和常用参数；高级设置使用 JSON。')

            window._set_wizard_step(3)
            events()
            assert window._start_selected_workflow(), window.last_error or 'wizard workflow failed to start'
            progress_deadline = time.monotonic() + 90
            while time.monotonic() < progress_deadline:
                app.processEvents()
                if window._wizard_completed_fits > 0 or window.process.state() == QProcess.ProcessState.NotRunning:
                    break
                time.sleep(0.02)
            assert window._wizard_completed_fits > 0 or window.process.state() != QProcess.ProcessState.NotRunning, (
                window.last_error or 'validation worker ended before reporting progress'
            )
            window._set_wizard_step(4)
            capture(window, 'wizard-monitor', '真实验证运行监控与阶段进展', 7,
                    '界面按真实完成的拟合数更新进度和验证指标；剩余时间在首个验证结果后估算。',
                    'landscape', event_cycles=0)
            wait(window)
            payload = window._batch_run_payload
            outcomes = payload.get('outcomes') if isinstance(payload, dict) else None
            assert isinstance(outcomes, list), 'wizard did not return completed validation outcomes'
            completed = [item for item in outcomes if isinstance(item, dict) and item.get('status') == 'completed']
            assert {str(item.get('model_id')) for item in completed} == selected_models
            assert all(isinstance(item.get('winner_metrics'), dict) for item in completed)
            assert window._wizard_completed_fits == window._wizard_total_fits == 2
            history_path = window._wizard_batch_directory / 'history.sqlite3'
            with sqlite3.connect(history_path) as connection:
                job_rows = connection.execute(
                    'SELECT model_id, status, actual_fit_count, proposal_count FROM jobs ORDER BY model_id'
                ).fetchall()
            assert len(job_rows) == 2 and all(row[1:] == ('completed', 1, 1) for row in job_rows), job_rows

            window._set_wizard_step(5)
            window.export_model_check.setChecked(True)
            events()
            assert window.export_comparison_button.isEnabled(), 'completed validation results are not exportable'
            assert window.export_wizard_comparison(), window.result_status_label.text()
            events()
            export_manifest = Path(window.artifact_paths['export_manifest'])
            export_info = json.loads(export_manifest.read_text(encoding='utf-8'))
            assert export_info['scope'] == 'validation'
            assert {
                'comparison_csv', 'comparison_xlsx', 'comparison_json', 'comparison_html',
                'validation_predictions_csv', 'chart_balanced_accuracy',
            } <= set(export_info['files'])
            assert Path(window.artifact_paths['export_manifest']).is_file()
            capture(window.result_export_group, 'wizard-results-export', '验证结果与所选格式导出', 7,
                    '两种模型的真实 validation 比较已完成；本例导出 CSV、XLSX、JSON、HTML、逐样本结果、模型和比较图。测试集未用于搜索。')
            capture(window, 'main-overview', '分步向导主窗口与真实结果', 3,
                    '六步向导可逐步确认设置；验证比较完成后，结果页显示真实指标并可导出所选内容。',
                    'landscape')

            old_items = report.get('screenshots', [])
            for item in old_items:
                item.setdefault('source_kind', original_source)
            merged = {item['name']: item for item in old_items}
            for item in captures:
                merged[item['name']] = item
            preferred_order = [
                'offline-guide', 'main-overview', 'data-selection', 'data-overview',
                'wizard-preprocessing-split', 'model-parameters', 'wizard-algorithm-metrics',
                'wizard-resources', 'validation-results', 'final-exports',
                'batch-configuration', 'batch-results', 'wizard-monitor', 'wizard-results-export',
                'sequence-configuration', 'basic-scatter', 'validation-confusion',
                'inference-controls', 'file-and-diagnostics',
            ]
            preferred_index = {name: index for index, name in enumerate(preferred_order)}
            screenshots = sorted(
                merged.values(),
                key=lambda item: (int(item['chapter']), preferred_index.get(item['name'], len(preferred_index))),
            )
            sources = sorted({str(item.get('source_kind', 'unknown')) for item in screenshots})
            source_summary = (
                sources[0] if len(sources) == 1 else
                'Mixed Qt captures; earlier verified screenshots were retained and changed wizard screens refreshed. '
                'See each screenshot source_kind for provenance.'
            )
            report.update({
                'version': metadata.version('pyml-workbench'),
                'revision_date': date.today().isoformat(),
                'source_kind': source_summary,
                'data_kind': 'Deterministic synthetic data, no user data',
                'sample_rows': len(frame),
                'sample_sha256': sample_hash,
                'wizard_validation': {
                    'models': sorted(selected_models),
                    'method': 'grid',
                    'actual_fit_count': sum(int(row[2]) for row in job_rows),
                    'proposal_count': sum(int(row[3]) for row in job_rows),
                    'outcomes': {str(item['model_id']): item['winner_metrics'] for item in completed},
                    'export_scope': export_info['scope'],
                    'export_files': sorted(export_info['files']),
                },
                'screenshots': screenshots,
            })
            manifest_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
            print(json.dumps({
                'status': 'PASS', 'refreshed_screenshots': len(captures),
                'retained_screenshots': len(screenshots) - len(captures),
                'total_screenshots': len(screenshots), 'models': sorted(selected_models),
                'jobs': job_rows, 'export_manifest': str(export_manifest),
            }, ensure_ascii=False, indent=2), flush=True)
            return
        window.target_combo.setCurrentIndex(window.target_combo.findData('label'))
        window.model_combo.setCurrentIndex(window.model_combo.findData('C01'))
        window.output_dir_edit.setText(str(work / 'single-output'))
        events()
        capture(window, 'main-overview', '主窗口与操作指南入口', 3,
                '顶部打开指南和批量队列；左侧加载数据与配置，右侧查看结果。左侧可滚动。', 'landscape')
        capture(window.data_group, 'data-selection', '数据预览与特征 / 目标选择', 4,
                '演示表格共 180 行。label 是目标，temperature 和 pressure 是特征。')
        overview = window.show_data_overview()
        overview.resize(790, 580)
        capture(overview, 'data-overview', '完整数据概况', 4,
                '查看完整数据的行列数、列类型、缺失数和 describe 统计；预览行数不等于总样本数。')
        overview.close()
        capture(window.model_group, 'model-parameters', '任务、模型与参数', 5,
                '选定 C01 分类模型；常用参数与高级 JSON 分开填写，seed 固定数据划分。')
        window.overall_guide_button.click()
        window._guide_dialog.resize(800, 600)
        capture(window._guide_dialog, 'offline-guide', '应用内操作指南', 1,
                '左侧目录跳转章节，上方按关键词搜索；Enter 向前查找，F1 可再次打开指南。')
        window._guide_dialog.close()

        assert window.start_training()
        wait(window)
        assert window._prepared
        assert 'validation' in window.current_metrics and 'test' not in window.current_metrics
        window.result_tabs.setCurrentIndex(0)
        capture(window.result_tabs, 'validation-results', '训练与验证结果', 6,
                '这是实际合成数据计算结果。当前只展示 train / validation，测试集尚未评分。')
        assert window.finalize_training()
        wait(window)
        assert window._frozen
        assert window.finalize_button.text() == '执行最终测试（一次）'
        assert window.finalize_training()
        wait(window)
        assert window.artifact_paths
        audit = window.bound_result['audit']
        assert audit['test_evaluation_count'] == 1
        window.result_tabs.setCurrentIndex(window.result_tabs.count() - 1)
        capture(window.result_tabs, 'final-exports', '最终测试后的导出与审计', 6,
                '单任务先冻结再测试同一个模型。结果目录含配置、指标、逐行结果和 model.joblib。')

        data_source = loaded_dataset_plot_source(window.dataset)
        scatter = next(spec for spec in available_plot_specs(data_source) if spec.kind == PlotKind.SCATTER)
        plot = PlotDialog([build_plot_payload(data_source, scatter)], window)
        dialogs.append(plot)
        plot.resize(800, 580)
        plot.show()
        capture(plot, 'basic-scatter', '完整数据的数值散点图', 9,
                '基础图表使用已加载表格，不启动模型；底部可保存当前 PNG 或 SVG。')
        plot.close()
        validation_source = window._session_plot_sources['validation']
        confusion = next(spec for spec in available_plot_specs(validation_source)
                         if spec.kind == PlotKind.CLASSIFICATION_CONFUSION)
        plot = PlotDialog([build_plot_payload(validation_source, confusion)], window)
        dialogs.append(plot)
        plot.resize(800, 580)
        plot.show()
        capture(plot, 'validation-confusion', '验证分区的分类结果图', 9,
                '混淆矩阵来自已核验 validation 缓存，不重新预测或测试。分区及有效样本数在底部。')
        plot.close()

        inference_group = next(group for group in window.findChildren(QGroupBox)
                               if group.title() == '已加载模型的推理')
        capture(inference_group, 'inference-controls', '已加载模型的推理入口', 10,
                '选择导出的可信 model.joblib，再选择新数据。只有模型实际支持的操作才会启用。')
        file_menu = window.menuBar().actions()[0].menu()
        file_menu.show()
        capture(file_menu, 'file-and-diagnostics', '文件菜单：配置、日志与诊断', 11,
                '参数配置 JSON、最近文件、日志目录和本机诊断 ZIP 都在文件菜单中。')
        file_menu.hide()

        batch = BatchSearchDialog(window)
        batch.resize(1100, 1100)
        batch.root_edit.setText(str(work / 'batch'))
        batch.datasets_edit.setPlainText(json.dumps([{
            'source_path': str(sample), 'task': 'classification', 'target_column': 'label',
            'feature_columns': ['temperature', 'pressure'], 'models': ['C01'], 'seed': 42,
        }], ensure_ascii=False, indent=2))
        batch.spaces_edit.setPlainText(json.dumps({'C01': {
            'fields': {'C': {'type': 'real', 'low': 0.1, 'high': 10,
                              'log': True, 'values': [0.1, 1.0, 10.0]}},
            'fixed': {'max_iter': 500},
        }}, ensure_ascii=False, indent=2))
        batch.max_fits_spin.setValue(3)
        batch.max_proposals_spin.setValue(15)
        batch.minutes_spin.setValue(2)
        batch.parallel_spin.setValue(1)
        batch.show()
        events()
        top_height = batch.pause_button.geometry().bottom() + 8
        capture(batch, 'batch-configuration', '批量队列与网格搜索配置', 7,
                '本例搜索 C 的三个候选值；拟合上限 3 次、提案上限 15 次、时限 2 分钟、1 个 worker。实际完成 3 次拟合和 3 次提案。',
                'landscape', QRect(0, 0, batch.width(), top_height))
        assert batch.start_batch()
        wait(batch)
        batch.refresh_history()
        assert batch.jobs_table.rowCount() == 1
        batch.jobs_table.selectRow(0)
        events()
        assert batch.jobs_table.item(0, 4).text() == 'completed'
        with sqlite3.connect(work / 'batch/history.sqlite3') as connection:
            batch_jobs = connection.execute(
                'SELECT job_id, status, actual_fit_count, proposal_count FROM jobs'
            ).fetchall()
        assert len(batch_jobs) == 1 and batch_jobs[0][1:] == ('completed', 3, 3)
        top = batch.freeze_button.geometry().top() - 8
        bottom = batch.status_label.geometry().bottom() + 8
        capture(batch, 'batch-results', '批量历史、验证 winner 与最终化入口', 7,
                '真实 Grid 任务已 completed。先选中任务并冻结验证 winner，再决定合并重拟合和最终测试。',
                'landscape', QRect(0, top, batch.width(), bottom - top))

        sequence_sample = work / 'sequence-demo.csv'
        sequence_frame = frame.drop(columns='label').copy()
        sequence_frame['device_id'] = ['device_' + str(index // 30) for index in range(len(frame))]
        sequence_frame['timestamp'] = [f'2026-01-01T{index % 30:02d}:00:00Z'
                                        if index % 30 < 24 else f'2026-01-02T{index % 30 - 24:02d}:00:00Z'
                                        for index in range(len(frame))]
        sequence_frame.to_csv(sequence_sample, index=False, encoding='utf-8')
        sequence = WorkbenchWindow(preferences_path=work / 'sequence-preferences.json')
        sequence.resize(1060, 720)
        sequence.show()
        assert sequence.load_data(sequence_sample)
        sequence.task_combo.setCurrentIndex(sequence.task_combo.findData('sequence_modeling'))
        sequence.model_combo.setCurrentIndex(sequence.model_combo.findData('H01'))
        sequence.sequence_config_edit.setPlainText(
            '{\n "group_column": "device_id", "time_column": "timestamp",\n'
            ' "order_mode": "time", "observation_columns": ["temperature", "pressure"]\n}'
        )
        events()
        capture(sequence.model_group, 'sequence-configuration', 'HMM 的分组、时间和观测配置', 8,
                '这里只展示真实配置入口，未运行 HMM。组 / 时间是结构列，不会被当成观测特征。')
        for capture_item in captures:
            assert len((assets / capture_item['file']).read_bytes()) > 1000
        report = {
            'version': metadata.version('pyml-workbench'),
            'revision_date': date.today().isoformat(),
            'source_kind': source_kind,
            'data_kind': 'Deterministic synthetic data, no user data',
            'sample_rows': len(frame), 'sample_sha256': hashlib.sha256(sample.read_bytes()).hexdigest(),
            'single_actual_stages': ['train', 'freeze', 'test', 'export'],
            'single_test_evaluation_count': audit['test_evaluation_count'],
            'batch_actual_search_fits': batch_jobs[0][2], 'batch_method': 'grid',
            'batch_database_check': {'job_id': batch_jobs[0][0], 'status': batch_jobs[0][1],
                                     'actual_fit_count': batch_jobs[0][2], 'proposal_count': batch_jobs[0][3]},
            'screenshots': captures,
        }
        (assets / 'screenshots.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        print(f'PASS {len(captures)} screenshots from real UI and computed synthetic results', flush=True)
    finally:
        if batch is not None:
            batch.close()
        if sequence is not None:
            sequence.close()
        for dialog in dialogs:
            # Plot dialogs delete on close, so they may already be destroyed.
            try:
                dialog.close()
            except RuntimeError:
                pass
        window.close()


if __name__ == '__main__':
    main()
