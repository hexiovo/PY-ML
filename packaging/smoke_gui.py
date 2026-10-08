"""Real frozen GUI startup plus source Qt integration with the frozen worker.

The integration controller needs the development venv; end users do not.
This deliberately reports the two verification scopes separately.
"""
from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from unittest.mock import patch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('distribution', type=Path)
    parser.add_argument('evidence', type=Path)
    parser.add_argument('--startup-only', action='store_true')
    parser.add_argument('--guide', action='store_true', help='Open the actual frozen offline guide with F1')
    args = parser.parse_args()
    distribution = args.distribution.resolve()
    evidence = args.evidence.resolve()
    evidence.mkdir(parents=True, exist_ok=True)
    executable = distribution / 'PYML-Workbench.exe'
    env = dict(os.environ)
    for key in ('PYTHONPATH', 'PYTHONHOME', 'QT_PLUGIN_PATH', 'QT_QPA_PLATFORM_PLUGIN_PATH', 'QT_QPA_PLATFORM'):
        env.pop(key, None)
    env['PATH'] = os.pathsep.join([str(Path(os.environ['SystemRoot']) / 'System32'), os.environ['SystemRoot']])
    windows = []
    user32 = ctypes.WinDLL('user32', use_last_error=True)
    callback_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    user32.GetWindowThreadProcessId.argtypes = [wintypes.HWND, ctypes.POINTER(wintypes.DWORD)]
    user32.GetWindowTextW.argtypes = [wintypes.HWND, wintypes.LPWSTR, ctypes.c_int]
    user32.SetForegroundWindow.argtypes = [wintypes.HWND]
    user32.PostMessageW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
    with tempfile.TemporaryDirectory(prefix='PYML-界面检查-') as temp:
        root = Path(temp)
        env['PYML_LOG_ROOT'] = str(root / 'logs')
        started = time.monotonic()
        process = subprocess.Popen([str(executable)], cwd=root, env=env)
        @callback_type
        def inspect_window(hwnd, _):
            pid = wintypes.DWORD()
            user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
            if pid.value == process.pid:
                buffer = ctypes.create_unicode_buffer(512)
                user32.GetWindowTextW(hwnd, buffer, len(buffer))
                if buffer.value:
                    children = []
                    @callback_type
                    def inspect_child(child, _):
                        text = ctypes.create_unicode_buffer(30000)
                        user32.GetWindowTextW(child, text, len(text))
                        if text.value:
                            children.append(text.value)
                        return True
                    user32.EnumChildWindows(hwnd, inspect_child, 0)
                    windows.append({'hwnd': int(hwnd), 'title': buffer.value, 'visible': bool(user32.IsWindowVisible(hwnd)), 'child_text': children})
            return True
        try:
            # A freshly extracted 1.2 GB distribution took 38 seconds on this
            # host; the previous 30-second check incorrectly rejected it.
            deadline = time.monotonic() + 60
            while time.monotonic() < deadline and process.poll() is None:
                windows.clear()
                user32.EnumWindows(inspect_window, 0)
                if any(w['title'] == 'PY-ML 机器学习工作台' and w['visible'] for w in windows):
                    break
                time.sleep(0.2)
            if not (process.poll() is None and any(w['title'] == 'PY-ML 机器学习工作台' and w['visible'] for w in windows)):
                (evidence / 'frozen-gui-failure.json').write_text(json.dumps({
                    'windows': windows, 'pid': process.pid, 'returncode': process.poll(),
                    'elapsed_seconds': round(time.monotonic() - started, 3),
                }, ensure_ascii=False, indent=2), encoding='utf-8')
                raise AssertionError(windows)
            (evidence / 'frozen-gui-startup.json').write_text(json.dumps({
                'overall': 'PASS', 'windows': windows, 'PYTHONPATH_removed': True,
                'startup_elapsed_seconds': round(time.monotonic() - started, 3),
                'PATH': 'Windows system directories only', 'cwd': 'external temporary Unicode directory'},
                ensure_ascii=False, indent=2), encoding='utf-8')
            print('PASS actual frozen GUI visible', flush=True)
            if args.guide:
                main_window = next(w for w in windows if w['title'] == 'PY-ML 机器学习工作台' and w['visible'])
                user32.SetForegroundWindow(main_window['hwnd'])
                # Native key messages target only this test's process/window.
                user32.PostMessageW(main_window['hwnd'], 0x0100, 0x70, 0x003B0001)
                user32.PostMessageW(main_window['hwnd'], 0x0101, 0x70, 0xC03B0001)
                deadline = time.monotonic() + 15
                while time.monotonic() < deadline and process.poll() is None:
                    windows.clear()
                    user32.EnumWindows(inspect_window, 0)
                    if any(w['title'] == '操作指南 · PY-ML 工作台' and w['visible'] for w in windows):
                        break
                    time.sleep(0.2)
                assert any(w['title'] == '操作指南 · PY-ML 工作台' and w['visible'] for w in windows), windows
                resource = distribution / '_internal/pyml_workbench/resources/overall-guide.md'
                published = distribution / 'docs/overall-guide.md'
                canonical = Path(__file__).resolve().parents[1] / 'docs/overall-guide.md'
                assert resource.read_bytes() == canonical.read_bytes()
                if published.exists():
                    assert published.read_bytes() == canonical.read_bytes()
                (evidence / 'frozen-guide.json').write_text(json.dumps({
                    'overall': 'PASS', 'windows': windows,
                    'entry': 'native F1 -> same help handler used by the header button',
                    'resource_matches_canonical_document': True,
                    'note': 'Header button, content, navigation/search covered separately by Qt tests.',
                    'environment': 'no Python path, system-only PATH, external Unicode cwd'},
                    ensure_ascii=False, indent=2), encoding='utf-8')
                print('PASS actual frozen offline guide window and bundled resource', flush=True)
        finally:
            if process.poll() is None:
                process.terminate()
            process.wait(timeout=10)

    if args.startup_only:
        return 0

    os.environ['QT_QPA_PLATFORM'] = 'offscreen'
    import pyml_workbench
    from pyml_workbench.gui import WorkbenchWindow
    from pyml_workbench.batch_gui import BatchSearchDialog
    from PySide6.QtCore import QProcess
    from PySide6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication(['frozen-worker-integration'])
    with tempfile.TemporaryDirectory(prefix='PYML-Qt-worker-') as temp:
        root = Path(temp)
        os.environ['PYML_LOG_ROOT'] = str(root / 'logs')
        source = root / 'data.csv'
        source.write_text('x,target\n' + ''.join(f'{i},{i%2}\n' for i in range(80)), encoding='utf-8')
        window = WorkbenchWindow(preferences_path=root / 'preferences.json')
        dialog = BatchSearchDialog(window)
        def wait(process, owner):
            deadline = time.monotonic() + 120
            while time.monotonic() < deadline:
                app.processEvents()
                if process.state() == QProcess.ProcessState.NotRunning and owner._worker_action is None:
                    break
                time.sleep(0.01)
            assert process.state() == QProcess.ProcessState.NotRunning
        try:
            assert window.load_data(source)
            window.target_combo.setCurrentIndex(window.target_combo.findData('target'))
            window.model_combo.setCurrentIndex(window.model_combo.findData('C01'))
            window.output_dir_edit.setText(str(root / 'single-output'))
            with patch.object(sys, 'frozen', True, create=True), patch.object(sys, 'executable', str(executable)):
                assert window.start_training()
                wait(window.process, window)
                if not window._prepared or window.last_error is not None:
                    (evidence / 'qt-worker-failure.json').write_text(json.dumps({
                        'prepared': window._prepared, 'last_error': window.last_error,
                        'status': window.result_status_label.text(), 'state': window.state_label.text(),
                        'stderr': window.stderr_text, 'events': window.events_text.toPlainText(),
                        'program': window.process.program(), 'arguments': window.process.arguments(),
                        'exit_code': window.process.exitCode()}, ensure_ascii=False, indent=2), encoding='utf-8')
                assert window._prepared and window.last_error is None, window.last_error
                window.finalize_button.click()
                wait(window.process, window)
                assert window._frozen
                window.finalize_button.click()
                wait(window.process, window)
                assert window.artifact_paths and window.bound_result, window.last_error
                dialog.root_edit.setText(str(root / 'batch'))
                dialog.datasets_edit.setPlainText(json.dumps([{'source_path': str(source),
                    'task': 'classification', 'target_column': 'target', 'feature_columns': ['x'],
                    'models': ['C01'], 'seed': 42}]))
                dialog.spaces_edit.setPlainText(json.dumps({'C01': {'fields': {}, 'fixed': {'max_iter': 80}}}))
                dialog.max_fits_spin.setValue(1)
                dialog.max_proposals_spin.setValue(1)
                dialog.minutes_spin.setValue(1)
                assert dialog.start_batch()
                wait(dialog.process, dialog)
                assert dialog._last_error is None, dialog._last_error
            report = {'overall': 'PASS', 'scope': 'source Qt controller -> actual frozen worker EXE',
                'single': ['train', 'freeze', 'test', 'automatic export'], 'batch': 'one actual search fit',
                'worker_program': window.process.program(), 'model_count': len(pyml_workbench.list_models()),
                'note': 'GUI EXE startup was verified separately; controller uses development Python.'}
            catalog_path = Path(pyml_workbench.__file__).resolve().with_name('model_catalog.json')
            catalog = json.loads(catalog_path.read_text(encoding='utf-8'))
            expected_catalog_count = len(catalog['active_models']) + sum(
                item.get('implementation_status') == 'available'
                for item in catalog['deferred_models']
            )
            assert report['model_count'] == expected_catalog_count, (
                report['model_count'], expected_catalog_count, str(catalog_path)
            )
            (evidence / 'qt-frozen-worker-integration.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
            print('PASS source Qt -> actual frozen single/batch workers', flush=True)
        finally:
            dialog.close()
            window.close()
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
