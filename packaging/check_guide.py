"""Verify installed guide provenance and render narrow-window UI previews."""
import hashlib
import importlib.metadata as metadata
from importlib.resources import files
import json
import os
from pathlib import Path
import tempfile

os.environ['QT_QPA_PLATFORM'] = 'offscreen'

from pyml_workbench.gui import WorkbenchWindow, _configure_application_font
from PySide6.QtWidgets import QApplication


def main():
    root = Path(__file__).resolve().parents[1]
    version = metadata.version('pyml-workbench')
    evidence = root / f'delivery/exe-{version}'
    evidence.mkdir(parents=True, exist_ok=True)
    canonical = (root / 'docs/overall-guide.md').read_bytes()
    installed = files('pyml_workbench').joinpath('resources', 'overall-guide.md').read_bytes()
    assert canonical == installed
    app = QApplication.instance() or QApplication([])
    _configure_application_font(app)
    with tempfile.TemporaryDirectory(prefix='PYML-guide-preview-') as temporary:
        window = WorkbenchWindow(preferences_path=Path(temporary) / 'preferences.json')
        try:
            window.resize(1060, 720)
            window.show()
            app.processEvents()
            button = window.overall_guide_button
            assert window.centralWidget().rect().contains(button.geometry())
            assert button.isVisible() and button.isEnabled()
            assert window.grab().save(str(evidence / 'main-guide-button.png'))
            button.click()
            app.processEvents()
            dialog = window._guide_dialog
            assert dialog.chapters.count() == 13
            assert dialog.grab().save(str(evidence / 'guide-preview.png'))
            report = {
                'overall': 'PASS', 'version': version,
                'guide_sha256': hashlib.sha256(canonical).hexdigest(),
                'installed_resource_matches_document': True,
                'minimum_main_window': [window.width(), window.height()],
                'guide_button_visible_and_inside_window': True,
                'chapter_count': dialog.chapters.count(),
                'scope': 'non-editable installed package Qt offscreen rendering',
            }
            (evidence / 'guide-resource-and-layout.json').write_text(
                json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8'
            )
            print(json.dumps(report, ensure_ascii=False), flush=True)
        finally:
            window.close()


if __name__ == '__main__':
    main()
