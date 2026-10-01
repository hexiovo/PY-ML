import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pyml_workbench.runtime import worker_command


class WorkerRoutingTests(unittest.TestCase):
    def test_python_mode_preserves_module_protocol(self):
        with patch.object(sys, 'frozen', False, create=True):
            program, args = worker_command(['train', '--config', '中文.json'])
        self.assertEqual(program, sys.executable)
        self.assertEqual(args, ['-X', 'utf8', '-m', 'pyml_workbench._worker', 'train', '--config', '中文.json'])

    def test_frozen_routes_to_sibling_without_python_flags(self):
        with tempfile.TemporaryDirectory(prefix='pyml-runtime-') as folder:
            worker = Path(folder) / 'PYML-Worker.exe'
            worker.touch()
            original = ['batch-run', '--job-id', 'id']
            with patch.object(sys, 'frozen', True, create=True), patch.object(sys, 'executable', str(Path(folder) / 'PYML-Workbench.exe')):
                program, args = worker_command(original)
            self.assertEqual(Path(program), worker)
            self.assertEqual(args, original)
            self.assertIsNot(args, original)

    def test_missing_frozen_worker_is_actionable(self):
        with tempfile.TemporaryDirectory(prefix='pyml-runtime-') as folder:
            with patch.object(sys, 'frozen', True, create=True), patch.object(sys, 'executable', str(Path(folder) / 'PYML-Workbench.exe')):
                with self.assertRaisesRegex(FileNotFoundError, '完整'):
                    worker_command(['train'])
