"""Installation publishes validated generations and preserves working packages on failure."""
from contextlib import ExitStack
import json
import importlib
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from pyml_workbench import dependency_paths as paths
from pyml_workbench import environment_install as installer


class DependencyInstallTests(unittest.TestCase):
    def test_complete_wheels_supply_submodules_without_replacing_bundled_packages(self):
        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
            root = Path(temporary)
            stack.enter_context(patch.dict(os.environ, {'PYML_DEPENDENCY_ROOT': str(root / 'managed')}))
            stack.enter_context(patch.object(sys, 'path', list(sys.path)))
            stack.enter_context(patch.object(sys, 'meta_path', list(sys.meta_path)))
            base = root / 'bundled'
            packages = paths.generation_packages('gen-test')
            packages.mkdir(parents=True)
            names = ('pyml_test_preloaded', 'pyml_test_future')
            try:
                for name in names:
                    (base / name).mkdir(parents=True)
                    (base / name / '__init__.py').write_text('origin = "bundled"', encoding='utf-8')
                    (packages / name).mkdir()
                    (packages / name / '__init__.py').write_text('origin = "wheel"', encoding='utf-8')
                    (packages / name / 'extra.py').write_text('value = 42', encoding='utf-8')
                sys.path.insert(0, str(base))
                importlib.import_module(names[0])
                installer._publish(paths.dependency_root(), 'gen-test')
                paths.activate_dependencies()
                for name in names:
                    self.assertEqual(importlib.import_module(name).origin, 'bundled')
                    self.assertEqual(importlib.import_module(name + '.extra').value, 42)
            finally:
                for name in tuple(sys.modules):
                    if name.startswith(names):
                        sys.modules.pop(name, None)

    def test_only_missing_registered_requirements_are_installable(self):
        report = {'rows': [
            {'name': 'torch', 'status': 'missing', 'group': 'deep'},
            {'name': 'catboost', 'status': 'optional', 'group': 'catboost'},
            {'name': 'numpy', 'status': 'version', 'group': 'base'},
            {'name': 'other', 'status': 'missing', 'group': 'base'},
            {'name': '后台程序', 'status': 'missing', 'group': 'runtime'},
        ]}
        requirements = ['torch==2.14.1; extra == "deep"', 'catboost==1.2.10',
                        'numpy==2.5.3', 'other @ https://example.invalid/package.whl']
        with patch.object(installer.metadata, 'requires', return_value=requirements):
            self.assertEqual(installer.install_targets(report), ['torch==2.14.1', 'catboost==1.2.10'])

    def test_frozen_installation_requires_the_bundled_helper(self):
        with patch.object(sys, 'frozen', True, create=True), \
                patch.object(sys, 'executable', 'C:/app/PYML-Worker.exe'):
            self.assertEqual(installer.installer_python(),
                             Path('C:/app/_internal/dependency-tools/python/python.exe'))

    def test_invalid_or_traversing_active_generation_is_ignored(self):
        with tempfile.TemporaryDirectory() as temporary, \
                patch.dict(os.environ, {'PYML_DEPENDENCY_ROOT': temporary}):
            for state in ([], {'schema_version': 9},
                          {'schema_version': 1, 'generation': 'gen-../../outside'}):
                with self.subTest(state=state):
                    (Path(temporary) / 'current.json').write_text(json.dumps(state), encoding='utf-8')
                    self.assertIsNone(paths.active_packages())

    def test_install_lock_prevents_concurrent_installation(self):
        with tempfile.TemporaryDirectory() as temporary:
            with installer._install_lock(Path(temporary)):
                with self.assertRaises(RuntimeError):
                    with installer._install_lock(Path(temporary)):
                        self.fail('Second installer acquired the lock')
            with installer._install_lock(Path(temporary)):
                pass

    def transaction(self, *, failure=None, validation_status='ok'):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(patch.dict(os.environ, {'PYML_DEPENDENCY_ROOT': temporary.name}))
        root = Path(temporary.name)
        old = paths.generation_packages('gen-old')
        old.mkdir(parents=True)
        (old / 'existing.txt').write_text('previous package', encoding='utf-8')
        installer._publish(root, 'gen-old')
        before = (root / 'current.json').read_bytes()
        report = {'rows': [{'name': 'demo', 'group': 'search', 'status': 'optional'}]}
        stack.enter_context(patch('pyml_workbench.environment_check.collect_environment', return_value=report))
        stack.enter_context(patch.object(installer.metadata, 'requires', return_value=['demo==1.0']))
        stack.enter_context(patch.object(installer.metadata, 'distributions', return_value=[]))
        stack.enter_context(patch.object(installer, 'installer_python', return_value=Path(sys.executable)))
        stack.enter_context(patch('pyml_workbench.runtime.worker_command', return_value=('worker', ['environment-check'])))
        commands = []

        def run(command, emit, *, environment=None):
            commands.append(command)
            if len(commands) == 2:
                if failure:
                    raise RuntimeError(failure)
                target = Path(command[command.index('--target') + 1])
                (target / 'new.txt').write_text('downloaded package', encoding='utf-8')
            return json.dumps({'type': 'complete', 'report': {
                'rows': [{'name': 'demo', 'status': validation_status}]}}) if command[0] == 'worker' else ''

        stack.enter_context(patch.object(installer, '_run', side_effect=run))
        return root, old, before, commands

    def test_success_publishes_only_after_validation_and_preserves_old_packages(self):
        root, old, before, commands = self.transaction()
        result = installer.install_missing(lambda event: None)
        current = paths.active_packages()
        self.assertNotEqual(current, old)
        self.assertEqual((current / 'existing.txt').read_text(), 'previous package')
        self.assertEqual((current / 'new.txt').read_text(), 'downloaded package')
        self.assertEqual((old / 'existing.txt').read_text(), 'previous package')
        self.assertTrue(result['restart_required'])
        self.assertEqual(commands[-1], ['worker', 'environment-check'])
        self.assertEqual(commands[0][1:4], ['-I', '-m', 'venv'])
        self.assertIn('--only-binary=:all:', commands[1])
        self.assertIn('demo==1.0', commands[1])

    def test_download_failure_preserves_active_generation(self):
        root, old, before, commands = self.transaction(failure='offline')
        with self.assertRaisesRegex(RuntimeError, 'offline'):
            installer.install_missing(lambda event: None)
        self.assertEqual((root / 'current.json').read_bytes(), before)
        self.assertEqual(paths.active_packages(), old)
        self.assertEqual(list(root.glob('gen-*')), [old.parent])

    def test_import_failure_preserves_active_generation(self):
        root, old, before, commands = self.transaction(validation_status='error')
        with self.assertRaisesRegex(RuntimeError, '加载验证未通过'):
            installer.install_missing(lambda event: None)
        self.assertEqual((root / 'current.json').read_bytes(), before)
        self.assertEqual(list(root.glob('gen-*')), [old.parent])


if __name__ == '__main__':
    unittest.main()
