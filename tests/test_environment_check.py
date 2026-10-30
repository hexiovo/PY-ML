"""Release profiles must determine both the title and required dependencies."""
from contextlib import ExitStack
from email.message import Message
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from pyml_workbench import environment_check as checker


class ReleaseEnvironmentTests(unittest.TestCase):
    @unittest.skipUnless(checker.importlib.util.find_spec('umap'), 'manifold extra required')
    def test_umap_distribution_uses_its_importable_module_name(self):
        row = checker.check_dependency(checker.Requirement('umap-learn'), 'manifold', True)
        self.assertEqual(row['status'], 'ok', row)

    def report(self, profile, extras, *, missing=(), frozen=True):
        with tempfile.TemporaryDirectory() as temporary, ExitStack() as stack:
            root = Path(temporary)
            (root / 'build-info.json').write_text(json.dumps({
                'profile': profile, 'profile_extras': extras}), encoding='utf-8')
            (root / 'PYML-Worker.exe').touch()
            (root / '_internal').mkdir()
            package = root / 'pyml_workbench'
            (package / 'resources').mkdir(parents=True)
            (package / 'resources/overall-guide.md').touch()
            (package / 'model_catalog.json').touch()
            project = Message()
            project['Requires-Python'] = '>=3.12,<3.13'
            for extra in ('desktop', 'deep', 'sequence', 'search'):
                project['Provides-Extra'] = extra
            requirements = [
                'numpy>=1', 'PySide6>=1; extra == "desktop"',
                'torch>=1; extra == "deep"', 'skorch>=1; extra == "deep"',
                'hmmlearn>=1; extra == "sequence"', 'optuna>=1; extra == "search"',
            ]

            def dependency(requirement, group, required):
                absent = requirement.name in missing
                return dict(name=requirement.name, group=group, required=required,
                            status=('missing' if required else 'optional') if absent else 'ok')

            stack.enter_context(patch.object(checker.sys, 'frozen', frozen, create=True))
            stack.enter_context(patch.object(checker.sys, 'executable', str(root / 'PYML-Worker.exe')))
            stack.enter_context(patch.object(checker, '__file__', str(package / 'environment_check.py')))
            stack.enter_context(patch.object(checker.metadata, 'metadata', return_value=project))
            stack.enter_context(patch.object(checker.metadata, 'requires', return_value=requirements))
            stack.enter_context(patch.object(checker, 'check_dependency', side_effect=dependency))
            return checker.collect_environment(lambda event: None)

    def test_all_shipped_profiles_have_correct_titles_and_required_groups(self):
        profiles = json.loads((Path(__file__).resolve().parents[1] /
                               'packaging/profiles.json').read_text(encoding='utf-8'))['profiles']
        for name, profile in profiles.items():
            with self.subTest(profile=name):
                report = self.report(name, profile['extras'])
                self.assertEqual(report['mode'], 'EXE 环境 · ' + profile['title'])
                self.assertFalse([row for row in report['rows']
                                  if row['required'] and row['status'] != 'ok'])
                for row in report['rows']:
                    if row['group'] != 'runtime':
                        self.assertEqual(row['required'], row['group'] in
                                         {'base', 'desktop', *profile['extras']})

    def test_deep_missing_dependencies_remain_required_failures(self):
        for module in ('torch', 'skorch'):
            with self.subTest(module=module):
                report = self.report('deep', ['desktop', 'deep'], missing=(module,))
                failed = [row['name'] for row in report['rows']
                          if row['required'] and row['status'] != 'ok']
                self.assertEqual(failed, [module])
                self.assertIn('1 项必需组件', report['summary'])

    def test_deep_excluded_dependencies_remain_optional(self):
        report = self.report('deep', ['desktop', 'deep'], missing=('hmmlearn', 'optuna'))
        optional = [row['name'] for row in report['rows']
                    if not row['required'] and row['status'] == 'optional']
        self.assertEqual(optional, ['hmmlearn', 'optuna'])
        self.assertIn('所需组件检查通过', report['summary'])

    def test_invalid_profiles_and_extras_are_still_rejected(self):
        for profile, extras in (('unknown', []), ('deep', 'desktop'), ('deep', None)):
            with self.subTest(profile=profile, extras=extras):
                report = self.report(profile, extras)
                rows = [row for row in report['rows'] if row['name'] == '发行配置']
                self.assertEqual(len(rows), 1)
                self.assertEqual(rows[0]['status'], 'error')

    def test_python_mode_does_not_require_a_release_profile(self):
        report = self.report('unknown', None, frozen=False)
        self.assertEqual(report['mode'], 'Python 环境')
        self.assertFalse([row for row in report['rows'] if row['name'] == '发行配置'])


if __name__ == '__main__':
    unittest.main()
