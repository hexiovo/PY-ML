"""Startup IPC failures must not abort the application or mask its errors."""
from pathlib import Path
import runpy
import sys
from types import ModuleType
import unittest
from unittest.mock import Mock, patch

ENTRY = Path(__file__).resolve().parents[1] / 'packaging/desktop_entry.py'


class StartupSplashTests(unittest.TestCase):
    def guard(self, splash):
        runpy.run_path(str(ENTRY))['_make_splash_optional'](splash)

    def test_broken_close_is_attempted_once_even_during_final_cleanup(self):
        splash = ModuleType('pyi_splash')
        raw_close = Mock(side_effect=ConnectionAbortedError(10053, 'aborted'))
        splash.close = raw_close
        splash.update_text = Mock()
        self.guard(splash)
        splash.close()
        splash.close()  # GUI, entry finally, and PyInstaller atexit share this wrapper.
        splash.close()
        splash.update_text('too late')
        raw_close.assert_called_once()

    def test_update_failure_closes_splash_and_allows_main_to_run(self):
        splash = ModuleType('pyi_splash')
        splash.close = Mock(side_effect=OSError('disconnected'))
        splash.update_text = Mock(side_effect=ConnectionError('disconnected'))
        gui = ModuleType('pyml_workbench.gui')
        gui.main = Mock(return_value=0)
        with patch.dict(sys.modules, {'pyi_splash': splash, 'pyml_workbench.gui': gui}):
            with self.assertRaises(SystemExit) as result:
                runpy.run_path(str(ENTRY), run_name='__main__')
        self.assertEqual(result.exception.code, 0)
        gui.main.assert_called_once()

    def test_splash_failure_does_not_replace_real_application_error(self):
        splash = ModuleType('pyi_splash')
        splash.close = Mock(side_effect=ConnectionAbortedError('aborted'))
        splash.update_text = Mock()
        gui = ModuleType('pyml_workbench.gui')
        gui.main = Mock(side_effect=ValueError('application failure'))
        with patch.dict(sys.modules, {'pyi_splash': splash, 'pyml_workbench.gui': gui}):
            with self.assertRaisesRegex(ValueError, 'application failure'):
                runpy.run_path(str(ENTRY), run_name='__main__')

    def test_normal_close_and_update_still_work(self):
        splash = ModuleType('pyi_splash')
        raw_close = Mock()
        raw_update = Mock()
        splash.close, splash.update_text = raw_close, raw_update
        self.guard(splash)
        splash.update_text('ready')
        splash.close()
        splash.close()
        raw_update.assert_called_once_with('ready')
        raw_close.assert_called_once()

    def test_missing_splash_is_supported(self):
        self.guard(None)

    def test_uninitialized_splash_is_supported(self):
        splash = ModuleType('pyi_splash')
        splash.close = Mock()
        splash.update_text = Mock(side_effect=RuntimeError('not initialized'))
        self.guard(splash)
        splash.update_text('ready')
        splash.close()


if __name__ == '__main__':
    unittest.main()
