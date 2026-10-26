"""Preserve the application's existing core-before-Qt import contract."""
import os
import sys

if os.name == 'nt' and os.path.basename(sys.executable).casefold() == 'pyml-workbench.exe':
    import ctypes
    ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID('hexiovo.PYML.Workbench')

# The bootloader has already displayed the splash before any Python imports.
# Only the GUI owns it; the worker must never connect to its parent's splash.
if os.path.basename(sys.executable).casefold() == 'pyml-workbench.exe' and os.environ.get('_PYI_SPLASH_IPC', '0') != '0':
    import pyi_splash
    pyi_splash.update_text('正在加载机器学习组件，请稍候…')

import pyml_workbench
