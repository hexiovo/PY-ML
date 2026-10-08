"""Windowed PyInstaller entry point."""
import multiprocessing
import os
import sys

if __name__ == "__main__":
    multiprocessing.freeze_support()
    # Libraries can legitimately print during imports; the GUI has no console.
    if sys.stdout is None:
        sys.stdout = open(os.devnull, "w", encoding="utf-8")
    if sys.stderr is None:
        sys.stderr = open(os.devnull, "w", encoding="utf-8")
    splash = sys.modules.get('pyi_splash')
    try:
        from pyml_workbench.gui import main
        if splash is not None:
            splash.update_text('正在准备工作台界面，请稍候…')
        raise SystemExit(main())
    finally:
        if splash is not None:
            splash.close()
