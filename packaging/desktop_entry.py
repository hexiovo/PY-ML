"""Windowed PyInstaller entry point."""
import multiprocessing
import os
import sys


def _make_splash_optional(splash):
    """Contain optional splash IPC failures, including its registered atexit close."""
    if splash is None:
        return
    original_close, original_update = splash.close, splash.update_text
    closed = False

    def close():
        nonlocal closed
        if closed:
            return
        # Mark before sending: a failed send must not be retried by finally/atexit.
        closed = True
        try:
            original_close()
        except (OSError, RuntimeError):
            pass

    def update_text(message):
        if closed:
            return
        try:
            original_update(message)
        except (OSError, RuntimeError):
            close()

    splash.close, splash.update_text = close, update_text


if __name__ == "__main__":
    multiprocessing.freeze_support()
    # Libraries can legitimately print during imports; the GUI has no console.
    if sys.stdout is None:
        sys.stdout = open(os.devnull, "w", encoding="utf-8")
    if sys.stderr is None:
        sys.stderr = open(os.devnull, "w", encoding="utf-8")
    splash = sys.modules.get('pyi_splash')
    _make_splash_optional(splash)
    try:
        from pyml_workbench.gui import main
        if splash is not None:
            splash.update_text('正在准备工作台界面，请稍候…')
        raise SystemExit(main())
    finally:
        if splash is not None:
            splash.close()
