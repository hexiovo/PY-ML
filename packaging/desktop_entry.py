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
    from pyml_workbench.gui import main
    raise SystemExit(main())
