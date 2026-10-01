"""Console-subsystem entry retaining QProcess JSON pipes."""
import multiprocessing
import sys

if __name__ == "__main__":
    multiprocessing.freeze_support()
    for stream in (sys.stdout, sys.stderr):
        if stream is not None:
            stream.reconfigure(encoding="utf-8", errors="backslashreplace")
    from pyml_workbench._worker import main
    raise SystemExit(main())
