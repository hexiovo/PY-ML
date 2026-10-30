"""Collect a clean CPython installer runtime without the build environment's packages."""
from pathlib import Path
import sys


def installer_runtime_datas(prefix=None):
    root = Path(prefix or sys.base_prefix)
    for required in ("python.exe", "python312.dll", "Lib/ensurepip/_bundled"):
        if not (root / required).exists():
            raise RuntimeError(f"The installer requires a complete CPython 3.12 runtime: {root / required}")
    ignored = {"site-packages", "__pycache__", "test", "tests", "idlelib", "tkinter", "turtledemo"}
    entries = []
    for folder in ("Lib", "DLLs"):
        for path in (root / folder).rglob("*"):
            relative = path.relative_to(root)
            if path.is_file() and not ignored.intersection(relative.parts):
                entries.append((str(path), str(Path("dependency-tools/python") / relative.parent)))
    for pattern in ("python*.exe", "python*.dll", "vcruntime*.dll", "LICENSE*", "BUILD"):
        for path in root.glob(pattern):
            if path.is_file():
                entries.append((str(path), "dependency-tools/python"))
    return entries
