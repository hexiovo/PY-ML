"""Select a Python worker or its sibling frozen executable."""
from __future__ import annotations

from pathlib import Path
import sys


def worker_command(arguments: list[str]) -> tuple[str, list[str]]:
    if getattr(sys, "frozen", False):
        executable = Path(sys.executable).with_name("PYML-Worker.exe")
        if not executable.is_file():
            raise FileNotFoundError(
                f"后台程序缺失：{executable}。请解压并保留完整的程序文件夹。"
            )
        return str(executable), list(arguments)
    return sys.executable, ["-X", "utf8", "-m", "pyml_workbench._worker", *arguments]
