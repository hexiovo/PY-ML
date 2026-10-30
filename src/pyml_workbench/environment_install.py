"""Install registered missing dependencies without replacing a working generation."""
from __future__ import annotations

from contextlib import contextmanager, redirect_stdout
from importlib import metadata
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
from typing import Callable
import uuid

from packaging.requirements import Requirement
from packaging.utils import canonicalize_name

from .dependency_paths import active_packages, dependency_root, generation_packages


def installer_python() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).parent / "_internal/dependency-tools/python/python.exe"
    return Path(sys.executable)


def install_targets(report: dict) -> list[str]:
    """Only project-registered missing modules can become pip arguments."""
    missing = {canonicalize_name(row["name"]) for row in report.get("rows", ())
               if row.get("status") in {"missing", "optional"}
               and row.get("group") not in {"runtime"}}
    targets = []
    for raw in metadata.requires("pyml-workbench") or ():
        requirement = Requirement(raw)
        if canonicalize_name(requirement.name) not in missing or requirement.url:
            continue
        requirement.marker = None
        value = str(requirement)
        if value not in targets:
            targets.append(value)
    return targets


def install_support(report: dict) -> dict:
    targets = install_targets(report)
    python = installer_python()
    return dict(install_requirements=targets,
                install_supported=bool(targets and python.is_file()),
                install_reason=("安装报告中缺少的组件；联网下载，完成后自动复查。"
                                if targets and python.is_file() else
                                "安装工具缺失，请使用包含 dependency-tools 的完整发行包。"
                                if targets else "没有可安装的缺失组件。已有组件加载或版本异常需单独修复。"))


@contextmanager
def _install_lock(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    with (root / "install.lock").open("a+b") as stream:
        stream.seek(0, os.SEEK_END)
        if stream.tell() == 0:
            stream.write(b"0")
            stream.flush()
        stream.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError("另一个工作台正在安装组件，请等待完成后重试。") from exc
        try:
            yield
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def _run(command: list[str], emit: Callable, *, environment=None) -> str:
    flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    # Avoid leaking the frozen process's DLL search directory to normal Python.
    if os.name == "nt" and getattr(sys, "frozen", False):
        import ctypes
        ctypes.windll.kernel32.SetDllDirectoryW(None)
    try:
        process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, encoding="utf-8", errors="replace",
                                   env=environment, creationflags=flags)
    finally:
        if os.name == "nt" and getattr(sys, "frozen", False):
            ctypes.windll.kernel32.SetDllDirectoryW(getattr(sys, "_MEIPASS", None))
    output = []
    for line in process.stdout:
        output.append(line)
        emit({"type": "install_log", "message": line.rstrip()})
    code = process.wait()
    if code:
        raise RuntimeError(f"安装子进程退出码 {code}：\n{''.join(output)[-4000:]}")
    return "".join(output)


def _publish(root: Path, generation: str) -> None:
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=root, delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(json.dumps({"schema_version": 1, "generation": generation}).encode("utf-8"))
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, root / "current.json")
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def install_missing(emit: Callable) -> dict:
    from .environment_check import collect_environment
    from .runtime import worker_command

    emit({"type": "progress", "message": "正在重新核对待安装组件…"})
    report = collect_environment(lambda event: None)
    targets = install_targets(report)
    if not targets:
        return dict(installed=[], restart_required=False, message="没有需要安装的缺失组件。")
    python = installer_python()
    if not python.is_file():
        raise RuntimeError("安装工具缺失，请重新解压包含 dependency-tools 的完整发行包。")
    root = dependency_root()
    with _install_lock(root):
        generation = "gen-" + uuid.uuid4().hex
        packages = generation_packages(generation)
        packages.parent.mkdir()
        environment = dict(os.environ)
        for key in ("PYTHONPATH", "PYTHONHOME", "QT_PLUGIN_PATH", "QT_QPA_PLATFORM_PLUGIN_PATH"):
            environment.pop(key, None)
        try:
            old = active_packages()
            if old is not None:
                shutil.copytree(old, packages)
            else:
                packages.mkdir()
            constraints = packages.parent / "constraints.txt"
            pinned = {}
            for distribution in metadata.distributions():
                name = distribution.metadata.get("Name")
                if name and distribution.version:
                    pinned.setdefault(canonicalize_name(name), distribution.version)
            constraints.write_text("".join(f"{name}=={version}\n" for name, version in sorted(pinned.items())),
                                   encoding="utf-8")
            emit({"type": "progress", "message": "正在准备组件安装工具…"})
            tools = packages.parent / "installer"
            _run([str(python), "-I", "-m", "venv", str(tools)], emit, environment=environment)
            pip_python = tools / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
            emit({"type": "progress", "message": "正在联网安装：" + "、".join(targets)})
            _run([str(pip_python), "-I", "-m", "pip", "--isolated", "install",
                  "--disable-pip-version-check", "--no-input", "--progress-bar", "off",
                  "--only-binary=:all:", "--upgrade", "--target", str(packages),
                  "--constraint", str(constraints), *targets], emit, environment=environment)
            emit({"type": "progress", "message": "正在用工作台后台验证新组件…"})
            validation_environment = dict(os.environ, PYML_DEPENDENCY_GENERATION=generation)
            validation_environment["PYINSTALLER_RESET_ENVIRONMENT"] = "1"
            executable, arguments = worker_command(["environment-check"])
            output = _run([executable, *arguments], lambda event: None, environment=validation_environment)
            complete = None
            for line in output.splitlines():
                try:
                    event = json.loads(line)
                except ValueError:
                    continue  # Native libraries can write warnings to stderr.
                if isinstance(event, dict) and event.get("type") == "complete":
                    complete = event["report"]
            if complete is None:
                raise RuntimeError("新组件验证进程没有返回完整报告。")
            names = {canonicalize_name(Requirement(target).name) for target in targets}
            checked = {canonicalize_name(row["name"]): row["status"] for row in complete["rows"]}
            failed = [name for name in names if checked.get(name) != "ok"]
            if failed:
                details = [f"{row['name']}: {row.get('detail', row['status'])}"
                           for row in complete["rows"]
                           if canonicalize_name(row["name"]) in failed]
                raise RuntimeError("组件下载完成，但工作台加载验证未通过：" + "；".join(details))
            _publish(root, generation)
            return dict(installed=targets, restart_required=True,
                        message="组件安装完成，后台验证通过。正在重新检查；重启工作台后算法列表生效。")
        except BaseException:
            # Only remove this newly created generation, never the active one.
            if packages.parent.resolve().parent == root.resolve():
                shutil.rmtree(packages.parent, ignore_errors=True)
            raise


def main() -> int:
    output = sys.stdout

    def emit(event):
        print(json.dumps(event, ensure_ascii=False), file=output, flush=True)

    try:
        with redirect_stdout(sys.stderr):
            result = install_missing(emit)
        emit({"type": "install_complete", "result": result})
        return 0
    except Exception as exc:
        emit({"type": "failure", "message": f"安装失败：{type(exc).__name__}: {exc}"})
        return 1
