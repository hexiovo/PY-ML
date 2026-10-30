"""On-demand dependency checks, isolated from the desktop's training worker."""
from __future__ import annotations

from contextlib import redirect_stdout
import importlib
from importlib import metadata
import importlib.util
import json
from pathlib import Path
import platform
import struct
import sys
from typing import Any, Callable

from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet


GROUP_LABELS = {
    "base": "基础功能", "desktop": "桌面 / 图表 / XLS",
    "search": "TPE / 遗传搜索", "sequence": "HMM 序列模型",
    "deep": "神经网络 / LSTM / GRU", "xgboost": "XGBoost",
    "lightgbm": "LightGBM", "catboost": "CatBoost",
}
MODULE_NAMES = {"scikit-learn": "sklearn", "pyside6": "PySide6", "umap-learn": "umap"}
STATUS_LABELS = {
    "ok": "正常", "missing": "缺少组件", "error": "加载失败",
    "version": "版本不符", "optional": "可选未安装",
}
PROFILE_TITLES = {
    "standard": "标准版", "deep": "标准深度版", "full": "完整扩展版",
}


def check_dependency(requirement: Requirement, group: str, required: bool) -> dict[str, Any]:
    name = requirement.name
    module_name = MODULE_NAMES.get(name.casefold(), name.replace("-", "_"))
    row = dict(name=name, group=group, required=required,
               expected=str(requirement.specifier) or "可加载", installed="—",
               status="ok", detail="组件可正常加载。")
    try:
        present = importlib.util.find_spec(module_name) is not None
        try:
            row["installed"] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pass
        if not present:
            row.update(status="missing" if required else "optional",
                       detail="未找到模块；此功能不可用。" if required else "未安装，仅影响此可选功能。")
            return row
        # Match the desktop's core-before-Qt/dateutil import order.
        if module_name == "PySide6":
            importlib.import_module("dateutil.rrule")
            module = importlib.import_module("PySide6.QtWidgets")
            version_module = importlib.import_module("PySide6")
        else:
            module = importlib.import_module(module_name)
            version_module = module
        if row["installed"] == "—":
            row["installed"] = str(getattr(version_module, "__version__", "未知"))
        if row["installed"] == "未知":
            row.update(status="version", detail="可以加载，但无法核对版本。")
        elif not requirement.specifier.contains(row["installed"], prereleases=True):
            row.update(status="version", detail="可以加载，但版本与项目锁定要求不一致。")
        if module_name == "matplotlib":
            importlib.import_module("dateutil.rrule")
            importlib.import_module("matplotlib.backends.backend_qtagg")
        elif module_name == "pymoo":
            importlib.import_module("pymoo.algorithms.soo.nonconvex.ga")
    except Exception as exc:
        row.update(status="error", detail=f"{type(exc).__name__}: {exc}")
    return row


def collect_environment(emit: Callable[[dict[str, Any]], None]) -> dict[str, Any]:
    """Emit progress and return a report using installed project requirements."""
    rows: list[dict[str, Any]] = []
    frozen = bool(getattr(sys, "frozen", False))
    required_groups = {"base", "desktop"}
    mode = "Python 环境"

    def add(row: dict[str, Any]) -> None:
        rows.append(row)
        emit({"type": "row", "row": row})

    def runtime_row(name: str, expected: str, installed: str, valid: bool, detail: str = "") -> None:
        add(dict(name=name, group="runtime", required=True, expected=expected,
                 installed=installed, status="ok" if valid else "error", detail=detail))

    if frozen:
        mode = "EXE 环境"
        folder = Path(sys.executable).parent
        try:
            build = json.loads((folder / "build-info.json").read_text(encoding="utf-8"))
            profile = build["profile"]
            if profile not in PROFILE_TITLES or not isinstance(build["profile_extras"], list):
                raise ValueError("发行配置无效")
            required_groups.update(build["profile_extras"])
            mode += " · " + PROFILE_TITLES[profile]
        except (OSError, ValueError, KeyError, TypeError) as exc:
            runtime_row("发行配置", "build-info.json", "无法读取", False, str(exc))
        runtime_row("后台程序", "PYML-Worker.exe", str(folder / "PYML-Worker.exe"),
                    (folder / "PYML-Worker.exe").is_file(), "缺失时请重新完整解压发行包。")
        runtime_row("运行库目录", "_internal", str(folder / "_internal"),
                    (folder / "_internal").is_dir(), "请保留整个程序目录。")

    emit({"type": "progress", "message": "正在核对 Python 与项目依赖清单…"})
    project = metadata.metadata("pyml-workbench")
    python_spec = project.get("Requires-Python") or ">=3.12,<3.13"
    runtime_row("Python", python_spec, platform.python_version(),
                SpecifierSet(python_spec).contains(platform.python_version()),
                "EXE 已内置 Python，无需单独安装。" if frozen else str(sys.executable))
    bits = struct.calcsize("P") * 8
    runtime_row("运行架构", "64 位", f"{bits} 位", bits == 64, platform.platform())
    package_root = Path(__file__).parent
    guide = package_root / "resources/overall-guide.md"
    if not guide.is_file() and not frozen:
        guide = package_root.parents[1] / "docs/overall-guide.md"
    runtime_row("操作指南", "离线指南文件", "存在" if guide.is_file() else "缺失", guide.is_file())
    catalog = package_root / "model_catalog.json"
    runtime_row("模型目录", "model_catalog.json", "存在" if catalog.is_file() else "缺失", catalog.is_file())
    extras = list(project.get_all("Provides-Extra") or [])
    requirements = metadata.requires("pyml-workbench")
    if not requirements:
        raise ValueError("项目安装信息缺少依赖清单，请重新安装项目或完整解压发行包。")
    for raw in requirements:
        requirement = Requirement(raw)
        if requirement.marker is None or requirement.marker.evaluate({"extra": ""}):
            group = "base"
        else:
            group = next((extra for extra in extras
                          if requirement.marker.evaluate({"extra": extra})), None)
            if group is None:
                continue
        emit({"type": "progress", "message": f"正在检查 {requirement.name}…"})
        add(check_dependency(requirement, group, group in required_groups))

    failures = [row for row in rows if row["required"] and row["status"] != "ok"]
    optional_issues = [row for row in rows if not row["required"] and row["status"] != "ok"]
    summary = (f"当前环境所需组件检查通过；{len(optional_issues)} 项可选组件未安装或需处理。"
               if not failures else f"当前环境不完整：{len(failures)} 项必需组件需处理。")
    affected_groups = sorted({row["group"] for row in rows
                              if row["status"] != "ok" and row["group"] not in {"base", "runtime"}})
    advice = ["检查涵盖组件版本、实际导入和必要资源；不执行模型训练。"]
    if frozen:
        advice.append("必需组件异常：重新解压完整发行包，保留主程序、后台程序和 _internal 目录。")
        if optional_issues:
            advice.append("缺少的组件可点击“一键安装”联网补齐，安装后自动复查；重启工作台后用于训练。可选组件不影响已通过的基础功能。")
    else:
        advice.append("在项目根目录同步当前环境后重启程序：")
        advice.append("uv sync --locked --no-editable --no-dev --extra desktop")
        if affected_groups:
            advice.append("如需补齐相关可选功能，在上述命令后追加：" + " ".join(f"--extra {group}" for group in affected_groups))
    report = dict(mode=mode, rows=rows, summary=summary, advice=advice, complete=True)
    from .environment_install import install_support

    report.update(install_support(report))
    return report


def main() -> int:
    output = sys.stdout

    def emit(event: dict[str, Any]) -> None:
        print(json.dumps(event, ensure_ascii=False), file=output, flush=True)

    try:
        # Native libraries may print during imports; keep the JSON pipe clean.
        with redirect_stdout(sys.stderr):
            report = collect_environment(emit)
        emit({"type": "complete", "report": report})
        return 0
    except Exception as exc:
        emit({"type": "failure", "message": f"检查未完成：{type(exc).__name__}: {exc}"})
        return 1
