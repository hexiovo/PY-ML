"""Use the checkout's project version for builds and artifact validation."""
from importlib import metadata
from pathlib import Path
import tomllib


def source_version(root: Path) -> str:
    return tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]


def require_build_version(root: Path) -> str:
    version = source_version(root)
    try:
        installed = metadata.version("pyml-workbench")
    except metadata.PackageNotFoundError:
        installed = "未安装"
    if installed != version:
        raise SystemExit(f"构建需要当前源码 {version} 对应的已安装 wheel，当前环境为 {installed}。")
    return version
