"""Activate only complete, application-owned dependency generations."""
from __future__ import annotations

import importlib
from importlib.machinery import PathFinder
import json
import os
from pathlib import Path
import sys

from .preferences import default_application_data_directory


class _DependencyPackageFinder:
    """Allow missing submodules of partially frozen packages to use full wheels."""

    def __init__(self, packages: Path):
        self.packages = packages

    def find_spec(self, fullname, path=None, target=None):
        fallback = self.packages.joinpath(*fullname.split("."))
        if not fallback.is_dir():
            return None
        spec = PathFinder.find_spec(fullname, path, target)
        if spec is not None and spec.submodule_search_locations is not None:
            locations = list(spec.submodule_search_locations)
            if str(fallback) not in locations:
                locations.append(str(fallback))
            spec.submodule_search_locations = locations
            return spec
        return None


def dependency_root() -> Path:
    override = os.environ.get("PYML_DEPENDENCY_ROOT")
    return Path(override).resolve() if override else (
        default_application_data_directory() / "dependencies" / sys.implementation.cache_tag
    ).resolve()


def generation_packages(generation: str) -> Path:
    if not isinstance(generation, str) or not generation.startswith("gen-"):
        raise ValueError("依赖环境标识无效。")
    root = dependency_root()
    directory = (root / generation).resolve()
    if directory.parent != root:
        raise ValueError("依赖目录超出工作台管理范围。")
    packages = (directory / "packages").resolve()
    if packages.parent != directory:
        raise ValueError("依赖目录超出工作台管理范围。")
    return packages


def active_packages() -> Path | None:
    try:
        candidate = os.environ.get("PYML_DEPENDENCY_GENERATION")
        if candidate:
            packages = generation_packages(candidate)
        else:
            raw = (dependency_root() / "current.json").read_text(encoding="utf-8")
            state = json.loads(raw)
            if not isinstance(state, dict) or state.get("schema_version") != 1:
                return None
            packages = generation_packages(state["generation"])
        return packages if packages.is_dir() else None
    except (OSError, ValueError, KeyError, TypeError):
        return None


def activate_dependencies() -> Path | None:
    packages = active_packages()
    if packages is not None and str(packages) not in sys.path:
        # Bundled core libraries keep precedence over downloaded optional SDKs.
        sys.path.append(str(packages))
        for name, module in tuple(sys.modules.items()):
            if module is None:
                continue
            namespace = vars(module)
            locations = namespace.get("__path__")
            fallback = packages.joinpath(*name.split("."))
            if locations is not None and fallback.is_dir() and str(fallback) not in locations:
                module.__path__ = [*locations, str(fallback)]
        # Keep existing meta finders, frozen loaders and bundled modules first;
        # only add a fallback location to package specs found by PathFinder.
        index = next((i for i, finder in enumerate(sys.meta_path) if finder is PathFinder), len(sys.meta_path))
        sys.meta_path.insert(index, _DependencyPackageFinder(packages))
        importlib.invalidate_caches()
    return packages
