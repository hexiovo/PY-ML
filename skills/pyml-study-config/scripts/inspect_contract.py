#!/usr/bin/env python3
"""Export the live PYML source contract without fitting or changing project files."""
from __future__ import annotations

import argparse
from dataclasses import MISSING, fields, is_dataclass
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any


SOURCE_FILES = (
    "src/pyml_workbench/config.py",
    "src/pyml_workbench/data.py",
    "src/pyml_workbench/preprocessing.py",
    "src/pyml_workbench/validation.py",
    "src/pyml_workbench/cross_validation.py",
    "src/pyml_workbench/sequence.py",
    "src/pyml_workbench/objectives.py",
    "src/pyml_workbench/search.py",
    "src/pyml_workbench/search_space.py",
    "src/pyml_workbench/batch.py",
    "src/pyml_workbench/_worker.py",
    "src/pyml_workbench/__init__.py",
    "src/pyml_workbench/catalog.py",
    "src/pyml_workbench/parameters.py",
    "src/pyml_workbench/model_catalog.json",
    "src/pyml_workbench/metrics.py",
    "src/pyml_workbench/experiment.py",
    "src/pyml_workbench/extended_experiment.py",
    "src/pyml_workbench/gui.py",
    "src/pyml_workbench/batch_gui.py",
    "pyproject.toml",
)


def dataclass_contract(cls: type) -> dict[str, Any]:
    if not is_dataclass(cls):
        raise TypeError(f"{cls.__name__} is not a dataclass")
    result = []
    for item in fields(cls):
        if item.default is not MISSING:
            default = repr(item.default)
        elif item.default_factory is not MISSING:
            default = f"default_factory:{item.default_factory.__qualname__}"
        else:
            default = "required"
        result.append({
            "name": item.name,
            "type": str(item.type),
            "default": default,
        })
    return {"class": f"{cls.__module__}.{cls.__qualname__}", "fields": result}


def json_safe(value: Any) -> Any:
    if isinstance(value, float) and not math.isfinite(value):
        if math.isnan(value):
            return "NaN"
        return "Infinity" if value > 0 else "-Infinity"
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    return value

def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read the chosen PYML checkout and export its current contracts."
    )
    parser.add_argument(
        "--project", required=True,
        help="Absolute or relative path to the PYML project checkout.",
    )
    parser.add_argument(
        "--output",
        help="Optional new JSON output file. Existing files are never overwritten.",
    )
    parser.add_argument(
        "--portable",
        action="store_true",
        help="Keep project and module paths relative in the snapshot; still verify the live import path.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project = Path(args.project).expanduser().resolve()
    source_root = project / "src"
    expected_package = source_root / "pyml_workbench" / "__init__.py"
    if not expected_package.is_file():
        raise FileNotFoundError(f"PYML source package not found: {expected_package}")
    sys.path.insert(0, str(source_root))

    import pyml_workbench
    from pyml_workbench.config import (
        DatasetConfig,
        EvaluationConfig,
        ExperimentConfig,
        SplitConfig,
    )
    from pyml_workbench.objectives import ObjectiveSpec
    from pyml_workbench.parameters import parameter_schema
    from pyml_workbench.search import SearchSpec
    from pyml_workbench.search_space import SearchSpace
    from pyml_workbench.sequence import SequenceConfig

    imported_module = Path(pyml_workbench.__file__).resolve()
    if imported_module != expected_package.resolve():
        raise RuntimeError(
            "Imported PYML package is not the selected project source: "
            f"expected {expected_package.resolve()}, got {imported_module}"
        )

    classes = (
        DatasetConfig,
        SplitConfig,
        EvaluationConfig,
        ExperimentConfig,
        SequenceConfig,
        ObjectiveSpec,
        SearchSpec,
        SearchSpace,
    )
    models = pyml_workbench.list_models(include_deferred=True)
    model_contracts = []
    for model in models:
        model_id = model["id"]
        item = {
            "id": model_id,
            "task": model.get("task"),
            "display_name_zh": model.get("display_name_zh"),
            "implementation_status": model.get("implementation_status", "active"),
            "runtime_available": model.get("runtime_available", True),
            "optional_extra": model.get("optional_extra"),
            "unavailable_reason": model.get("unavailable_reason"),
        }
        if item["implementation_status"] not in {"active", "available"}:
            item["parameter_schema_status"] = "not_implemented"
        else:
            try:
                item["parameter_schema"] = json_safe(parameter_schema(model_id))
                item["parameter_schema_status"] = "available"
            except Exception as exc:
                item["parameter_schema_status"] = "not_verified"
                item["parameter_schema_error"] = f"{type(exc).__name__}: {exc}"
        model_contracts.append(item)

    source_hashes = {}
    for relative in SOURCE_FILES:
        path = project / relative
        if path.is_file():
            source_hashes[relative] = {
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
        else:
            source_hashes[relative] = {"status": "missing"}

    payload = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": "." if args.portable else str(project),
        "python_executable": "not captured in portable snapshot" if args.portable else str(Path(sys.executable).resolve()),
        "python_version": sys.version,
        "package_version": pyml_workbench.__version__,
        "module_file": "src/pyml_workbench/__init__.py" if args.portable else str(imported_module),
        "portable_paths": args.portable,
        "source_package_verified": True,
        "snapshot_warning": (
            "This export describes the selected checkout and runtime. "
            "It is a drafting aid; it does not validate a study, dataset, "
            "external preprocessing, inferential statistics, or scientific claims."
        ),
        "dataclasses": {
            cls.__name__: dataclass_contract(cls) for cls in classes
        },
        "models": model_contracts,
        "source_hashes": source_hashes,
    }
    encoded = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output:
        output = Path(args.output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(encoded)
        print(f"Wrote {output}")
    else:
        sys.stdout.write(encoded)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"NOT VERIFIED: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(2)
