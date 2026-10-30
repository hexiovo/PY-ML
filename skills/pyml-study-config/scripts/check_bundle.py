#!/usr/bin/env python3
"""Strictly validate native PYML JSON and report what was actually checked.

This checker never fits a model, reads test metrics, or grants execution readiness.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
from typing import Any


CONFIG_KEYS = {
    "dataset", "task", "model_id", "parameters", "split",
    "output_dir", "sequence", "evaluation", "pca_components",
    "reduction_method", "reduction_components",
}
DATASET_KEYS = {
    "source_path", "sheet_name", "target_column", "feature_columns",
    "reference_labels_column",
}
SPLIT_KEYS = {
    "seed", "train_fraction", "validation_fraction", "test_fraction",
    "strategy", "group_column", "time_column", "n_splits",
}
EVALUATION_KEYS = {"folds", "strategy", "gap"}
SEQUENCE_KEYS = {
    "group_column", "time_column", "order_mode", "observation_columns",
    "window", "horizon",
}
SEARCH_KEYS = {
    "method", "space", "objective", "max_fits", "timeout_seconds",
    "max_proposals", "seed",
}
SPACE_KEYS = {"fields", "fixed"}
FIELD_KEYS = {"type", "low", "high", "values", "log", "when"}
OBJECTIVE_KEYS = {
    "metric", "direction", "split", "objective_labels_column",
    "label_mapping", "coverage_min",
}


class BundleError(ValueError):
    """Raised for malformed or unsupported bundle input."""


def reject_constant(value: str):
    raise BundleError(f"Non-finite JSON number is not allowed: {value}")


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise BundleError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def reject_nonfinite(value: Any, path: str = "$") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise BundleError(f"Non-finite JSON number at {path}")
    if isinstance(value, dict):
        for key, item in value.items():
            reject_nonfinite(item, f"{path}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            reject_nonfinite(item, f"{path}[{index}]")


def read_json(path: Path) -> Any:
    value = json.loads(
        path.read_text(encoding="utf-8-sig"),
        object_pairs_hook=unique_object,
        parse_constant=reject_constant,
    )
    reject_nonfinite(value)
    return value


def check_keys(value: Any, label: str, allowed: set[str], required=()):
    if not isinstance(value, dict):
        raise BundleError(f"{label} must be a JSON object")
    unknown = set(value) - allowed
    missing = set(required) - set(value)
    if unknown:
        raise BundleError(
            f"Unknown {label} field(s): {', '.join(sorted(unknown))}. "
            "Keep non-native preprocessing/statistics plans in their own files."
        )
    if missing:
        raise BundleError(f"Missing {label} field(s): {', '.join(sorted(missing))}")


def check_string_list(value: Any, label: str) -> None:
    if not isinstance(value, list) or any(not isinstance(item, str) or not item for item in value):
        raise BundleError(f"{label} must be a list of non-empty strings")


def check_config_shape(payload: Any) -> None:
    check_keys(payload, "experiment config", CONFIG_KEYS, {"dataset", "task", "model_id"})
    dataset = payload["dataset"]
    check_keys(dataset, "dataset", DATASET_KEYS, {"source_path"})
    if not isinstance(dataset["source_path"], str) or not dataset["source_path"].strip():
        raise BundleError("dataset.source_path must be a non-empty string")
    if "feature_columns" in dataset and dataset["feature_columns"] is not None:
        check_string_list(dataset["feature_columns"], "dataset.feature_columns")
    for key in ("sheet_name", "target_column", "reference_labels_column"):
        if key in dataset and dataset[key] is not None and not isinstance(dataset[key], (str, int)):
            raise BundleError(f"dataset.{key} must be a string, integer, or null")
    if "parameters" in payload and not isinstance(payload["parameters"], dict):
        raise BundleError("parameters must be a JSON object")
    if "output_dir" in payload and payload["output_dir"] is not None and not isinstance(payload["output_dir"], str):
        raise BundleError("output_dir must be a string or null")
    if "split" in payload:
        check_keys(payload["split"], "split", SPLIT_KEYS)
    if "evaluation" in payload and payload["evaluation"] is not None:
        check_keys(payload["evaluation"], "evaluation", EVALUATION_KEYS)
    if "sequence" in payload and payload["sequence"] is not None:
        check_keys(payload["sequence"], "sequence", SEQUENCE_KEYS)
        sequence = payload["sequence"]
        if "observation_columns" in sequence and sequence["observation_columns"] is not None:
            check_string_list(sequence["observation_columns"], "sequence.observation_columns")


def check_search_shape(payload: Any) -> None:
    check_keys(payload, "search spec", SEARCH_KEYS)
    if "space" in payload:
        space = payload["space"]
        check_keys(space, "search space", SPACE_KEYS)
        if "fields" in space:
            if not isinstance(space["fields"], dict):
                raise BundleError("search space fields must be a JSON object")
            for name, description in space["fields"].items():
                check_keys(description, f"search field {name}", FIELD_KEYS, {"type"})
                if "when" in description and not isinstance(description["when"], dict):
                    raise BundleError(f"search field {name}.when must be a JSON object")
        if "fixed" in space and not isinstance(space["fixed"], dict):
            raise BundleError("search space fixed must be a JSON object")
    if "objective" in payload and payload["objective"] is not None:
        check_keys(payload["objective"], "objective", OBJECTIVE_KEYS)
    if "seed" in payload and isinstance(payload["seed"], bool):
        raise BundleError("search seed must be an integer, not a boolean")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Check strict JSON shape, live PYML API parsing, and an optional source header."
    )
    parser.add_argument("--project", required=True, help="PYML project checkout.")
    parser.add_argument("--config", required=True, help="Native ExperimentConfig JSON.")
    parser.add_argument("--search", help="Optional native SearchSpec JSON.")
    parser.add_argument("--output", help="Optional new JSON report path; never overwritten.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    project = Path(args.project).expanduser().resolve()
    config_path = Path(args.config).expanduser().resolve()
    report = {
        "schema_version": 1,
        "project_root": str(project),
        "config_file": str(config_path),
        "checks": {
            "strict_json": "not_run",
            "native_config_shape": "not_run",
            "project_source_import": "not_run",
            "live_api_parse": "not_run",
            "model_parameters": "not_run",
            "search_json": "not_requested",
            "search_shape": "not_requested",
            "search_api_parse": "not_requested",
            "source_header_preview": "not_run",
            "full_data_values": "not_checked",
            "model_fit": "not_run",
            "final_test": "not_read",
            "external_preprocessing": "not_checked",
            "inferential_statistics": "not_checked",
        },
        "errors": [],
        "readiness": {
            "state": "not_assessed",
            "ready_for_pilot": False,
            "ready_for_execution": False,
            "reason": (
                "A strict JSON/API check or header preview is not a full data, "
                "transformation, model-fit, or scientific verification."
            ),
        },
    }
    config_payload = None
    try:
        config_payload = read_json(config_path)
        report["checks"]["strict_json"] = "passed"
        report["config_sha256"] = sha256_file(config_path)
    except Exception as exc:
        report["checks"]["strict_json"] = "failed"
        report["errors"].append(f"config JSON: {type(exc).__name__}: {exc}")
        report["readiness"]["state"] = "blocked"
    if config_payload is not None:
        try:
            check_config_shape(config_payload)
            report["checks"]["native_config_shape"] = "passed"
        except Exception as exc:
            report["checks"]["native_config_shape"] = "failed"
            report["errors"].append(f"config structure: {type(exc).__name__}: {exc}")
            report["readiness"]["state"] = "blocked"

    search_payload = None
    if args.search:
        search_path = Path(args.search).expanduser().resolve()
        report["search_file"] = str(search_path)
        try:
            search_payload = read_json(search_path)
            report["checks"]["search_json"] = "passed"
            report["search_sha256"] = sha256_file(search_path)
        except Exception as exc:
            report["checks"]["search_json"] = "failed"
            report["errors"].append(f"search JSON: {type(exc).__name__}: {exc}")
        if search_payload is not None:
            try:
                check_search_shape(search_payload)
                report["checks"]["search_shape"] = "passed"
            except Exception as exc:
                report["checks"]["search_shape"] = "failed"
                report["errors"].append(f"search structure: {type(exc).__name__}: {exc}")

    config = None
    imported_module = None
    try:
        source_root = project / "src"
        expected_package = source_root / "pyml_workbench" / "__init__.py"
        if not expected_package.is_file():
            raise FileNotFoundError(f"PYML source package not found: {expected_package}")
        sys.path.insert(0, str(source_root))
        import pyml_workbench
        from pyml_workbench import (
            ExperimentConfig, build_estimator, get_model, preview_dataset,
        )
        from pyml_workbench.sequence_models import (
            EXTENDED_MODEL_IDS, validate_extended_parameters,
        )
        from pyml_workbench.search import SearchSpec

        imported_module = Path(pyml_workbench.__file__).resolve()
        if imported_module != expected_package.resolve():
            raise RuntimeError(
                "Imported PYML package is not the selected project source: "
                f"expected {expected_package.resolve()}, got {imported_module}"
            )
        report["module_file"] = str(imported_module)
        report["package_version"] = pyml_workbench.__version__
        report["checks"]["project_source_import"] = "passed"

        if config_payload is not None and report["checks"]["native_config_shape"] == "passed":
            config = ExperimentConfig.from_dict(config_payload)
            report["checks"]["live_api_parse"] = "passed"
            model = get_model(config.model_id)
            if model.get("task") != config.task:
                raise BundleError(
                    f"Model {config.model_id} belongs to {model.get('task')}, "
                    f"not {config.task}"
                )
            if model.get("runtime_available") is False:
                raise BundleError(model.get("unavailable_reason", "Model dependency unavailable"))
            if config.model_id in EXTENDED_MODEL_IDS:
                validate_extended_parameters(config.model_id, config.parameters)
            else:
                build_estimator(config.model_id, config.parameters, seed=config.split.seed)
            report["checks"]["live_api_parse"] = "passed"
            report["checks"]["model_parameters"] = "passed"
            report["model"] = {
                "model_id": config.model_id,
                "task": config.task,
                "runtime_available": model.get("runtime_available", True),
            }

        if search_payload is not None and config is not None and report["checks"]["search_shape"] == "passed":
            report["checks"]["search_api_parse"] = "attempted"
            spec = SearchSpec.from_dict(search_payload)
            spec.objective.resolve(config.task)
            spec.space.validate_for(config)
            if config.evaluation is not None and spec.max_fits < config.evaluation.folds + 1:
                raise BundleError(
                    "Search fit budget must cover one primary fit plus every evaluation fold"
                )
            report["checks"]["search_api_parse"] = "passed"
    except Exception as exc:
        report["errors"].append(f"live runtime: {type(exc).__name__}: {exc}")
        if report["checks"]["project_source_import"] != "passed":
            report["checks"]["project_source_import"] = "failed"
        if (
            config_payload is not None
            and report["checks"]["native_config_shape"] == "passed"
            and report["checks"]["live_api_parse"] == "not_run"
        ):
            report["checks"]["live_api_parse"] = "failed"
        if config is not None and report["checks"]["model_parameters"] == "not_run":
            report["checks"]["model_parameters"] = "failed"
        if report["checks"]["search_api_parse"] == "attempted":
            report["checks"]["search_api_parse"] = "failed"

    if config is not None and imported_module is not None:
        source = Path(config.dataset.source_path).expanduser()
        if not source.is_absolute():
            source = (Path.cwd() / source).resolve()
        report["source_file"] = str(source)
        if not source.is_file():
            report["checks"]["source_header_preview"] = "not_available"
            report["readiness"]["state"] = "pending_real_data"
        else:
            try:
                from pyml_workbench import preview_dataset
                preview = preview_dataset(config.dataset)
                columns = set(preview.columns)
                required = {
                    name for name in (
                        config.dataset.target_column,
                        *(config.dataset.feature_columns or ()),
                        config.split.group_column,
                        config.split.time_column,
                        config.dataset.reference_labels_column,
                    ) if name
                }
                if config.sequence is not None:
                    required.update(
                        name for name in (
                            config.sequence.group_column,
                            config.sequence.time_column,
                            *(config.sequence.observation_columns or ()),
                        ) if name
                    )
                missing = sorted(required - columns)
                if missing:
                    raise BundleError(f"Source is missing configured column(s): {missing}")
                report["checks"]["source_header_preview"] = "passed"
                report["source_preview"] = {
                    "rows_read_for_preview": len(preview.sample),
                    "columns": list(preview.columns),
                    "selected_sheet": preview.selected_sheet,
                    "sample_values_included": False,
                }
                report["readiness"]["state"] = "input_header_checked"
            except Exception as exc:
                report["errors"].append(f"source preview: {type(exc).__name__}: {exc}")
                report["checks"]["source_header_preview"] = "failed"
                report["readiness"]["state"] = "blocked"

    if report["errors"]:
        report["readiness"]["state"] = "blocked"
    report["readiness"]["ready_for_pilot"] = False
    report["readiness"]["ready_for_execution"] = False
    report["readiness"]["reason"] = (
        "This report never grants readiness. It does not validate all data values, "
        "run external scripts, fit a model, inspect test results, or certify the study."
    )
    encoded = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.output:
        output = Path(args.output).expanduser().resolve()
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(encoded)
        print(f"Wrote {output}")
    else:
        sys.stdout.write(encoded)
    return 2 if report["errors"] else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"BLOCKED: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(2)
