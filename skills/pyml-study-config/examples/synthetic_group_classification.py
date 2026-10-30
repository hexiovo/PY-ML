#!/usr/bin/env python3
"""Synthetic-only behavior example: grouped classification, fold-local prep, no test."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import random
import sys
import tempfile
from typing import Any


def make_rows():
    rng = random.Random(41207)
    rows = []
    for subject_index in range(20):
        subject = f"S{subject_index + 1:02d}"
        for trial in range(12):
            outcome = "positive" if trial % 2 else "negative"
            direction = 1.5 if outcome == "positive" else -1.5
            signal = direction + rng.gauss(0, 0.35)
            category = "blue" if outcome == "positive" else "amber"
            if (subject_index * 12 + trial) % 29 == 0:
                signal_value = ""
            else:
                signal_value = f"{signal:.8f}"
            if (subject_index * 12 + trial) % 31 == 0:
                category_value = ""
            else:
                category_value = category
            rows.append({
                "subject_id": subject,
                "trial": trial + 1,
                "signal": signal_value,
                "category": category_value,
                "outcome": outcome,
            })
    return rows


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a labeled synthetic data file and verify current-source PYML behavior."
    )
    parser.add_argument("--project", required=True, help="PYML checkout used by this example.")
    parser.add_argument(
        "--work-dir",
        help="Optional new or empty output directory. Existing files are never overwritten.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(work_dir: Path, project: Path) -> dict[str, Any]:
    source_root = project / "src"
    expected_package = source_root / "pyml_workbench" / "__init__.py"
    if not expected_package.is_file():
        raise FileNotFoundError(f"PYML source package not found: {expected_package}")
    sys.path.insert(0, str(source_root))

    import pyml_workbench
    from pyml_workbench import (
        DatasetConfig,
        EvaluationConfig,
        ExperimentConfig,
        SplitConfig,
    )
    import pyml_workbench.experiment as experiment_module
    build_snapshot = experiment_module.build_snapshot
    import pyml_workbench.preprocessing as preprocessing_module

    imported_module = Path(pyml_workbench.__file__).resolve()
    if imported_module != expected_package.resolve():
        raise RuntimeError(
            "Imported PYML package is not the selected source: "
            f"expected {expected_package.resolve()}, got {imported_module}"
        )

    data_path = work_dir / "synthetic_group_trials.csv"
    config_path = work_dir / "experiment.json"
    summary_path = work_dir / "verification.json"
    rows = make_rows()
    with data_path.open("x", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("subject_id", "trial", "signal", "category", "outcome"),
        )
        writer.writeheader()
        writer.writerows(rows)

    config = ExperimentConfig(
        dataset=DatasetConfig(
            source_path=str(data_path.resolve()),
            target_column="outcome",
            feature_columns=("signal", "category"),
        ),
        task="classification",
        model_id="C01",
        parameters={"max_iter": 400},
        split=SplitConfig(
            seed=42,
            strategy="group",
            group_column="subject_id",
        ),
        evaluation=EvaluationConfig(folds=2, strategy="group", gap=0),
        reduction_method="pca",
        reduction_components=1,
    )
    config_payload = config.to_dict()
    config_path.write_text(
        json.dumps(config_payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    snapshot = build_snapshot(config)

    original_fit_transform = preprocessing_module.fit_transform_preprocessor
    fit_positions: list[dict[str, Any]] = []

    def tracked_fit_transform(preprocessor, values, *, fit_scope):
        fitted, reduction = original_fit_transform(
            preprocessor,
            values,
            fit_scope=fit_scope,
        )
        fit_positions.append({
            "scope": fit_scope,
            "positions": [int(position) for position in values.index],
            "reduction": reduction,
        })
        return fitted, reduction

    preprocessing_module.fit_transform_preprocessor = tracked_fit_transform
    experiment_module.fit_transform_preprocessor = tracked_fit_transform
    try:
        session = experiment_module.prepare_experiment(config, snapshot=snapshot)
    finally:
        preprocessing_module.fit_transform_preprocessor = original_fit_transform
        experiment_module.fit_transform_preprocessor = original_fit_transform

    frame = snapshot.structure
    if frame is None or "subject_id" not in frame:
        raise AssertionError("The source snapshot did not retain the grouping column")
    group_values = frame["subject_id"]
    all_classes = set(snapshot.target.unique())
    split_groups = {
        name: set(group_values.iloc[positions].unique())
        for name, positions in snapshot.splits.items()
    }
    for name, positions in snapshot.splits.items():
        if set(snapshot.target.iloc[positions].unique()) != all_classes:
            raise AssertionError(f"Holdout partition {name} does not contain every class")
    if any(
        split_groups[left] & split_groups[right]
        for left, right in (
            ("train", "validation"),
            ("train", "test"),
            ("validation", "test"),
        )
    ):
        raise AssertionError("A subject appears in more than one holdout partition")

    cross_validation = session.metrics.get("cross_validation")
    if not cross_validation or len(cross_validation["folds"]) != 2:
        raise AssertionError("Expected the requested two group-preserving CV folds")
    fold_summaries = []
    expected_fit_sets = [set(map(int, snapshot.splits["train"]))]
    for fold in cross_validation["folds"]:
        train_positions = set(map(int, fold["train_positions"]))
        validation_positions = set(map(int, fold["validation_positions"]))
        train_groups = set(group_values.iloc[sorted(train_positions)].unique())
        validation_groups = set(group_values.iloc[sorted(validation_positions)].unique())
        if train_groups & validation_groups:
            raise AssertionError(f"Group overlap in CV fold {fold['fold']}")
        if train_positions & validation_positions:
            raise AssertionError(f"Row overlap in CV fold {fold['fold']}")
        if train_positions & set(map(int, snapshot.splits["test"])):
            raise AssertionError("CV training included a final-test row")
        if validation_positions & set(map(int, snapshot.splits["test"])):
            raise AssertionError("CV validation included a final-test row")
        expected_fit_sets.append(train_positions)
        fold_summaries.append({
            "fold": fold["fold"],
            "train_rows": len(train_positions),
            "validation_rows": len(validation_positions),
            "train_groups": sorted(train_groups),
            "validation_groups": sorted(validation_groups),
            "group_overlap": [],
        })

    observed_fit_sets = [set(item["positions"]) for item in fit_positions]
    if observed_fit_sets != expected_fit_sets:
        raise AssertionError(
            "Observed preprocessor fit row positions differ from the main training "
            "partition and CV fold training partitions"
        )
    expected_scopes = ["train", *("cv_fold_train" for _ in cross_validation["folds"])]
    if [item["scope"] for item in fit_positions] != expected_scopes:
        raise AssertionError(f"Unexpected preprocessor fit scopes: {fit_positions}")
    if session.test_evaluation_count != 0 or "test" in session.metrics:
        raise AssertionError("prepare_experiment unexpectedly evaluated the final test set")
    if any("test" in event for event in session.events):
        raise AssertionError(f"Unexpected test event during preparation: {session.events}")
    preprocessing = session.fitted_model.preprocessing
    if preprocessing.get("fit_scope") != "train":
        raise AssertionError(f"Unexpected preprocessing fit scope: {preprocessing}")
    if set(preprocessing.get("numeric_columns", [])) != {"signal"}:
        raise AssertionError(f"Unexpected numeric preprocessing columns: {preprocessing}")
    if set(preprocessing.get("categorical_columns", [])) != {"category"}:
        raise AssertionError(f"Unexpected categorical preprocessing columns: {preprocessing}")
    reduction = preprocessing.get("reduction")
    if not reduction or reduction.get("method") != "pca" or reduction.get("fit_scope") != "train":
        raise AssertionError(f"Main PCA was not fitted on training rows only: {preprocessing}")
    if any(
        not fold.get("reduction") or fold["reduction"].get("fit_scope") != "cv_fold_train"
        for fold in cross_validation["folds"]
    ):
        raise AssertionError("PCA was not independently fitted inside every CV training fold")

    summary = {
        "schema_version": 1,
        "synthetic_only": True,
        "status": "verified",
        "verification_scope": [
            "live project source imported and module path checked",
            "group holdout partitions have no subject overlap",
            "group CV folds have no subject or row overlap",
            "numeric/category preprocessing and PCA were fitted only on the main/CV training positions",
            "final test was not evaluated",
        ],
        "python_executable": str(Path(sys.executable).resolve()),
        "package_version": pyml_workbench.__version__,
        "module_file": str(imported_module),
        "source_file": data_path.name,
        "source_sha256": sha256(data_path),
        "rows": len(rows),
        "independent_units": len(set(row["subject_id"] for row in rows)),
        "missing_signal_cells": sum(row["signal"] == "" for row in rows),
        "missing_category_cells": sum(row["category"] == "" for row in rows),
        "native_config_file": config_path.name,
        "native_config_sha256": sha256(config_path),
        "task": config.task,
        "model_id": config.model_id,
        "features": list(config.dataset.feature_columns),
        "excluded_group_column": "subject_id",
        "target_column": "outcome",
        "holdout_group_counts": {
            name: len(groups) for name, groups in split_groups.items()
        },
        "holdout_group_overlap": {
            "train_validation": [],
            "train_test": [],
            "validation_test": [],
        },
        "preprocessor_fit_row_counts": [len(item["positions"]) for item in fit_positions],
        "preprocessor_fit_scopes": [item["scope"] for item in fit_positions],
        "preprocessing": {
            "numeric_columns": preprocessing.get("numeric_columns"),
            "categorical_columns": preprocessing.get("categorical_columns"),
            "numeric_scaler": preprocessing.get("numeric_scaler"),
            "fit_scope": preprocessing.get("fit_scope"),
            "main_fit_row_count": preprocessing.get("fit_row_count"),
            "reduction": reduction,
        },
        "cross_validation": {
            "scheme": cross_validation["scheme"],
            "fold_count": cross_validation["fold_count"],
            "scope": cross_validation["scope"],
            "folds": fold_summaries,
        },
        "final_test": {
            "evaluated": False,
            "evaluation_count": session.test_evaluation_count,
        },
        "note": (
            "Synthetic software-behavior evidence only. No scientific validity, "
            "real-data performance, or inferential result is claimed."
        ),
    }
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> int:
    args = parse_args()
    project = Path(args.project).expanduser().resolve()
    if args.work_dir:
        work_dir = Path(args.work_dir).expanduser().resolve()
        work_dir.mkdir(parents=True, exist_ok=True)
        if any(work_dir.iterdir()):
            raise FileExistsError(
                f"Work directory is not empty; refusing to overwrite: {work_dir}"
            )
        summary = run(work_dir, project)
        print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
        print(f"Evidence directory: {work_dir}")
        return 0
    with tempfile.TemporaryDirectory(prefix="pyml-study-config-") as temporary:
        work_dir = Path(temporary)
        summary = run(work_dir, project)
        print(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(2)
