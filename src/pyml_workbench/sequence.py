"""Immutable sequence plans and owned persistence for sequence experiments.

This module intentionally has no dependency on optional model libraries. It is
safe to import from a base-only installation.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Literal, TYPE_CHECKING

import joblib
import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from .config import ExperimentConfig


JSONValue = Any
_PLAN_SCHEMA_VERSION = 1
_PLAN_KIND = "pyml_workbench.sequence_plan"
_OWNED_PLAN_KIND = "pyml_workbench.owned_sequence_plan"
_RECEIPT_FIELDS = {
    "schema_version",
    "kind",
    "relative_path",
    "file_sha256",
    "manifest_sha256",
    "manifest",
}
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_SPLIT_NAMES = ("train", "validation", "test")
_HMM_IDS = {"H01": "hmm_continuous", "H02": "hmm_continuous", "H03": "hmm_categorical"}
_WINDOW_IDS = {"N04": "window_regression", "N06": "window_regression"}


class SequenceError(ValueError):
    """Raised when sequence structure, ownership, or persisted identity is invalid."""


@dataclass(frozen=True)
class SequenceConfig:
    """JSON-safe structural settings for sequence inputs."""

    group_column: str | None = None
    time_column: str | None = None
    order_mode: Literal["row", "time"] = "row"
    observation_columns: tuple[str, ...] | None = None
    window: int = 10
    horizon: int = 1

    def __post_init__(self) -> None:
        if self.observation_columns is not None:
            object.__setattr__(self, "observation_columns", tuple(self.observation_columns))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any] | "SequenceConfig") -> "SequenceConfig":
        if isinstance(payload, cls):
            return payload
        if not isinstance(payload, Mapping):
            raise SequenceError("sequence configuration must be a JSON object")
        allowed = {"group_column", "time_column", "order_mode", "observation_columns", "window", "horizon"}
        unknown = set(payload) - allowed
        if unknown:
            raise SequenceError(f"Unknown sequence configuration fields: {sorted(unknown)}")
        values = dict(payload)
        columns = values.get("observation_columns")
        if columns is not None:
            if isinstance(columns, str):
                raise SequenceError("observation_columns must be a list of column names")
            values["observation_columns"] = tuple(columns)
        try:
            return cls(**values)
        except TypeError as exc:
            raise SequenceError(f"Invalid sequence configuration: {exc}") from exc

    def to_dict(self) -> dict[str, JSONValue]:
        return {
            "group_column": self.group_column,
            "time_column": self.time_column,
            "order_mode": self.order_mode,
            "observation_columns": None if self.observation_columns is None else list(self.observation_columns),
            "window": self.window,
            "horizon": self.horizon,
        }

    def validate(
        self,
        *,
        task: str,
        model_id: str | None = None,
        feature_columns: list[str] | tuple[str, ...] | None = None,
        target_column: str | None = None,
        available_columns: list[str] | tuple[str, ...] | None = None,
    ) -> "SequenceConfig":
        if self.order_mode not in {"row", "time"}:
            raise SequenceError("order_mode must be 'row' or 'time'")
        if self.time_column and self.order_mode != "time":
            raise SequenceError("A configured time_column requires order_mode='time'")
        if self.order_mode == "time" and not self.time_column:
            raise SequenceError("order_mode='time' requires a time_column")
        for name, value in (("group_column", self.group_column), ("time_column", self.time_column)):
            if value is not None and (not isinstance(value, str) or not value.strip()):
                raise SequenceError(f"{name} must be a non-empty column name")
        if self.group_column and self.time_column and self.group_column == self.time_column:
            raise SequenceError("group_column and time_column must be different columns")
        if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in (self.window, self.horizon)):
            raise SequenceError("window and horizon must be positive integers")

        selected = None if feature_columns is None else tuple(feature_columns)
        structural = {value for value in (self.group_column, self.time_column) if value is not None}
        if selected is not None:
            if len(set(selected)) != len(selected):
                raise SequenceError("feature_columns must be unique")
            if structural.intersection(selected):
                raise SequenceError("group/time structural columns cannot be model features")
            if target_column is not None and target_column in selected:
                raise SequenceError("target_column cannot also appear in feature_columns")
        if available_columns is not None:
            available = set(available_columns)
            for name in (*structural, *(self.observation_columns or ())):
                if name not in available:
                    raise SequenceError(f"Configured sequence column {name!r} does not exist")
            if selected is not None:
                missing = [name for name in selected if name not in available]
                if missing:
                    raise SequenceError(f"Selected feature columns do not exist: {missing}")
            if target_column is not None and target_column not in available:
                raise SequenceError(f"Target column {target_column!r} does not exist")
        if self.observation_columns is not None:
            if not self.observation_columns or len(set(self.observation_columns)) != len(self.observation_columns):
                raise SequenceError("observation_columns must contain unique column names")
            if structural.intersection(self.observation_columns):
                raise SequenceError("group/time structural columns cannot be observation columns")
            if target_column is not None and target_column in self.observation_columns:
                raise SequenceError("target_column cannot also appear in observation_columns")
            if selected is not None and not set(self.observation_columns).issubset(selected):
                raise SequenceError("observation_columns must be included in feature_columns")

        if model_id in _HMM_IDS:
            if task != "sequence_modeling":
                raise SequenceError("HMM models require task='sequence_modeling'")
            if target_column is not None:
                raise SequenceError("HMM sequence modeling does not accept a supervised target column")
            if not (self.observation_columns or selected):
                raise SequenceError("HMM models require explicit observation columns or feature columns")
            observation_columns = self.observation_columns or selected
            if model_id == "H03" and len(observation_columns or ()) != 1:
                raise SequenceError("H03 CategoricalHMM requires exactly one observation column")
        elif model_id in _WINDOW_IDS:
            if task != "regression":
                raise SequenceError("N04/N06 sequence models require task='regression'")
            if not target_column:
                raise SequenceError("N04/N06 sequence regression requires a target column")
            if not selected:
                raise SequenceError("N04/N06 sequence regression requires feature columns")
            if self.observation_columns is not None:
                raise SequenceError("observation_columns applies only to HMM models")
        elif model_id is not None:
            raise SequenceError(f"Model {model_id!r} does not use a SequencePlan")
        elif task not in {"sequence_modeling", "regression"}:
            raise SequenceError("Sequence settings require sequence_modeling or regression task")
        return self


@dataclass(frozen=True)
class SequencePartition:
    """Read-only partition view with original row positions and model inputs."""

    name: str
    row_positions: np.ndarray
    group_ids: tuple[JSONValue, ...]
    lengths: np.ndarray
    observations: np.ndarray | None = None
    observation_encoding: str | None = None
    window_inputs: np.ndarray | None = None
    window_targets: np.ndarray | None = None
    window_source_positions: np.ndarray | None = None
    window_target_positions: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.name not in _SPLIT_NAMES:
            raise SequenceError(f"Unknown sequence partition {self.name!r}")
        for name in (
            "row_positions", "lengths", "observations", "window_inputs", "window_targets",
            "window_source_positions", "window_target_positions",
        ):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _readonly_array(value))
        object.__setattr__(self, "group_ids", tuple(_copy_json_value(item) for item in self.group_ids))


@dataclass(frozen=True)
class SequencePlan:
    """Self-contained plan, data views, and identity for one sequence dataset."""

    sequence_config: SequenceConfig
    task: str
    model_family: str
    source_path: str
    source_sha256: str
    split_seed: int
    feature_columns: tuple[str, ...]
    target_column: str | None
    _features_frame: pd.DataFrame = field(repr=False, compare=False)
    _target_series: pd.Series | None = field(repr=False, compare=False)
    _group_keys_by_row: tuple[str, ...] = field(repr=False, compare=False)
    _group_values_by_row: tuple[JSONValue, ...] = field(repr=False, compare=False)
    _order_values_ns: np.ndarray | None = field(repr=False, compare=False)
    _split_positions: tuple[tuple[str, np.ndarray], ...] = field(repr=False, compare=False)
    train: SequencePartition
    validation: SequencePartition
    test: SequencePartition
    categorical_alphabet: tuple[JSONValue, ...] = ()
    categorical_n_features: int | None = None
    _manifest_json: str = field(default="{}", repr=False, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "sequence_config", SequenceConfig.from_dict(self.sequence_config))
        object.__setattr__(self, "feature_columns", tuple(self.feature_columns))
        object.__setattr__(self, "_features_frame", self._features_frame.copy(deep=True))
        if self._target_series is not None:
            object.__setattr__(self, "_target_series", self._target_series.copy(deep=True))
        object.__setattr__(self, "_group_keys_by_row", tuple(self._group_keys_by_row))
        object.__setattr__(self, "_group_values_by_row", tuple(_copy_json_value(item) for item in self._group_values_by_row))
        if self._order_values_ns is not None:
            object.__setattr__(self, "_order_values_ns", _readonly_array(self._order_values_ns, dtype=np.int64))
        frozen_splits = tuple((name, _readonly_array(values, dtype=np.int64)) for name, values in self._split_positions)
        object.__setattr__(self, "_split_positions", frozen_splits)
        object.__setattr__(self, "categorical_alphabet", tuple(_copy_json_value(item) for item in self.categorical_alphabet))

    @property
    def row_count(self) -> int:
        return len(self._features_frame)

    @property
    def features(self) -> pd.DataFrame:
        return self._features_frame.copy(deep=True)

    @property
    def target(self) -> pd.Series | None:
        return None if self._target_series is None else self._target_series.copy(deep=True)

    @property
    def split_positions(self) -> dict[str, np.ndarray]:
        return {name: values.copy() for name, values in self._split_positions}

    @property
    def partitions(self) -> dict[str, SequencePartition]:
        return {"train": self.train, "validation": self.validation, "test": self.test}

    @property
    def manifest(self) -> dict[str, JSONValue]:
        return json.loads(self._manifest_json)

    @property
    def plan_sha256(self) -> str:
        return self.manifest["plan_sha256"]

    @property
    def comparison_protocol_sha256(self) -> str:
        return self.manifest["comparison_protocol_sha256"]

    def to_dict(self) -> dict[str, JSONValue]:
        return self.manifest

    def verify(self, config: "ExperimentConfig | Mapping[str, Any] | None" = None, snapshot: Any = None) -> None:
        """Recompute owned data/plan identities and optionally cross-check config/snapshot."""
        manifest = self.manifest
        expected = self._make_manifest()
        if manifest != expected:
            raise SequenceError("Sequence plan data, split, window mapping, or metadata was modified")
        _verify_partition_invariants(self)
        if config is not None:
            _verify_config_identity(self, config)
        if snapshot is not None:
            snapshot_manifest = snapshot.to_dict()
            if snapshot_manifest.get("source_sha256") != self.source_sha256:
                raise SequenceError("Sequence plan source identity differs from its owned snapshot")
            if snapshot_manifest.get("row_count") != self.row_count:
                raise SequenceError("Sequence plan row count differs from its owned snapshot")
            if snapshot_manifest.get("data_sha256") != manifest["snapshot_data_sha256"]:
                raise SequenceError("Sequence plan data identity differs from its owned snapshot")
            if snapshot_manifest.get("split_sha256") != manifest["split_sha256"]:
                raise SequenceError("Sequence plan split identity differs from its owned snapshot")
            if snapshot_manifest.get("split_positions") != manifest["split_positions"]:
                raise SequenceError("Sequence plan positions differ from its owned snapshot")
            declared_plan_sha = snapshot_manifest.get("sequence_plan_sha256")
            if declared_plan_sha is not None and declared_plan_sha != self.plan_sha256:
                raise SequenceError("Owned snapshot does not bind this sequence plan digest")

    def _make_manifest(self) -> dict[str, JSONValue]:
        from .experiment import _json_digest, _table_digest

        snapshot_data_sha = _json_digest({
            "features": _table_digest(self._features_frame),
            "target": _table_digest(self._target_series),
            "objective_labels": None,
        })
        positions = {name: [int(item) for item in values] for name, values in self._split_positions}
        split_sha = _json_digest(positions)
        raw_data_sha = _json_digest({
            "snapshot_data_sha256": snapshot_data_sha,
            "group_keys": list(self._group_keys_by_row),
            "group_values": [_json_value(item) for item in self._group_values_by_row],
            "order_values_ns": None if self._order_values_ns is None else [int(item) for item in self._order_values_ns],
        })
        partition_manifest = {name: _partition_to_manifest(part) for name, part in self.partitions.items()}
        partition_data_sha = _json_digest(partition_manifest)
        sequence_config = self.sequence_config.to_dict()
        protocol = {
            "model_family": self.model_family,
            "task": self.task,
            "sequence_config": sequence_config,
            "feature_columns": list(self.feature_columns),
            "target_column": self.target_column,
            "source_sha256": self.source_sha256,
            "snapshot_data_sha256": snapshot_data_sha,
            "split_sha256": split_sha,
            "split_positions": positions,
            "partition_data_sha256": partition_data_sha,
            "preprocessing_protocol": "train_only_v1",
        }
        base = {
            "schema_version": _PLAN_SCHEMA_VERSION,
            "kind": _PLAN_KIND,
            "source_path": self.source_path,
            "source_sha256": self.source_sha256,
            "row_count": self.row_count,
            "task": self.task,
            "model_family": self.model_family,
            "sequence_config": sequence_config,
            "split_seed": int(self.split_seed),
            "feature_columns": list(self.feature_columns),
            "target_column": self.target_column,
            "snapshot_data_sha256": snapshot_data_sha,
            "raw_plan_data_sha256": raw_data_sha,
            "split_sha256": split_sha,
            "split_positions": positions,
            "split_counts": {name: len(values) for name, values in positions.items()},
            "partitions": partition_manifest,
            "partition_data_sha256": partition_data_sha,
            "categorical_alphabet": [_json_value(item) for item in self.categorical_alphabet],
            "categorical_n_features": self.categorical_n_features,
            "comparison_protocol": self.model_family,
            "comparison_protocol_sha256": _json_digest(protocol),
            "snapshot_identity_sha256": _json_digest({
                "source_sha256": self.source_sha256,
                "snapshot_data_sha256": snapshot_data_sha,
                "split_sha256": split_sha,
                "row_count": self.row_count,
            }),
        }
        base["plan_sha256"] = _json_digest(base)
        return base


def build_sequence_plan(
    frame: pd.DataFrame,
    sequence_config: SequenceConfig | Mapping[str, Any],
    *,
    task: str,
    model_id: str,
    target_column: str | None,
    feature_columns: list[str] | tuple[str, ...] | None,
    split_seed: int,
    source_path: str | Path,
    source_sha256: str,
) -> SequencePlan:
    """Build an owned, deterministic sequence plan from an already-loaded frame."""
    config = SequenceConfig.from_dict(sequence_config)
    if not isinstance(frame, pd.DataFrame):
        raise SequenceError("frame must be a pandas DataFrame loaded by the caller")
    if frame.empty:
        raise SequenceError("Sequence source frame is empty")
    if frame.columns.has_duplicates:
        raise SequenceError("Sequence source has duplicate column names")
    if not isinstance(split_seed, int) or isinstance(split_seed, bool) or split_seed < 0:
        raise SequenceError("split_seed must be a non-negative integer")
    if not isinstance(source_sha256, str) or not _SHA256_PATTERN.fullmatch(source_sha256):
        raise SequenceError("source_sha256 must be a lowercase 64-character SHA-256 digest")
    if model_id not in {*_HMM_IDS, *_WINDOW_IDS}:
        raise SequenceError(f"Model {model_id!r} does not use a SequencePlan")
    family = _HMM_IDS.get(model_id, _WINDOW_IDS.get(model_id))
    available_columns = tuple(str(column) for column in frame.columns)
    requested_features = None if feature_columns is None else tuple(feature_columns)
    if requested_features is None:
        base_columns = config.observation_columns if model_id in _HMM_IDS and config.observation_columns else available_columns
        excluded = {value for value in (target_column, config.group_column, config.time_column) if value is not None}
        requested_features = tuple(name for name in base_columns if name not in excluded)
    config.validate(
        task=task,
        model_id=model_id,
        feature_columns=requested_features,
        target_column=target_column,
        available_columns=available_columns,
    )
    missing = [name for name in requested_features if name not in frame.columns]
    if missing:
        raise SequenceError(f"Selected feature columns do not exist: {missing}")
    if not requested_features:
        raise SequenceError("Sequence plan requires at least one model feature")
    if config.group_column and config.group_column not in frame.columns:
        raise SequenceError(f"Group column {config.group_column!r} does not exist")
    if config.time_column and config.time_column not in frame.columns:
        raise SequenceError(f"Time column {config.time_column!r} does not exist")

    features = frame.loc[:, list(requested_features)].copy(deep=True)
    target = None
    if target_column is not None:
        if target_column not in frame.columns:
            raise SequenceError(f"Target column {target_column!r} does not exist")
        target = frame[target_column].copy(deep=True)

    group_keys, group_values = _group_keys(frame, config.group_column)
    time_values = _time_values(frame, config, group_keys)
    ordered_groups = _ordered_groups(group_keys, group_values, time_values)
    split_groups = _assign_groups(ordered_groups, group_column=config.group_column, row_count=len(frame))
    positions = {
        name: tuple(int(pos) for group in groups for pos in group["positions"])
        for name, groups in split_groups.items()
    }
    if any(not values for values in positions.values()):
        raise SequenceError("Sequence split produced an empty train/validation/test partition")

    target_numeric_by_row: dict[int, float] = {}
    if family == "window_regression":
        assert target is not None
        training_rows = list(positions["train"]) + list(positions["validation"])
        training_target = target.iloc[training_rows]
        if training_target.isna().any():
            raise SequenceError("N04/N06 train/validation targets contain missing values")
        try:
            numeric = pd.to_numeric(training_target, errors="raise").to_numpy(dtype=np.float32)
        except (TypeError, ValueError, OverflowError) as exc:
            raise SequenceError("N04/N06 train/validation targets must be numeric and fit float32") from exc
        if not np.isfinite(numeric).all():
            raise SequenceError("N04/N06 train/validation targets must be finite")
        target_numeric_by_row = {int(row): float(value) for row, value in zip(training_rows, numeric)}

    categorical_alphabet: tuple[JSONValue, ...] = ()
    categorical_n_features: int | None = None
    encoded_categorical: dict[int, int] = {}
    if model_id == "H03":
        assert config.observation_columns is not None or requested_features
        observation_column = (config.observation_columns or requested_features)[0]
        raw_observations = frame[observation_column]
        train_rows = positions["train"]
        alphabet: list[JSONValue] = []
        category_index: dict[str, int] = {}
        for row in train_rows:
            value = raw_observations.iloc[row]
            safe_value = _normal_scalar(value, context="categorical observation")
            key = _json_text(safe_value)
            if key not in category_index:
                category_index[key] = len(alphabet)
                alphabet.append(safe_value)
        if not alphabet:
            raise SequenceError("H03 training partition has no categorical observations")
        for row in positions["validation"]:
            value = raw_observations.iloc[row]
            safe_value = _normal_scalar(value, context="categorical observation")
            key = _json_text(safe_value)
            if key not in category_index:
                raise SequenceError(
                    f"H03 validation observation {safe_value!r} at source row {row} is unknown to the train-only alphabet"
                )
            encoded_categorical[row] = category_index[key]
        for row in train_rows:
            value = raw_observations.iloc[row]
            safe_value = _normal_scalar(value, context="categorical training observation")
            encoded_categorical[row] = category_index[_json_text(safe_value)]
        categorical_alphabet = tuple(alphabet)
        categorical_n_features = len(alphabet)

    raw_continuous_observations = None
    continuous_training_values: dict[int, np.ndarray] = {}
    if model_id in {"H01", "H02"}:
        observation_columns = config.observation_columns or requested_features
        raw_continuous_observations = frame.loc[:, list(observation_columns)].to_numpy(dtype=object, copy=True)
        for split_name in ("train", "validation"):
            rows = list(positions[split_name])
            try:
                numeric = frame.loc[:, list(observation_columns)].iloc[rows].apply(pd.to_numeric, errors="raise").to_numpy(dtype=np.float32)
            except (TypeError, ValueError, OverflowError) as exc:
                raise SequenceError(f"H01/H02 {split_name} observations must be continuous numeric values") from exc
            if np.isinf(numeric).any():
                raise SequenceError(f"H01/H02 {split_name} observations cannot be infinite")
            continuous_training_values.update({int(row): value for row, value in zip(rows, numeric)})

    partitions: dict[str, SequencePartition] = {}
    for name in _SPLIT_NAMES:
        groups = split_groups[name]
        part_rows = np.asarray(positions[name], dtype=np.int64)
        lengths = np.asarray([len(group["positions"]) for group in groups], dtype=np.int64)
        group_ids = tuple(_copy_json_value(group["label"]) for group in groups)
        observations = None
        observation_encoding = None
        window_inputs = window_targets = window_source_positions = window_target_positions = None
        if model_id in {"H01", "H02"}:
            assert raw_continuous_observations is not None
            if name == "test":
                observations = raw_continuous_observations[part_rows]
                observation_encoding = "continuous_raw_unchecked"
            else:
                observations = np.stack([continuous_training_values[int(row)] for row in part_rows]).astype(np.float32, copy=False)
                observation_encoding = "continuous_numeric"
        elif model_id == "H03":
            if name == "test":
                observation_column = (config.observation_columns or requested_features)[0]
                raw = frame[observation_column].iloc[part_rows].to_numpy(dtype=object, copy=True)
                observations = raw.reshape(-1, 1)
                observation_encoding = "categorical_raw_unchecked"
            else:
                observations = np.asarray([[encoded_categorical[int(row)]] for row in part_rows], dtype=np.int64)
                observation_encoding = "categorical_train_encoded"
        else:
            source_maps: list[np.ndarray] = []
            target_maps: list[int] = []
            for group in groups:
                group_rows = np.asarray(group["positions"], dtype=np.int64)
                needed = config.window + config.horizon
                if len(group_rows) < needed:
                    raise SequenceError(
                        f"N04/N06 split {name!r} group {group['label']!r} has {len(group_rows)} rows; "
                        f"window={config.window}, horizon={config.horizon} requires at least {needed}"
                    )
                window_count = len(group_rows) - needed + 1
                for start in range(window_count):
                    source_maps.append(group_rows[start : start + config.window])
                    target_maps.append(int(group_rows[start + needed - 1]))
            if not source_maps:
                raise SequenceError(f"N04/N06 split {name!r} has no valid windows")
            window_source_positions = np.stack(source_maps).astype(np.int64, copy=False)
            window_target_positions = np.asarray(target_maps, dtype=np.int64)
            feature_values = features.to_numpy(dtype=object, copy=True)
            window_inputs = feature_values[window_source_positions]
            if name == "test":
                assert target is not None
                window_targets = target.iloc[window_target_positions].to_numpy(dtype=object, copy=True).reshape(-1, 1)
            else:
                window_targets = np.asarray(
                    [target_numeric_by_row[int(row)] for row in window_target_positions], dtype=np.float32
                ).reshape(-1, 1)
        partitions[name] = SequencePartition(
            name=name,
            row_positions=part_rows,
            group_ids=group_ids,
            lengths=lengths,
            observations=observations,
            observation_encoding=observation_encoding,
            window_inputs=window_inputs,
            window_targets=window_targets,
            window_source_positions=window_source_positions,
            window_target_positions=window_target_positions,
        )

    plan = SequencePlan(
        sequence_config=config,
        task=task,
        model_family=family,
        source_path=str(Path(source_path).expanduser().resolve()),
        source_sha256=source_sha256,
        split_seed=split_seed,
        feature_columns=tuple(requested_features),
        target_column=target_column,
        _features_frame=features,
        _target_series=target,
        _group_keys_by_row=group_keys,
        _group_values_by_row=group_values,
        _order_values_ns=time_values,
        _split_positions=tuple((name, np.asarray(positions[name], dtype=np.int64)) for name in _SPLIT_NAMES),
        train=partitions["train"],
        validation=partitions["validation"],
        test=partitions["test"],
        categorical_alphabet=categorical_alphabet,
        categorical_n_features=categorical_n_features,
        _manifest_json="{}",
    )
    object.__setattr__(plan, "_manifest_json", _json_text(plan._make_manifest()))
    plan.verify()
    return plan


def save_owned_sequence_plan(plan: SequencePlan, path: str | Path) -> dict[str, JSONValue]:
    """Atomically persist a validated sequence plan without reading its source."""
    if not isinstance(plan, SequencePlan):
        raise SequenceError("Owned sequence plan must be a SequencePlan")
    plan.verify()
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=target.name + ".", suffix=".tmp", dir=target.parent)
    os.close(descriptor)
    try:
        joblib.dump(plan, temporary, compress=3)
        with open(temporary, "r+b") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        Path(temporary).unlink(missing_ok=True)
    manifest = plan.to_dict()
    return {
        "schema_version": _PLAN_SCHEMA_VERSION,
        "kind": _OWNED_PLAN_KIND,
        "relative_path": target.name,
        "file_sha256": _file_sha256(target),
        "manifest_sha256": _json_digest(manifest),
        "manifest": manifest,
    }


def load_owned_sequence_plan(
    path: str | Path,
    receipt: Mapping[str, object],
    *,
    expected_manifest: Mapping[str, object] | None = None,
    config: "ExperimentConfig | Mapping[str, Any] | None" = None,
    snapshot: Any = None,
) -> SequencePlan:
    """Load a plan only when its receipt, contents, config, and snapshot agree."""
    target = Path(path).expanduser().resolve()
    receipt_manifest = _validate_receipt(receipt, target)
    try:
        observed_file_sha = _file_sha256(target)
    except FileNotFoundError as exc:
        raise SequenceError(f"Owned sequence plan file is missing: {target}") from exc
    if observed_file_sha != receipt["file_sha256"]:
        raise SequenceError("Owned sequence plan file checksum does not match its receipt")
    try:
        plan = joblib.load(target)
    except Exception as exc:
        raise SequenceError(f"Owned sequence plan could not be deserialized: {target}") from exc
    if not isinstance(plan, SequencePlan):
        raise SequenceError("Owned sequence plan artifact has an unexpected type")
    _restore_plan_immutability(plan)
    plan.verify(config=config, snapshot=snapshot)
    if plan.to_dict() != receipt_manifest:
        raise SequenceError("Owned sequence plan manifest does not match its receipt")
    if expected_manifest is not None:
        expected = _json_object(expected_manifest, "expected plan manifest")
        if plan.to_dict() != expected:
            raise SequenceError("Owned sequence plan manifest does not match the queued plan")
    return plan


def _restore_plan_immutability(plan: SequencePlan) -> None:
    """Reapply NumPy read-only flags that pickle formats do not preserve."""
    for name in _SPLIT_NAMES:
        partition = getattr(plan, name)
        frozen = SequencePartition(
            name=partition.name,
            row_positions=partition.row_positions,
            group_ids=partition.group_ids,
            lengths=partition.lengths,
            observations=partition.observations,
            observation_encoding=partition.observation_encoding,
            window_inputs=partition.window_inputs,
            window_targets=partition.window_targets,
            window_source_positions=partition.window_source_positions,
            window_target_positions=partition.window_target_positions,
        )
        object.__setattr__(plan, name, frozen)
    if plan._order_values_ns is not None:
        object.__setattr__(plan, "_order_values_ns", _readonly_array(plan._order_values_ns, dtype=np.int64))
    object.__setattr__(
        plan,
        "_split_positions",
        tuple((name, _readonly_array(values, dtype=np.int64)) for name, values in plan._split_positions),
    )


def _group_keys(frame: pd.DataFrame, group_column: str | None) -> tuple[tuple[str, ...], tuple[JSONValue, ...]]:
    keys: list[str] = []
    values: list[JSONValue] = []
    if group_column is None:
        key = _json_text({"group": "all_rows"})
        label = "all_rows"
        return tuple(key for _ in range(len(frame))), tuple(label for _ in range(len(frame)))
    for row, value in enumerate(frame[group_column].tolist()):
        safe = _normal_scalar(value, context=f"group value at source row {row}")
        key = _json_text(safe)
        keys.append(key)
        values.append(safe)
    return tuple(keys), tuple(values)


def _time_values(frame: pd.DataFrame, config: SequenceConfig, group_keys: tuple[str, ...]) -> np.ndarray | None:
    if config.time_column is None:
        return None
    try:
        parsed = pd.to_datetime(frame[config.time_column], errors="coerce", utc=True)
    except (TypeError, ValueError, OverflowError) as exc:
        raise SequenceError(f"Time column {config.time_column!r} cannot be parsed") from exc
    if parsed.isna().any():
        bad = int(np.flatnonzero(parsed.isna().to_numpy())[0])
        raise SequenceError(f"Time column {config.time_column!r} is missing or invalid at source row {bad}")
    values = parsed.astype("int64").to_numpy(copy=True)
    seen: set[tuple[str, int]] = set()
    for row, (key, value) in enumerate(zip(group_keys, values)):
        identity = (key, int(value))
        if identity in seen:
            raise SequenceError(f"Duplicate time value within a group at source row {row}")
        seen.add(identity)
    return values


def _ordered_groups(
    group_keys: tuple[str, ...],
    group_values: tuple[JSONValue, ...],
    time_values: np.ndarray | None,
) -> list[dict[str, Any]]:
    grouped: dict[str, list[int]] = {}
    labels: dict[str, JSONValue] = {}
    for position, key in enumerate(group_keys):
        grouped.setdefault(key, []).append(position)
        labels.setdefault(key, group_values[position])
    result = []
    for key, positions in grouped.items():
        if time_values is not None:
            positions = sorted(positions, key=lambda pos: (int(time_values[pos]), pos))
            first_time = int(time_values[positions[0]])
        else:
            first_time = min(positions)
        result.append({"key": key, "label": labels[key], "positions": positions, "order_key": (first_time, min(positions))})
    result.sort(key=lambda group: group["order_key"])
    return result


def _assign_groups(
    ordered_groups: list[dict[str, Any]],
    *,
    group_column: str | None,
    row_count: int,
) -> dict[str, list[dict[str, Any]]]:
    if group_column is None:
        positions = ordered_groups[0]["positions"]
        train_end = _nearest_cut(row_count, target_fraction=0.6, minimum=1, maximum=row_count - 2)
        validation_end = _nearest_cut(row_count, target_fraction=0.8, minimum=train_end + 1, maximum=row_count - 1)
        return {
            "train": [{**ordered_groups[0], "positions": positions[:train_end]}],
            "validation": [{**ordered_groups[0], "positions": positions[train_end:validation_end]}],
            "test": [{**ordered_groups[0], "positions": positions[validation_end:]}],
        }
    if len(ordered_groups) < 3:
        raise SequenceError("Group-wise sequence splitting needs at least three distinct groups")
    counts = [len(group["positions"]) for group in ordered_groups]
    cumulative = np.cumsum(counts).tolist()
    total = sum(counts)
    first_cut = min(range(1, len(ordered_groups) - 1), key=lambda index: (abs(cumulative[index - 1] - 0.6 * total), index))
    second_cut = min(
        range(first_cut + 1, len(ordered_groups)),
        key=lambda index: (abs(cumulative[index - 1] - 0.8 * total), index),
    )
    return {
        "train": ordered_groups[:first_cut],
        "validation": ordered_groups[first_cut:second_cut],
        "test": ordered_groups[second_cut:],
    }


def _nearest_cut(row_count: int, *, target_fraction: float, minimum: int, maximum: int) -> int:
    if maximum < minimum:
        raise SequenceError("At least three ordered rows are required for sequence splitting")
    target = row_count * target_fraction
    return min(range(minimum, maximum + 1), key=lambda value: (abs(value - target), value))


def _verify_partition_invariants(plan: SequencePlan) -> None:
    partitions = plan.partitions
    observed: list[int] = []
    for name, partition in partitions.items():
        if partition.name != name or partition.row_positions.ndim != 1:
            raise SequenceError(f"Sequence partition {name!r} has invalid row positions")
        if partition.lengths.ndim != 1 or np.any(partition.lengths <= 0) or int(partition.lengths.sum()) != len(partition.row_positions):
            raise SequenceError(f"Sequence partition {name!r} has invalid segment lengths")
        if len(partition.group_ids) != len(partition.lengths):
            raise SequenceError(f"Sequence partition {name!r} group IDs do not match segment lengths")
        observed.extend(int(item) for item in partition.row_positions)
        if partition.observations is not None and len(partition.observations) != len(partition.row_positions):
            raise SequenceError(f"Sequence partition {name!r} observation rows do not match source positions")
        if partition.window_inputs is not None:
            assert partition.window_source_positions is not None
            assert partition.window_target_positions is not None
            assert partition.window_targets is not None
            if (
                partition.window_inputs.shape[0] != len(partition.window_target_positions)
                or partition.window_source_positions.shape[0] != len(partition.window_target_positions)
                or partition.window_targets.shape[0] != len(partition.window_target_positions)
            ):
                raise SequenceError(f"Sequence partition {name!r} window arrays have inconsistent row counts")
            allowed = set(int(item) for item in partition.row_positions)
            if any(int(item) not in allowed for item in partition.window_target_positions):
                raise SequenceError(f"Sequence partition {name!r} window target crosses its split")
            if any(int(item) not in allowed for item in partition.window_source_positions.flat):
                raise SequenceError(f"Sequence partition {name!r} window context crosses its split")
            if partition.window_inputs.shape[1] != plan.sequence_config.window:
                raise SequenceError(f"Sequence partition {name!r} window length differs from its config")
            if partition.window_source_positions.shape[1] != plan.sequence_config.window:
                raise SequenceError(f"Sequence partition {name!r} source-row mapping has the wrong window length")
            for sources, target in zip(partition.window_source_positions, partition.window_target_positions):
                positions = [int(item) for item in sources]
                if any(plan._group_keys_by_row[pos] != plan._group_keys_by_row[int(target)] for pos in positions):
                    raise SequenceError(f"Sequence partition {name!r} window crosses a group boundary")
                # Validate the positional mapping against each split's ordered segment,
                # not only the source values. This catches gaps, reordering, a target
                # attached to the wrong segment, and incorrect horizon offsets.
                expected_rows: dict[str, list[int]] = {}
                cursor = 0
                for group_id, length in zip(partition.group_ids, partition.lengths):
                    segment = [int(item) for item in partition.row_positions[cursor : cursor + int(length)]]
                    cursor += int(length)
                    expected_rows[_json_text(_json_value(group_id))] = segment
                group_label = _json_text(_json_value(plan._group_values_by_row[int(target)]))
                segment = expected_rows.get(group_label)
                if segment is None:
                    # Group IDs are serialized labels, while the authoritative key is
                    # their normalized JSON representation. A missing entry indicates
                    # a corrupted partition mapping.
                    raise SequenceError(f"Sequence partition {name!r} window maps outside its declared group segment")
                rank = {position: index for index, position in enumerate(segment)}
                if any(position not in rank for position in positions) or int(target) not in rank:
                    raise SequenceError(f"Sequence partition {name!r} window maps outside its declared group segment")
                source_ranks = [rank[position] for position in positions]
                target_rank = rank[int(target)]
                expected_window = list(range(source_ranks[0], source_ranks[0] + plan.sequence_config.window))
                if (
                    source_ranks != expected_window
                    or target_rank != source_ranks[-1] + plan.sequence_config.horizon
                ):
                    raise SequenceError(f"Sequence partition {name!r} window mapping violates its window/horizon offsets")
                if plan._order_values_ns is not None:
                    ordered = [int(plan._order_values_ns[pos]) for pos in positions]
                    if ordered != sorted(ordered) or int(plan._order_values_ns[int(target)]) <= ordered[-1]:
                        raise SequenceError(f"Sequence partition {name!r} window crosses or reverses a time boundary")
    if sorted(observed) != list(range(plan.row_count)):
        raise SequenceError("Sequence splits must cover every source row exactly once")
    if plan.sequence_config.group_column is not None:
        group_to_split: dict[str, str] = {}
        for name, part in partitions.items():
            for row in part.row_positions:
                key = plan._group_keys_by_row[int(row)]
                previous = group_to_split.setdefault(key, name)
                if previous != name:
                    raise SequenceError("A group appears in more than one partition")
    if plan.model_family.startswith("hmm_"):
        for name, part in partitions.items():
            if part.observations is None or len(part.observations) != len(part.row_positions):
                raise SequenceError(f"HMM partition {name!r} is missing observation rows")
            if plan.model_family == "hmm_categorical":
                expected_encoding = "categorical_raw_unchecked" if name == "test" else "categorical_train_encoded"
                if part.observation_encoding != expected_encoding:
                    raise SequenceError(f"HMM partition {name!r} has an invalid categorical encoding state")
            if plan.model_family == "hmm_categorical" and part.observations.shape[1:] != (1,):
                raise SequenceError("Categorical HMM observations must have exactly one column")


def _verify_config_identity(plan: SequencePlan, config: "ExperimentConfig | Mapping[str, Any]") -> None:
    if isinstance(config, Mapping):
        payload = dict(config)
    else:
        to_dict = getattr(config, "to_dict", None)
        if not callable(to_dict):
            raise SequenceError("config must be an ExperimentConfig or JSON-compatible mapping")
        payload = to_dict()
    if payload.get("task") != plan.task:
        raise SequenceError("Sequence plan task does not match the requested configuration")
    model_id = payload.get("model_id")
    model_family = _HMM_IDS.get(model_id, _WINDOW_IDS.get(model_id))
    if model_family != plan.model_family:
        raise SequenceError("Sequence plan comparison protocol does not match the model family")
    dataset = payload.get("dataset", {})
    if dataset.get("source_path") and Path(dataset["source_path"]).expanduser().resolve() != Path(plan.source_path):
        raise SequenceError("Sequence plan source path does not match the requested configuration")
    if dataset.get("target_column") != plan.target_column:
        raise SequenceError("Sequence plan target column does not match the requested configuration")
    features = dataset.get("feature_columns")
    if features is not None and list(features) != list(plan.feature_columns):
        raise SequenceError("Sequence plan feature columns do not match the requested configuration")
    split = payload.get("split", {})
    if split.get("seed", 42) != plan.split_seed:
        raise SequenceError("Sequence plan split seed does not match the requested configuration")
    sequence_settings = payload.get("sequence")
    if sequence_settings is None:
        sequence_settings = payload.get("sequence_config")
    if sequence_settings is not None and SequenceConfig.from_dict(sequence_settings).to_dict() != plan.sequence_config.to_dict():
        raise SequenceError("Sequence plan settings do not match the requested configuration")


def _partition_to_manifest(partition: SequencePartition) -> dict[str, Any]:
    return {
        "row_positions": [int(item) for item in partition.row_positions],
        "group_ids": [_json_value(item) for item in partition.group_ids],
        "lengths": [int(item) for item in partition.lengths],
        "observation_encoding": partition.observation_encoding,
        "observations_sha256": None if partition.observations is None else _array_sha256(partition.observations),
        "window_inputs_sha256": None if partition.window_inputs is None else _array_sha256(partition.window_inputs),
        "window_targets_sha256": None if partition.window_targets is None else _array_sha256(partition.window_targets),
        "window_source_positions": None if partition.window_source_positions is None else partition.window_source_positions.tolist(),
        "window_target_positions": None if partition.window_target_positions is None else partition.window_target_positions.tolist(),
    }


def _snapshot_data_digest(features: pd.DataFrame, target: pd.Series | None) -> str:
    from .experiment import _json_digest, _table_digest

    return _json_digest({
        "features": _table_digest(features),
        "target": _table_digest(target),
        "objective_labels": None,
    })


def _array_sha256(array: np.ndarray) -> str:
    values = np.asarray(array)
    digest = hashlib.sha256()
    digest.update(_json_text({"dtype": values.dtype.str, "shape": list(values.shape)}).encode("utf-8"))
    if values.dtype.kind in {"O", "U", "S"}:
        payload = [_json_value(item) for item in values.reshape(-1).tolist()]
        digest.update(_json_text(payload).encode("utf-8"))
    else:
        digest.update(np.ascontiguousarray(values).tobytes())
    return digest.hexdigest()


def _readonly_array(value: Any, dtype: Any = None) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def _normal_scalar(value: Any, *, context: str) -> JSONValue:
    if value is None or value is pd.NA or (not isinstance(value, (list, tuple, dict)) and pd.isna(value)):
        raise SequenceError(f"{context} cannot be missing")
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, (pd.Timestamp, datetime, date, np.datetime64)):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, bool | int | str):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise SequenceError(f"{context} must be finite")
        return value
    if isinstance(value, tuple):
        return [_normal_scalar(item, context=context) for item in value]
    raise SequenceError(f"{context} must be a JSON-compatible scalar, got {type(value).__name__}")


def _json_value(value: Any) -> JSONValue:
    if isinstance(value, np.generic):
        value = value.item()
    if value is pd.NA or value is None:
        return None
    if isinstance(value, (pd.Timestamp, datetime, date, np.datetime64)):
        return pd.Timestamp(value).isoformat()
    if isinstance(value, float) and not math.isfinite(value):
        return {"__float__": repr(value)}
    if isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, tuple):
        return [_json_value(item) for item in value]
    if isinstance(value, list):
        return [_json_value(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    return str(value)


def _copy_json_value(value: Any) -> JSONValue:
    return json.loads(_json_text(_json_value(value)))


def _json_text(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _json_digest(value: Any) -> str:
    return hashlib.sha256(_json_text(value).encode("utf-8")).hexdigest()


def _json_object(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise SequenceError(f"{label} must be a JSON object")
    try:
        parsed = json.loads(_json_text(dict(value)))
    except (TypeError, ValueError) as exc:
        raise SequenceError(f"{label} is not valid JSON") from exc
    if not isinstance(parsed, dict):
        raise SequenceError(f"{label} must be a JSON object")
    return parsed


def _validate_receipt(receipt: Mapping[str, object], target: Path) -> dict[str, Any]:
    if not isinstance(receipt, Mapping) or set(receipt) != _RECEIPT_FIELDS:
        raise SequenceError("Owned sequence plan receipt has an invalid shape")
    if type(receipt["schema_version"]) is not int or receipt["schema_version"] != _PLAN_SCHEMA_VERSION:
        raise SequenceError("Unsupported owned sequence plan receipt schema")
    if receipt["kind"] != _OWNED_PLAN_KIND:
        raise SequenceError("Owned sequence plan receipt has an unexpected kind")
    relative_path = receipt["relative_path"]
    if (
        not isinstance(relative_path, str)
        or not relative_path
        or relative_path in {".", ".."}
        or "/" in relative_path
        or "\\" in relative_path
        or relative_path != target.name
    ):
        raise SequenceError("Owned sequence plan receipt path does not match the artifact filename")
    for field_name in ("file_sha256", "manifest_sha256"):
        value = receipt[field_name]
        if not isinstance(value, str) or not _SHA256_PATTERN.fullmatch(value):
            raise SequenceError(f"Owned sequence plan receipt has an invalid {field_name}")
    manifest = _json_object(receipt["manifest"], "owned sequence plan manifest")
    if _json_digest(manifest) != receipt["manifest_sha256"]:
        raise SequenceError("Owned sequence plan manifest checksum does not match its receipt")
    return manifest


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
