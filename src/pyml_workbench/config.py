"""Stable JSON-serializable configuration contracts for one experiment."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .sequence import SequenceConfig, SequenceError

_SUPPORTED_TASKS = {
    "classification", "regression", "clustering", "dimensionality reduction", "anomaly detection",
    "sequence_modeling",
}
_SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".xls"}


class ConfigError(ValueError):
    """Raised when an experiment configuration violates the v0.1 contract."""


@dataclass(frozen=True)
class DatasetConfig:
    source_path: str
    sheet_name: str | None = None
    target_column: str | None = None
    feature_columns: tuple[str, ...] | None = None
    reference_labels_column: str | None = None

    def validate(self, *, supervised: bool) -> None:
        if not self.source_path.strip():
            raise ConfigError("source_path cannot be empty")
        suffix = Path(self.source_path).suffix.lower()
        if suffix not in _SUPPORTED_EXTENSIONS:
            raise ConfigError(f"Unsupported file type {suffix!r}; use CSV, XLSX, or XLS")
        if supervised and not self.target_column:
            raise ConfigError("A target_column is required for supervised tasks")
        if self.feature_columns is not None and not self.feature_columns:
            raise ConfigError("feature_columns cannot be empty when explicitly provided")
        if self.feature_columns and len(set(self.feature_columns)) != len(self.feature_columns):
            raise ConfigError("feature_columns must be unique")
        if self.target_column and self.feature_columns and self.target_column in self.feature_columns:
            raise ConfigError("target_column cannot also appear in feature_columns")
        if self.reference_labels_column and self.reference_labels_column in (self.feature_columns or ()):
            raise ConfigError("Reference labels must be excluded from model features")


@dataclass(frozen=True)
class SplitConfig:
    seed: int = 42
    train_fraction: float = 0.6
    validation_fraction: float = 0.2
    test_fraction: float = 0.2
    strategy: str = "random"
    group_column: str | None = None
    time_column: str | None = None

    def validate(self) -> None:
        if self.strategy not in {"random", "group", "time"}:
            raise ConfigError("split.strategy must be random, group, or time")
        if self.strategy == "group" and not self.group_column:
            raise ConfigError("Group splitting requires group_column")
        if self.group_column and self.strategy != "group":
            raise ConfigError("group_column requires strategy='group'")
        if self.time_column and self.strategy != "time":
            raise ConfigError("time_column requires strategy='time'")
        if self.seed < 0:
            raise ConfigError("seed must be a non-negative integer")
        ratios = (self.train_fraction, self.validation_fraction, self.test_fraction)
        if any(value <= 0 or value >= 1 for value in ratios):
            raise ConfigError("train/validation/test fractions must be between 0 and 1")
        if abs(sum(ratios) - 1.0) > 1e-9:
            raise ConfigError("train/validation/test fractions must sum to 1")
        approved = (0.6, 0.2, 0.2)
        if any(abs(actual - expected) > 1e-9 for actual, expected in zip(ratios, approved)):
            raise ConfigError("v0.1 uses the approved 60/20/20 split")


@dataclass(frozen=True)
class EvaluationConfig:
    folds: int = 5
    strategy: str = "auto"
    gap: int = 0

    def validate(self):
        if isinstance(self.folds, bool) or not isinstance(self.folds, int) or not 2 <= self.folds <= 10:
            raise ConfigError("Cross-validation folds must be an integer between 2 and 10")
        if self.strategy not in {"auto", "stratified", "kfold", "group", "time"}:
            raise ConfigError("Unknown cross-validation strategy")
        if isinstance(self.gap, bool) or not isinstance(self.gap, int) or self.gap < 0:
            raise ConfigError("Cross-validation gap must be a nonnegative integer")


@dataclass(frozen=True)
class ExperimentConfig:
    dataset: DatasetConfig
    task: str
    model_id: str
    parameters: dict[str, Any] = field(default_factory=dict)
    split: SplitConfig = field(default_factory=SplitConfig)
    output_dir: str | None = None
    sequence: SequenceConfig | None = None
    evaluation: EvaluationConfig | None = None

    def validate(self) -> None:
        if self.task not in _SUPPORTED_TASKS:
            raise ConfigError(f"Unsupported task: {self.task}")
        if not self.model_id.strip():
            raise ConfigError("model_id cannot be empty")
        self.dataset.validate(supervised=self.task in {"classification", "regression"})
        self.split.validate()
        if self.model_id in {"H01", "H02", "H03", "N04", "N06", "N07", "N08", "N09"} and self.split.strategy != "random":
            raise ConfigError("SequencePlan models configure group/time ordering through sequence, not split.strategy")
        if not isinstance(self.parameters, dict):
            raise ConfigError("parameters must be a JSON object")
        structural = {name for name in (self.split.group_column, self.split.time_column) if name}
        if structural.intersection(self.dataset.feature_columns or ()) or self.dataset.target_column in structural:
            raise ConfigError("Split group/time columns must be excluded from features and target")
        if self.evaluation is not None:
            self.evaluation.validate()
            if self.task not in {"classification", "regression"} or self.model_id.startswith(("H", "N")):
                raise ConfigError("Fold evaluation supports ordinary tabular classifiers/regressors and T01–T05; sequence neural models use their SequencePlan validation")
            if self.evaluation.strategy == "group" and self.split.strategy != "group":
                raise ConfigError("Group cross-validation requires a group holdout split")
            if self.evaluation.strategy == "time" and self.split.strategy != "time":
                raise ConfigError("Rolling cross-validation requires a chronological holdout split")
            if self.split.strategy in {"time", "group"} and self.evaluation.strategy not in {"auto", self.split.strategy}:
                raise ConfigError("Cross-validation must preserve the configured group/time isolation")
        if self.model_id.startswith("T") and self.split.strategy != "time":
            raise ConfigError("Forecasting models require split.strategy='time'")
        if self.sequence is not None:
            if self.model_id not in {"H01", "H02", "H03", "N04", "N06", "N07", "N08", "N09"}:
                raise ConfigError("sequence settings apply only to HMM and window-regression models")
            try:
                SequenceConfig.from_dict(self.sequence).validate(
                    task=self.task,
                    model_id=self.model_id if self.dataset.feature_columns is not None else None,
                    feature_columns=self.dataset.feature_columns,
                    target_column=self.dataset.target_column,
                )
            except SequenceError as exc:
                raise ConfigError(str(exc)) from exc

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        payload = asdict(self)
        if self.dataset.reference_labels_column is None:
            payload["dataset"].pop("reference_labels_column", None)
        if self.evaluation is None:
            payload.pop("evaluation", None)
        if self.split.strategy == "random" and self.split.group_column is None and self.split.time_column is None:
            for key in ("strategy", "group_column", "time_column"):
                payload["split"].pop(key)
        if self.sequence is None:
            # Keep the canonical form of existing configurations byte-for-byte
            # stable; old sessions and search receipts must retain their hashes.
            payload.pop("sequence", None)
        else:
            payload["sequence"] = SequenceConfig.from_dict(self.sequence).to_dict()
        if self.dataset.feature_columns is not None:
            payload["dataset"]["feature_columns"] = list(self.dataset.feature_columns)
        return payload

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ExperimentConfig":
        if not isinstance(payload, dict):
            raise ConfigError("configuration must be a JSON object")
        try:
            dataset_payload = dict(payload["dataset"])
            features = dataset_payload.get("feature_columns")
            if features is not None:
                dataset_payload["feature_columns"] = tuple(features)
            split_payload = payload.get("split", {})
            config = cls(
                dataset=DatasetConfig(**dataset_payload),
                task=payload["task"],
                model_id=payload["model_id"],
                parameters=dict(payload.get("parameters", {})),
                split=SplitConfig(**split_payload),
                output_dir=payload.get("output_dir"),
                sequence=(SequenceConfig.from_dict(payload["sequence"]) if payload.get("sequence") is not None else None),
                evaluation=(EvaluationConfig(**payload["evaluation"]) if payload.get("evaluation") is not None else None),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ConfigError(f"Invalid experiment configuration: {exc}") from exc
        config.validate()
        return config
