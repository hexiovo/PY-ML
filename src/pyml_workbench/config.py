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


@dataclass(frozen=True)
class SplitConfig:
    seed: int = 42
    train_fraction: float = 0.6
    validation_fraction: float = 0.2
    test_fraction: float = 0.2

    def validate(self) -> None:
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
class ExperimentConfig:
    dataset: DatasetConfig
    task: str
    model_id: str
    parameters: dict[str, Any] = field(default_factory=dict)
    split: SplitConfig = field(default_factory=SplitConfig)
    output_dir: str | None = None
    sequence: SequenceConfig | None = None

    def validate(self) -> None:
        if self.task not in _SUPPORTED_TASKS:
            raise ConfigError(f"Unsupported task: {self.task}")
        if not self.model_id.strip():
            raise ConfigError("model_id cannot be empty")
        self.dataset.validate(supervised=self.task in {"classification", "regression"})
        self.split.validate()
        if not isinstance(self.parameters, dict):
            raise ConfigError("parameters must be a JSON object")
        if self.sequence is not None:
            if self.model_id not in {"H01", "H02", "H03", "N04", "N06"}:
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
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ConfigError(f"Invalid experiment configuration: {exc}") from exc
        config.validate()
        return config
