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
SUPPORTED_REDUCTION_METHODS = (
    "pca",
    "truncated_svd",
    "fast_ica",
    "gaussian_random_projection",
    "kernel_pca",
)
_REDUCTION_UNSUPPORTED_MODEL_IDS = {"H01", "H02", "H03", "N04", "N06"}
_REDUCTION_NONNEGATIVE_MODEL_IDS = {"C10", "C12", "C13"}


class ConfigError(ValueError):
    """Raised when an experiment configuration violates the v0.1 contract."""


def reduction_unsupported_reason(task: str, model_id: str, method: str) -> str | None:
    """Return why preprocessing dimensionality reduction is incompatible."""
    method_label = "PCA preprocessing" if method == "pca" else "Pre-model dimensionality reduction"
    if task == "dimensionality reduction":
        return f"{method_label} cannot be combined with the standalone dimensionality-reduction task"
    if task == "sequence_modeling" or model_id in _REDUCTION_UNSUPPORTED_MODEL_IDS:
        if method == "pca":
            return "PCA preprocessing is not supported for HMM or window-regression models"
        return "Pre-model dimensionality reduction is not supported for sequence, HMM, or window-regression workflows"
    if model_id in _REDUCTION_NONNEGATIVE_MODEL_IDS:
        if model_id == "C13":
            if method == "pca":
                return (
                    "PCA produces continuous signed components; CategoricalNB requires non-negative integer category IDs"
                )
            return (
                "Dimensionality reduction produces continuous component values; CategoricalNB requires non-negative integer category IDs"
            )
        if method == "pca":
            return (
                "PCA can produce negative components; MultinomialNB and ComplementNB require non-negative inputs"
            )
        return (
            "Dimensionality reduction can produce negative components; MultinomialNB and ComplementNB require non-negative inputs"
        )
    return None


def pca_unsupported_reason(task: str, model_id: str) -> str | None:
    """Return why legacy PCA preprocessing is incompatible with a task/model pair."""
    return reduction_unsupported_reason(task, model_id, "pca")


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
    strategy: str = "holdout"
    n_splits: int = 5

    def validate(self) -> None:
        if type(self.seed) is not int or not 0 <= self.seed < 2 ** 32:
            raise ConfigError("seed must be a non-negative integer")
        if self.strategy not in {"holdout", "kfold"}:
            raise ConfigError("split strategy must be holdout or kfold")
        if type(self.n_splits) is not int or not 2 <= self.n_splits <= 10:
            raise ConfigError("K-fold count must be an integer from 2 through 10")
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
    pca_components: int | None = None
    reduction_method: str | None = None
    reduction_components: int | None = None

    @property
    def seed(self) -> int:
        """Expose the experiment seed to keyword-only preprocessing factories."""
        return self.split.seed

    def validate(self) -> None:
        if self.task not in _SUPPORTED_TASKS:
            raise ConfigError(f"Unsupported task: {self.task}")
        if not self.model_id.strip():
            raise ConfigError("model_id cannot be empty")
        self.dataset.validate(supervised=self.task in {"classification", "regression"})
        self.split.validate()
        if self.split.strategy == "kfold":
            reason = cv_unsupported_reason(self.task, self.model_id)
            if reason is not None:
                raise ConfigError(reason)
        if not isinstance(self.parameters, dict):
            raise ConfigError("parameters must be a JSON object")
        if self.pca_components is not None and (
            type(self.pca_components) is not int or self.pca_components < 1
        ):
            raise ConfigError("pca_components must be a positive integer or None")
        has_method = self.reduction_method is not None
        has_components = self.reduction_components is not None
        if has_method != has_components:
            raise ConfigError(
                "reduction_method and reduction_components must be provided together"
            )
        if has_method:
            if self.reduction_method not in SUPPORTED_REDUCTION_METHODS:
                raise ConfigError(
                    f"Unsupported reduction_method: {self.reduction_method!r}; "
                    f"choose one of {', '.join(SUPPORTED_REDUCTION_METHODS)}"
                )
            if type(self.reduction_components) is not int or self.reduction_components < 1:
                raise ConfigError("reduction_components must be a positive integer")
            if self.pca_components is not None and (
                self.reduction_method != "pca"
                or self.reduction_components != self.pca_components
            ):
                raise ConfigError(
                    "pca_components conflicts with reduction_method/reduction_components"
                )
        method = self.reduction_method
        if method is None and self.pca_components is not None:
            method = "pca"
        if method is not None:
            reason = reduction_unsupported_reason(self.task, self.model_id, method)
            if reason is not None:
                raise ConfigError(reason)
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
        if self.split.strategy == "holdout":
            # Existing holdout configurations retain their canonical identity.
            payload["split"].pop("strategy", None)
            payload["split"].pop("n_splits", None)
        if self.sequence is None:
            # Keep the canonical form of existing configurations byte-for-byte
            # stable; old sessions and search receipts must retain their hashes.
            payload.pop("sequence", None)
        else:
            payload["sequence"] = SequenceConfig.from_dict(self.sequence).to_dict()
        if self.pca_components is None:
            # Keep the canonical form of existing configurations byte-for-byte
            # stable while making an enabled PCA setting part of the config hash.
            payload.pop("pca_components", None)
        if self.reduction_method is None:
            payload.pop("reduction_method", None)
        if self.reduction_components is None:
            payload.pop("reduction_components", None)
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
                pca_components=payload.get("pca_components"),
                reduction_method=payload.get("reduction_method"),
                reduction_components=payload.get("reduction_components"),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ConfigError(f"Invalid experiment configuration: {exc}") from exc
        config.validate()
        return config


def cv_unsupported_reason(task: str, model_id: str) -> str | None:
    """Describe workflows whose validation/sequence protocol requires holdout."""
    if model_id in {"N01", "N02"}:
        return "N01/N02 use external validation for early stopping; choose the 60/20/20 holdout strategy."
    if task not in {"classification", "regression"} or model_id in {"H01", "H02", "H03", "N04", "N06"}:
        return "K-fold is available for ordinary tabular classification/regression only; choose holdout for sequence, HMM, window or unsupervised workflows."
    return None
