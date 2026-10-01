"""Single-model train/validation/frozen-test runs and portable fitted artifacts."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.manifold import trustworthiness

from .adapters import UnsupportedOperationError
from .catalog import build_estimator, get_model, list_models, model_capabilities
from .config import ConfigError, ExperimentConfig
from .data import DatasetError, LoadedDataset, load_dataset, select_features_target
from .preprocessing import build_preprocessor
from .sequence import SequenceConfig, build_sequence_plan

_EXTENDED_MODEL_IDS = {"H01", "H02", "H03", "N01", "N02", "N04", "N06"}


class ExperimentError(RuntimeError):
    """Raised when an experiment cannot be run with its selected model/config."""


class SplitError(ExperimentError):
    """Raised when the approved 60/20/20 split cannot be formed safely."""


class ArtifactError(ExperimentError):
    """Raised when result artifacts would overwrite source or existing files."""


@dataclass
class ExperimentResult:
    config: dict[str, Any]
    metrics: dict[str, Any]
    audit: dict[str, Any]
    results: pd.DataFrame
    model: "FittedModel"
    artifact_paths: dict[str, str]


@dataclass
class ExperimentSession:
    """In-memory fitted validation session waiting for its one final test."""

    config: ExperimentConfig
    frozen_config: dict[str, Any]
    frozen_config_sha256: str
    source_path: str
    features: pd.DataFrame
    target: pd.Series | None
    splits: dict[str, np.ndarray]
    fitted_model: "FittedModel"
    metrics: dict[str, Any]
    rows: list[dict[str, Any]]
    events: list[str]
    test_evaluation_count: int = 0
    finalized: bool = False
    result: ExperimentResult | None = None
    frozen: bool = False
    snapshot_manifest: dict[str, Any] = field(default_factory=dict)
    fit_scope: str = "train"


def _json_digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def _table_digest(frame: pd.DataFrame | pd.Series | None) -> str | None:
    if frame is None:
        return None
    values = pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes()
    description = str(list(frame.dtypes.astype(str))) if isinstance(frame, pd.DataFrame) else str(frame.dtype)
    columns = str(list(frame.columns)) if isinstance(frame, pd.DataFrame) else str(frame.name)
    return hashlib.sha256(values + description.encode("utf-8") + columns.encode("utf-8")).hexdigest()


@dataclass
class DatasetSnapshot:
    """Owned local data and immutable fingerprints shared by all search trials."""

    features: pd.DataFrame
    target: pd.Series | None
    objective_labels: pd.Series | None
    splits: dict[str, np.ndarray]
    source_path: str
    manifest: dict[str, Any]

    def verify(self, config: ExperimentConfig | None = None) -> None:
        observed = {"features": _table_digest(self.features), "target": _table_digest(self.target), "objective_labels": _table_digest(self.objective_labels)}
        if _json_digest(observed) != self.manifest["data_sha256"]:
            raise ExperimentError("Dataset snapshot was modified")
        positions = {name: [int(item) for item in values] for name, values in self.splits.items()}
        if _json_digest(positions) != self.manifest["split_sha256"] or positions != self.manifest["split_positions"]:
            raise ExperimentError("Snapshot split was modified")
        if config is not None:
            requested = config.to_dict()
            if requested["dataset"] != self.manifest["dataset_config"] or requested["split"] != self.manifest["split_config"] or config.task != self.manifest["task"]:
                raise ExperimentError("Configuration does not belong to this dataset/split snapshot")

    def to_dict(self) -> dict[str, Any]:
        return json.loads(json.dumps(self.manifest, ensure_ascii=False))


def build_snapshot(config: ExperimentConfig | dict[str, Any], *, objective_labels_column: str | None = None) -> DatasetSnapshot:
    config = ExperimentConfig.from_dict(config) if isinstance(config, dict) else config
    config.validate()
    before_sha = hashlib.sha256(Path(config.dataset.source_path).read_bytes()).hexdigest()
    dataset = load_dataset(config.dataset)
    source_sha = hashlib.sha256(Path(dataset.source_path).read_bytes()).hexdigest()
    if source_sha != before_sha:
        raise DatasetError("Source changed while loading; retry with a stable dataset")
    return _build_snapshot_from_frame(
        config,
        dataset.frame,
        source_path=dataset.source_path,
        source_sha256=source_sha,
        splits=None,
        objective_labels_column=objective_labels_column,
    )


def _build_snapshot_from_frame(
    config: ExperimentConfig | dict[str, Any],
    frame: pd.DataFrame,
    *,
    source_path: str | Path,
    source_sha256: str,
    splits: dict[str, np.ndarray] | None,
    objective_labels_column: str | None = None,
) -> DatasetSnapshot:
    """Build the core snapshot from an already-loaded table and approved positions.

    Sequence workflows use this entry point to make their tabular snapshot and
    SequencePlan from the same single table read and split assignment.
    """
    config = ExperimentConfig.from_dict(config) if isinstance(config, dict) else config
    config.validate()
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame")
    if not isinstance(source_sha256, str) or len(source_sha256) != 64 or any(ch not in "0123456789abcdef" for ch in source_sha256):
        raise ValueError("source_sha256 must be a lowercase 64-character SHA-256 digest")
    source_path = str(Path(source_path).expanduser().resolve())

    selected = config.dataset.feature_columns
    if selected is None and config.model_id in {"H01", "H02", "H03", "N04", "N06"}:
        sequence = SequenceConfig.from_dict(config.sequence) if config.sequence is not None else SequenceConfig()
        structural = {value for value in (config.dataset.target_column, sequence.group_column, sequence.time_column) if value is not None}
        columns = sequence.observation_columns if config.model_id in {"H01", "H02", "H03"} and sequence.observation_columns else frame.columns
        selected = tuple(name for name in columns if name not in structural)
    if objective_labels_column:
        if objective_labels_column not in frame:
            raise ConfigError("Objective label column does not exist")
        if selected is not None and objective_labels_column in selected:
            raise ConfigError("Objective label column must be excluded from features")
        if selected is None:
            selected = tuple(name for name in frame.columns if name not in {config.dataset.target_column, objective_labels_column})
    loaded = LoadedDataset(source_path, config.dataset.sheet_name, (), frame)
    features, target = select_features_target(loaded, task=config.task, target_column=config.dataset.target_column, feature_columns=selected)
    labels = frame[objective_labels_column].copy(deep=True) if objective_labels_column else None
    if labels is not None and labels.isna().any():
        raise ConfigError("Objective labels contain missing values")
    if splits is None:
        normalized_splits = _split_rows(target, len(features), config.split.seed, stratified=config.task == "classification")
    else:
        if not isinstance(splits, dict) or set(splits) != {"train", "validation", "test"}:
            raise ExperimentError("Snapshot splits must contain train, validation, and test positions")
        normalized_splits = {}
        for name in ("train", "validation", "test"):
            values = np.asarray(splits[name], dtype=np.int64).copy()
            if values.ndim != 1 or values.size == 0 or np.any(values < 0) or np.any(values >= len(features)):
                raise ExperimentError(f"Snapshot split {name!r} contains invalid row positions")
            if len(np.unique(values)) != len(values):
                raise ExperimentError(f"Snapshot split {name!r} contains duplicate row positions")
            normalized_splits[name] = values
        joined = np.concatenate(list(normalized_splits.values()))
        if len(joined) != len(features) or len(np.unique(joined)) != len(features):
            raise ExperimentError("Snapshot splits must partition every source row exactly once")
    positions = {name: [int(item) for item in values] for name, values in normalized_splits.items()}
    payload = config.to_dict()
    manifest = {
        "schema_version": 1, "source_path": source_path,
        "source_sha256": source_sha256,
        "data_sha256": _json_digest({"features": _table_digest(features), "target": _table_digest(target), "objective_labels": _table_digest(labels)}),
        "split_sha256": _json_digest(positions), "split_positions": positions,
        "row_count": len(features), "feature_columns": list(features.columns),
        "task": config.task, "dataset_config": payload["dataset"], "split_config": payload["split"],
        "objective_labels_column": objective_labels_column,
    }
    return DatasetSnapshot(features.copy(deep=True), target.copy(deep=True) if target is not None else None, labels, normalized_splits, source_path, manifest)


def build_extended_snapshot(
    config: ExperimentConfig | dict[str, Any],
    *,
    objective_labels_column: str | None = None,
) -> tuple[DatasetSnapshot, Any | None]:
    """Load an extended-model source once and return its snapshot and optional plan."""
    config = ExperimentConfig.from_dict(config) if isinstance(config, dict) else config
    config.validate()
    before_sha = hashlib.sha256(Path(config.dataset.source_path).read_bytes()).hexdigest()
    dataset = load_dataset(config.dataset)
    source_sha = hashlib.sha256(Path(dataset.source_path).read_bytes()).hexdigest()
    if source_sha != before_sha:
        raise DatasetError("Source changed while loading; retry with a stable dataset")
    sequence_ids = {"H01", "H02", "H03", "N04", "N06"}
    if config.model_id not in sequence_ids:
        return (
            _build_snapshot_from_frame(
                config,
                dataset.frame,
                source_path=dataset.source_path,
                source_sha256=source_sha,
                splits=None,
                objective_labels_column=objective_labels_column,
            ),
            None,
        )
    sequence = SequenceConfig.from_dict(config.sequence) if config.sequence is not None else SequenceConfig()
    plan = build_sequence_plan(
        dataset.frame,
        sequence,
        task=config.task,
        model_id=config.model_id,
        target_column=config.dataset.target_column,
        feature_columns=config.dataset.feature_columns,
        split_seed=config.split.seed,
        source_path=dataset.source_path,
        source_sha256=source_sha,
    )
    snapshot = _build_snapshot_from_frame(
        config,
        dataset.frame,
        source_path=dataset.source_path,
        source_sha256=source_sha,
        splits=plan.split_positions,
        objective_labels_column=objective_labels_column,
    )
    snapshot.manifest["sequence_plan_sha256"] = plan.plan_sha256
    plan.verify(config, snapshot)
    return snapshot, plan


class FittedModel:
    """Self-contained preprocessing + estimator bundle with capability guards."""

    def __init__(
        self,
        *,
        model_id: str,
        task: str,
        estimator: Any,
        preprocessor: Any,
        feature_columns: list[Any],
        preprocessing: dict[str, Any],
        config: dict[str, Any],
        frozen_config_sha256: str,
    ):
        self.model_id = model_id
        self.task = task
        self.estimator = estimator
        self.preprocessor = preprocessor
        self.feature_columns = list(feature_columns)
        self.preprocessing = dict(preprocessing)
        encoded_config = json.dumps(
            config,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        calculated_sha = hashlib.sha256(encoded_config.encode("utf-8")).hexdigest()
        if calculated_sha != frozen_config_sha256:
            raise ExperimentError("Fitted model config does not match its frozen configuration hash")
        # Keep the authoritative snapshot as an immutable JSON string. Callers receive
        # a fresh object so editing returned metadata cannot relabel the fitted model.
        self._config_json = encoded_config
        self._frozen_config_sha256 = frozen_config_sha256
        self.capabilities = model_capabilities(model_id, estimator)

    @property
    def config(self) -> dict[str, Any]:
        return json.loads(self._config_json)

    @property
    def frozen_config_sha256(self) -> str:
        return self._frozen_config_sha256

    def _feature_frame(self, values: Any) -> pd.DataFrame:
        if isinstance(values, pd.DataFrame):
            frame = values.copy(deep=False)
        elif isinstance(values, dict):
            frame = pd.DataFrame([values])
        else:
            if self.preprocessing.get("categorical_columns"):
                raise DatasetError("Categorical models require a DataFrame with the fitted feature names")
            array = np.asarray(values)
            if array.ndim == 1:
                array = array.reshape(1, -1)
            if array.ndim != 2 or array.shape[1] != len(self.feature_columns):
                raise DatasetError(
                    f"Expected {len(self.feature_columns)} numeric feature column(s), received shape {array.shape}"
                )
            frame = pd.DataFrame(array, columns=self.feature_columns)
        missing = [column for column in self.feature_columns if column not in frame.columns]
        if missing:
            raise DatasetError(f"Prediction data is missing fitted feature column(s): {missing}")
        return frame.loc[:, self.feature_columns]

    def _transform_input(self, values: Any):
        frame = self._feature_frame(values)
        return self.preprocessor.transform(frame)

    def _invoke(self, operation: str, values: Any):
        if not self.capabilities.get(operation, False):
            raise UnsupportedOperationError(
                f"{self.model_id} does not support {operation} for its fitted configuration"
            )
        if self.model_id in _EXTENDED_MODEL_IDS:
            from .extended_experiment import predict_extended

            return predict_extended(self, values, operation=operation)
        transformed = self._transform_input(values)
        return getattr(self.estimator, operation)(transformed)

    def predict(self, values: Any):
        return self._invoke("predict", values)

    def predict_proba(self, values: Any):
        return self._invoke("predict_proba", values)

    def decision_function(self, values: Any):
        return self._invoke("decision_function", values)

    def score_samples(self, values: Any):
        return self._invoke("score_samples", values)

    def transform(self, values: Any):
        return self._invoke("transform", values)

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "model_id": self.model_id,
            "task": self.task,
            "feature_columns": [str(column) for column in self.feature_columns],
            "capabilities": dict(self.capabilities),
            "preprocessing": self.preprocessing,
            "frozen_config_sha256": self.frozen_config_sha256,
            "serialization": "joblib; includes fitted preprocessing and estimator, no source dataset",
        }


def _canonical_config(config: ExperimentConfig) -> tuple[dict[str, Any], str]:
    try:
        encoded = json.dumps(config.to_dict(), ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"Experiment configuration must contain only finite JSON values: {exc}") from exc
    return json.loads(encoded), hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _split_rows(target: pd.Series | None, row_count: int, seed: int, *, stratified: bool):
    if row_count < 10:
        raise SplitError("At least 10 rows are needed to create meaningful 60/20/20 train/validation/test partitions")
    positions = np.arange(row_count, dtype=np.int64)
    stratify_values = None
    if stratified:
        assert target is not None
        counts = target.value_counts(dropna=False)
        small = {str(label): int(count) for label, count in counts.items() if count < 3}
        if small:
            raise SplitError(
                "Stratified 60/20/20 split needs at least 3 rows per target class; "
                f"classes below that minimum: {small}. Add examples or reduce the number of classes."
            )
        stratify_values = target.to_numpy()
    try:
        train, remainder = train_test_split(
            positions,
            test_size=0.4,
            random_state=seed,
            shuffle=True,
            stratify=stratify_values,
        )
        remainder_strata = stratify_values[remainder] if stratify_values is not None else None
        validation, test = train_test_split(
            remainder,
            test_size=0.5,
            random_state=seed,
            shuffle=True,
            stratify=remainder_strata,
        )
    except ValueError as exc:
        if stratified:
            raise SplitError(
                "Could not form a stratified 60/20/20 split with every class represented in all partitions; "
                "add rows per class or simplify the target labels. "
                f"scikit-learn detail: {exc}"
            ) from exc
        raise SplitError(f"Could not form a 60/20/20 split: {exc}") from exc
    parts = {"train": train, "validation": validation, "test": test}
    if any(len(values) == 0 for values in parts.values()):
        raise SplitError("The dataset is too small for three non-empty 60/20/20 partitions")
    if stratified:
        expected = set(target.unique())
        for split_name, values in parts.items():
            observed = set(target.iloc[values].unique())
            if observed != expected:
                raise SplitError(
                    f"Split {split_name!r} does not contain every target class; add rows per class and retry"
                )
    return parts


def _target_for_task(target: pd.Series | None, task: str, estimator: Any) -> pd.Series | None:
    if target is None:
        return None
    if task == "classification":
        return target
    try:
        numeric = pd.to_numeric(target, errors="raise").astype(float)
    except (TypeError, ValueError) as exc:
        raise DatasetError("Regression target must contain numeric values") from exc
    if not np.isfinite(numeric.to_numpy()).all():
        raise DatasetError("Regression target must contain only finite numeric values")
    class_name = estimator.__class__.__name__
    if class_name == "PoissonRegressor" and (numeric < 0).any():
        raise DatasetError("PoissonRegressor requires a non-negative target")
    if class_name == "GammaRegressor" and (numeric <= 0).any():
        raise DatasetError("GammaRegressor requires a strictly positive target")
    if class_name == "TweedieRegressor":
        power = float(estimator.power)
        if power == 1 and (numeric < 0).any():
            raise DatasetError("TweedieRegressor with power=1 requires a non-negative target")
        if power >= 2 and (numeric <= 0).any():
            raise DatasetError(f"TweedieRegressor with power={power:g} requires a strictly positive target")
    return numeric


def _classification_metrics(actual, predicted) -> dict[str, Any]:
    return {
        "sample_count": int(len(actual)),
        "accuracy": float(accuracy_score(actual, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(actual, predicted)),
        "precision_macro": float(precision_score(actual, predicted, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(actual, predicted, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(actual, predicted, average="macro", zero_division=0)),
    }


def _regression_metrics(actual, predicted) -> dict[str, Any]:
    return {
        "sample_count": int(len(actual)),
        "r2": float(r2_score(actual, predicted)),
        "mae": float(mean_absolute_error(actual, predicted)),
        "rmse": float(np.sqrt(mean_squared_error(actual, predicted))),
    }


def _cluster_metrics(values, labels) -> dict[str, Any]:
    labels = np.asarray(labels)
    result: dict[str, Any] = {
        "sample_count": int(len(labels)),
        "cluster_count": int(len(np.unique(labels))),
        "noise_count": int(np.sum(labels == -1)),
        "metric_family": "unsupervised cluster geometry; no accuracy or target labels used",
        "silhouette": None,
        "calinski_harabasz": None,
        "davies_bouldin": None,
    }
    unique = np.unique(labels)
    if 1 < len(unique) < len(labels):
        try:
            result["silhouette"] = float(silhouette_score(values, labels))
            result["calinski_harabasz"] = float(calinski_harabasz_score(values, labels))
            result["davies_bouldin"] = float(davies_bouldin_score(values, labels))
        except ValueError:
            pass
    else:
        result["metric_note"] = "Silhouette/CH/DB are undefined with fewer than 2 clusters or one label per row"
    return result


def _outlier_metrics(labels, scores=None) -> dict[str, Any]:
    labels = np.asarray(labels)
    outliers = int(np.sum(labels == -1))
    result = {
        "sample_count": int(len(labels)),
        "outlier_count": outliers,
        "outlier_fraction": float(outliers / len(labels)) if len(labels) else None,
        "metric_family": "unsupervised outlier rate; no accuracy or target labels used",
    }
    if scores is not None:
        result["mean_score_samples"] = float(np.mean(scores))
    return result


def _trustworthiness_metrics(original, embedding) -> dict[str, Any]:
    count = len(embedding)
    if count < 5:
        return {"sample_count": count, "trustworthiness": None, "metric_note": "At least 5 rows are needed"}
    neighbors = min(5, (count - 1) // 2)
    try:
        score = float(trustworthiness(original, embedding, n_neighbors=neighbors))
    except ValueError as exc:
        return {"sample_count": count, "trustworthiness": None, "metric_note": str(exc)}
    return {"sample_count": count, "trustworthiness": score, "n_neighbors": neighbors}


def _result_rows(split: str, row_positions, source_index, actual=None, predicted=None, values=None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for offset, position in enumerate(row_positions):
        row: dict[str, Any] = {
            "split": split,
            "row_position": int(position),
            "source_index": str(source_index[offset]),
        }
        if actual is not None:
            row["y_true"] = actual[offset].item() if hasattr(actual[offset], "item") else actual[offset]
        if predicted is not None:
            row["y_pred"] = predicted[offset].item() if hasattr(predicted[offset], "item") else predicted[offset]
        if values is not None:
            vector = np.asarray(values[offset]).reshape(-1)
            for component, value in enumerate(vector, start=1):
                row[f"component_{component}"] = float(value)
        rows.append(row)
    return rows


def _session_training_curves(session: ExperimentSession) -> dict[str, Any] | None:
    """Return the helper-owned, JSON-safe epoch history from an extended fit."""
    metadata = getattr(session, "extended_training_metadata", None)
    if not isinstance(metadata, dict):
        metadata = getattr(session, "training_metadata", None)
    if not isinstance(metadata, dict):
        return None
    curves = metadata.get("training_curves")
    if curves is None:
        return None
    if not isinstance(curves, dict):
        raise ExperimentError("Training curves must be a JSON object")
    try:
        return json.loads(json.dumps(curves, ensure_ascii=False, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ExperimentError("Training curves must contain only finite JSON values") from exc


def _write_artifacts(
    output_dir: str | Path,
    source_path: str,
    frozen_config: dict[str, Any],
    frozen_sha: str,
    metrics: dict[str, Any],
    results: pd.DataFrame,
    model: FittedModel,
    training_curves: dict[str, Any] | None = None,
) -> dict[str, str]:
    target = Path(output_dir).expanduser().resolve()
    source = Path(source_path).expanduser().resolve()
    if target == source or target == source.parent:
        raise ArtifactError("output_dir must be a separate directory; the input file and its parent are protected")
    if target.exists() and (not target.is_dir() or any(target.iterdir())):
        raise ArtifactError(f"output_dir already exists and is not empty: {target}")
    target.mkdir(parents=True, exist_ok=True)
    metrics_frame = pd.DataFrame([
        {"split": split, "metric": name, "value": value}
        for split, values in metrics.items()
        if isinstance(values, dict)
        for name, value in values.items()
        if isinstance(value, (str, int, float, bool)) or value is None
    ])
    config_payload = {"experiment": frozen_config, "frozen_config_sha256": frozen_sha}
    (target / "config.json").write_text(json.dumps(config_payload, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    (target / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    if training_curves is not None:
        (target / "training_curves.json").write_text(
            json.dumps(training_curves, ensure_ascii=False, indent=2, allow_nan=False),
            encoding="utf-8",
        )
    (target / "model_manifest.json").write_text(json.dumps(model.manifest(), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    model_path = target / "model.joblib"
    if model.model_id in _EXTENDED_MODEL_IDS:
        from .extended_experiment import export_extended_model

        export_extended_model(model, model_path)
    else:
        joblib.dump(model, model_path, compress=3)
    metrics_frame.to_csv(target / "metrics.csv", index=False, encoding="utf-8-sig")
    results.to_csv(target / "results.csv", index=False, encoding="utf-8-sig")
    with pd.ExcelWriter(target / "results.xlsx", engine="openpyxl") as writer:
        results.to_excel(writer, sheet_name="results", index=False)
    with pd.ExcelWriter(target / "metrics.xlsx", engine="openpyxl") as writer:
        metrics_frame.to_excel(writer, sheet_name="metrics", index=False)
    paths = {
        "output_dir": str(target),
        "config": str(target / "config.json"),
        "metrics_json": str(target / "metrics.json"),
        "metrics_csv": str(target / "metrics.csv"),
        "metrics_xlsx": str(target / "metrics.xlsx"),
        "results_csv": str(target / "results.csv"),
        "results_xlsx": str(target / "results.xlsx"),
        "model": str(target / "model.joblib"),
        "model_manifest": str(target / "model_manifest.json"),
    }
    if training_curves is not None:
        paths["training_curves"] = str(target / "training_curves.json")
    return paths


def prepare_experiment(
    config: ExperimentConfig | dict[str, Any],
    *,
    on_event: Callable[[str], None] | None = None,
    snapshot: DatasetSnapshot | None = None,
    _fit_scope: str = "train",
    sequence_plan: Any | None = None,
    selected_epochs: int | None = None,
    refit_provenance: dict[str, Any] | None = None,
) -> ExperimentSession:
    """Fit once and return a frozen validation session before touching the test partition."""
    if isinstance(config, dict):
        config = ExperimentConfig.from_dict(config)
    if not isinstance(config, ExperimentConfig):
        raise TypeError("config must be an ExperimentConfig or its JSON-compatible dictionary")
    config.validate()
    extended_model_ids = {"H01", "H02", "H03", "N01", "N02", "N04", "N06"}
    if config.model_id in extended_model_ids:
        from .extended_experiment import ExtendedExperimentSession, prepare_extended_experiment

        if _fit_scope not in {"train", "train_validation"}:
            raise ConfigError("Invalid fit scope")
        if snapshot is None:
            snapshot, sequence_plan = build_extended_snapshot(config)
        else:
            snapshot.verify(config)
        sequence_model_ids = {"H01", "H02", "H03", "N04", "N06"}
        if config.model_id in sequence_model_ids:
            if sequence_plan is None:
                raise ExperimentError("Sequence models require their owned SequencePlan with a supplied snapshot")
            sequence_plan.verify(config, snapshot)
        elif sequence_plan is not None:
            raise ExperimentError("Tabular deep models do not accept a SequencePlan")
        prefit_events = ["dataset_loaded", "target_validated", "split_created"]
        events: list[str] = []
        for name in prefit_events:
            events.append(name)
            if on_event is not None:
                on_event(name)
        session = prepare_extended_experiment(
            config,
            snapshot=snapshot,
            sequence_plan=sequence_plan,
            fit_scope=_fit_scope,
            selected_epochs=selected_epochs,
            refit_provenance=refit_provenance,
        )
        if not isinstance(session, ExtendedExperimentSession):
            raise ExperimentError("Extended model preparation returned an incompatible session")
        digest_config, digest = _canonical_config(config)
        if (session.config.to_dict() != digest_config
                or session.frozen_config != digest_config
                or session.frozen_config_sha256 != digest
                or session.fitted_model.config != digest_config
                or session.fitted_model.frozen_config_sha256 != digest
                or session.snapshot_manifest != snapshot.to_dict()
                or session.fit_scope != _fit_scope):
            raise ExperimentError("Extended session configuration, fit scope, or data snapshot does not match")
        if session.test_evaluation_count or session.finalized or session.result is not None or "test" in session.metrics:
            raise ExperimentError("Extended preparation accessed or finalized the test partition")
        if config.model_id in sequence_model_ids:
            if session.sequence_plan is None or session.sequence_plan.to_dict() != sequence_plan.to_dict():
                raise ExperimentError("Extended session does not own the approved SequencePlan")
            session.sequence_plan.verify(config, snapshot)
        # Lifecycle events belong to this core module, not to the model helper.
        if _fit_scope == "train_validation":
            postfit_events = [
                "preprocessor_fit_train_validation_only",
                "estimator_fit_train_validation_only",
                "refit_metrics_computed",
                "selection_frozen",
                "refit_train_validation_only",
            ]
        else:
            postfit_events = [
                "preprocessor_fit_train_only",
                "estimator_fit_train_only",
                "train_metrics_computed",
                "validation_metrics_computed",
                "selection_frozen",
            ]
        session.events = events + postfit_events
        for name in postfit_events:
            if on_event is not None:
                on_event(name)
        selection = dict(session.metrics.get("selection", {}))
        selection.update(
            mode="extended sequence/deep model; validation only before final freeze",
            frozen_model_id=config.model_id,
            frozen_parameters=config.parameters,
            frozen_config_sha256=digest,
        )
        session.metrics["selection"] = selection
        return session
    # Capture the exact user selection before constructing or fitting an estimator.
    # Rebuild the internal config from this deep JSON snapshot so nested parameter
    # dictionaries shared with a caller cannot change the label attached to the fit.
    frozen_config, frozen_sha = _canonical_config(config)
    config = ExperimentConfig.from_dict(frozen_config)
    spec = get_model(config.model_id)
    if spec["task"] != config.task:
        raise ConfigError(f"Model {config.model_id} belongs to task {spec['task']!r}, not {config.task!r}")
    if _fit_scope not in {"train", "train_validation"}:
        raise ConfigError("Invalid fit scope")
    if snapshot is not None:
        snapshot.verify(config)
        features = snapshot.features.copy(deep=True)
        target = snapshot.target.copy(deep=True) if snapshot.target is not None else None
        source_path = snapshot.source_path
    else:
        dataset = load_dataset(config.dataset)
        features, target = select_features_target(dataset, task=config.task, target_column=config.dataset.target_column, feature_columns=config.dataset.feature_columns)
        source_path = dataset.source_path
    estimator = build_estimator(config.model_id, config.parameters, seed=config.split.seed)
    target = _target_for_task(target, config.task, estimator)
    splits = {name: values.copy() for name, values in snapshot.splits.items()} if snapshot is not None else _split_rows(
        target,
        len(features),
        config.split.seed,
        stratified=config.task == "classification",
    )
    if _fit_scope == "train_validation":
        splits["train"] = np.concatenate([splits["train"], splits["validation"]])
        splits["validation"] = np.array([], dtype=np.int64)
    train_idx = splits["train"]
    validation_idx = splits["validation"]
    test_idx = splits["test"]
    x_train = features.iloc[train_idx].copy(deep=True)
    y_train = target.iloc[train_idx].copy(deep=True) if target is not None else None
    x_validation = features.iloc[validation_idx].copy(deep=True)
    y_validation = target.iloc[validation_idx].copy(deep=True) if target is not None else None
    preprocessor, preprocessing = build_preprocessor(features, config.model_id, config.parameters)
    events: list[str] = []

    def record_event(name: str) -> None:
        if _fit_scope == "train_validation":
            if name == "validation_metrics_computed":
                return
            name = {"preprocessor_fit_train_only": "preprocessor_fit_train_validation_only", "estimator_fit_train_only": "estimator_fit_train_validation_only", "train_metrics_computed": "refit_metrics_computed"}.get(name, name)
        events.append(name)
        if on_event is not None:
            on_event(name)

    for name in ("dataset_loaded", "target_validated", "split_created"):
        record_event(name)
    try:
        x_train_ready = preprocessor.fit_transform(x_train)
        record_event("preprocessor_fit_train_only")
        x_validation_ready = preprocessor.transform(x_validation) if len(validation_idx) else None
        if config.task in {"classification", "regression"}:
            estimator.fit(x_train_ready, y_train.to_numpy())
            record_event("estimator_fit_train_only")
        elif config.task == "clustering":
            capabilities = model_capabilities(config.model_id, estimator)
            if capabilities.get("fit_predict"):
                train_labels = estimator.fit_predict(x_train_ready)
            else:
                estimator.fit(x_train_ready)
                train_labels = getattr(estimator, "labels_", None)
            record_event("estimator_fit_train_only")
        elif config.task == "dimensionality reduction":
            capabilities = model_capabilities(config.model_id, estimator)
            if capabilities.get("fit_transform"):
                train_embedding = estimator.fit_transform(x_train_ready)
            else:
                estimator.fit(x_train_ready)
                train_embedding = estimator.transform(x_train_ready) if capabilities.get("transform") else None
            record_event("estimator_fit_train_only")
        else:
            capabilities = model_capabilities(config.model_id, estimator)
            if capabilities.get("predict"):
                estimator.fit(x_train_ready)
                train_labels = estimator.predict(x_train_ready)
            elif capabilities.get("fit_predict"):
                train_labels = estimator.fit_predict(x_train_ready)
            else:
                estimator.fit(x_train_ready)
                train_labels = None
            record_event("estimator_fit_train_only")
    except Exception as exc:
        raise ExperimentError(
            f"Model {config.model_id} ({spec['class_name']}) failed during train fit on {len(train_idx)} rows "
            f"with requested parameters {config.parameters!r}: {type(exc).__name__}: {exc}"
        ) from exc

    capabilities = model_capabilities(config.model_id, estimator)
    preprocessing["fit_row_count"] = int(len(train_idx))
    preprocessing["fit_scope"] = _fit_scope
    if snapshot is not None:
        preprocessing["data_sha256"] = snapshot.manifest["data_sha256"]
        preprocessing["split_sha256"] = snapshot.manifest["split_sha256"]
    fitted = FittedModel(
        model_id=config.model_id,
        task=config.task,
        estimator=estimator,
        preprocessor=preprocessor,
        feature_columns=list(features.columns),
        preprocessing=preprocessing,
        config=frozen_config,
        frozen_config_sha256=frozen_sha,
    )
    rows: list[dict[str, Any]] = []
    metrics: dict[str, Any] = {}
    test_evaluation_count = 0
    if config.model_id in {"H01", "H02", "H03", "N01", "N02", "N04", "N06"}:
        from .extended_experiment import ExtendedTestComputation, evaluate_extended_test

        if session.test_evaluation_count or "test" in metrics:
            raise ExperimentError("Extended test evaluation was already recorded")
        computation = evaluate_extended_test(session)
        if not isinstance(computation, ExtendedTestComputation):
            raise ExperimentError("Extended test evaluator returned an incompatible computation")
        if not isinstance(computation.metrics, dict) or not isinstance(computation.rows, list):
            raise ExperimentError("Extended test computation must contain metric and row collections")
        metrics["test"] = dict(computation.metrics)
        rows.extend(dict(row) for row in computation.rows)
        session.test_evaluation_count = 1
        record_event("test_metrics_computed_once")
    elif config.task in {"classification", "regression"}:
        if not capabilities.get("predict"):
            raise ExperimentError(f"Supervised model {config.model_id} does not expose predict after fit")
        train_prediction = estimator.predict(x_train_ready)
        validation_prediction = estimator.predict(x_validation_ready) if len(validation_idx) else None
        metric_fn = _classification_metrics if config.task == "classification" else _regression_metrics
        metrics["train"] = metric_fn(y_train.to_numpy(), train_prediction)
        if len(validation_idx):
            metrics["validation"] = metric_fn(y_validation.to_numpy(), validation_prediction)
        rows.extend(_result_rows("train", train_idx, features.index[train_idx], y_train.to_numpy(), train_prediction))
        if len(validation_idx):
            rows.extend(_result_rows("validation", validation_idx, features.index[validation_idx], y_validation.to_numpy(), validation_prediction))
        record_event("train_metrics_computed")
        record_event("validation_metrics_computed")
    elif config.task == "clustering":
        if train_labels is None:
            metrics["train"] = {"status": "unsupported", "reason": "estimator returned no cluster labels"}
        else:
            metrics["train"] = _cluster_metrics(x_train_ready, train_labels)
            rows.extend(_result_rows("train", train_idx, features.index[train_idx], predicted=train_labels))
        if len(validation_idx) and capabilities.get("predict"):
            validation_prediction = estimator.predict(x_validation_ready)
            metrics["validation"] = _cluster_metrics(x_validation_ready, validation_prediction)
            rows.extend(_result_rows("validation", validation_idx, features.index[validation_idx], predicted=validation_prediction))
        else:
            metrics["validation"] = {
                "status": "not_supported",
                "reason": "fitted estimator has no predict; no independent validation fit was substituted",
                "metric_family": "unsupervised; accuracy is not applicable",
            }
        record_event("validation_metrics_computed")
    elif config.task == "dimensionality reduction":
        if train_embedding is None:
            metrics["train"] = {"status": "unsupported", "reason": "estimator returned no embedding"}
        else:
            metrics["train"] = _trustworthiness_metrics(x_train_ready, train_embedding)
            rows.extend(_result_rows("train", train_idx, features.index[train_idx], values=train_embedding))
        if len(validation_idx) and capabilities.get("transform"):
            validation_embedding = estimator.transform(x_validation_ready)
            metrics["validation"] = _trustworthiness_metrics(x_validation_ready, validation_embedding)
            rows.extend(_result_rows("validation", validation_idx, features.index[validation_idx], values=validation_embedding))
        else:
            metrics["validation"] = {
                "status": "not_supported",
                "reason": "fitted estimator has no transform; no validation refit was substituted",
                "metric_family": "unsupervised embedding trustworthiness; accuracy is not applicable",
            }
        record_event("validation_metrics_computed")
    else:
        if train_labels is None:
            metrics["train"] = {"status": "unsupported", "reason": "estimator returned no anomaly labels"}
        else:
            scores = estimator.score_samples(x_train_ready) if capabilities.get("score_samples") else None
            metrics["train"] = _outlier_metrics(train_labels, scores)
            rows.extend(_result_rows("train", train_idx, features.index[train_idx], predicted=train_labels))
        if len(validation_idx) and capabilities.get("predict"):
            validation_prediction = estimator.predict(x_validation_ready)
            validation_scores = estimator.score_samples(x_validation_ready) if capabilities.get("score_samples") else None
            metrics["validation"] = _outlier_metrics(validation_prediction, validation_scores)
            rows.extend(_result_rows("validation", validation_idx, features.index[validation_idx], predicted=validation_prediction))
        else:
            metrics["validation"] = {
                "status": "not_supported",
                "reason": "fitted detector has no predict for new rows; fit_predict labels are train-only",
                "metric_family": "unsupervised outlier rate; accuracy is not applicable",
            }
        record_event("validation_metrics_computed")

    metrics["selection"] = {
        "mode": "single configured model; no parameter search",
        "frozen_model_id": config.model_id,
        "frozen_parameters": config.parameters,
        "frozen_config_sha256": frozen_sha,
    }
    record_event("selection_frozen")

    if _fit_scope == "train_validation":
        metrics["refit"] = metrics.pop("train")
        metrics.pop("validation", None)
        for row in rows:
            row["split"] = "refit"
        events = [item for item in events if item != "validation_metrics_computed"]
        record_event("refit_train_validation_only")

    return ExperimentSession(
        config=config,
        frozen_config=frozen_config,
        frozen_config_sha256=frozen_sha,
        source_path=source_path,
        features=features,
        target=target,
        splits=splits,
        fitted_model=fitted,
        metrics=metrics,
        rows=rows,
        events=events,
        test_evaluation_count=test_evaluation_count,
        snapshot_manifest=snapshot.to_dict() if snapshot is not None else {},
        fit_scope=_fit_scope,
    )


def freeze_experiment(session: ExperimentSession, expected_config: ExperimentConfig | dict[str, Any] | None = None) -> ExperimentSession:
    """Approve this fitted snapshot; repeated approval never resets test state."""
    if not isinstance(session, ExperimentSession):
        raise TypeError("session must be an ExperimentSession from prepare_experiment")
    snapshot, digest = _canonical_config(session.config)
    if (digest != session.frozen_config_sha256 or snapshot != session.frozen_config
            or session.fitted_model.config != snapshot
            or session.fitted_model.frozen_config_sha256 != digest):
        raise ExperimentError("Session configuration differs from its training snapshot; retrain before freezing")
    if expected_config is not None:
        expected = ExperimentConfig.from_dict(expected_config) if isinstance(expected_config, dict) else expected_config
        if _canonical_config(expected)[1] != digest:
            raise ExperimentError("Current configuration differs from this fitted session; retrain before freezing")
    if not session.frozen:
        session.frozen = True
        session.events.append("user_selection_frozen")
    return session


def evaluate_test(
    session: ExperimentSession,
    *,
    on_event: Callable[[str], None] | None = None,
    export_artifacts: bool = True,
) -> ExperimentResult:
    """Evaluate an explicitly frozen session once, returning its cached result thereafter."""
    if not isinstance(session, ExperimentSession) or not session.frozen:
        raise ExperimentError("Freeze the fitted session before evaluating test")
    freeze_experiment(session)  # Validate, without changing a frozen/tested state.
    if session.result is not None:
        return session.result
    if session.finalized:
        raise ExperimentError("Test evaluation previously failed or was interrupted; this session cannot be retested")
    return _evaluate_test_once(session, on_event=on_event, export_artifacts=export_artifacts)


def finalize_experiment(
    session: ExperimentSession,
    *,
    on_event: Callable[[str], None] | None = None,
) -> ExperimentResult:
    """Compatibility API: approve and evaluate a prepared session exactly once."""
    if not isinstance(session, ExperimentSession):
        raise TypeError("session must be an ExperimentSession from prepare_experiment")
    if session.finalized:
        raise ExperimentError("This validation session has already entered final test evaluation")
    freeze_experiment(session)
    return evaluate_test(session, on_event=on_event)


def _evaluate_test_once(
    session: ExperimentSession,
    *,
    on_event: Callable[[str], None] | None = None,
    export_artifacts: bool = True,
) -> ExperimentResult:
    """Evaluate the already-fitted validation session on test exactly once."""
    if not isinstance(session, ExperimentSession):
        raise TypeError("session must be an ExperimentSession from prepare_experiment")
    if session.finalized:
        raise ExperimentError("This validation session has already entered final test evaluation")
    # Mark before touching test rows so a failure cannot silently permit a second test.
    session.finalized = True
    config = session.config
    features = session.features
    target = session.target
    splits = session.splits
    fitted = session.fitted_model
    estimator = fitted.estimator
    preprocessor = fitted.preprocessor
    capabilities = fitted.capabilities
    metrics = session.metrics
    rows = session.rows
    events = session.events
    test_idx = splits["test"]

    def record_event(name: str) -> None:
        events.append(name)
        if on_event is not None:
            on_event(name)

    if fitted.model_id in _EXTENDED_MODEL_IDS:
        from .extended_experiment import ExtendedExperimentSession, evaluate_extended_test

        if not isinstance(session, ExtendedExperimentSession):
            raise ExperimentError("Extended model session does not satisfy its test-computation contract")
        computation = evaluate_extended_test(session)
        metrics["test"] = computation.metrics
        rows.extend(computation.rows)
        session.test_evaluation_count = 1
        record_event("test_metrics_computed_once")
    elif config.task in {"classification", "regression"}:
        x_test = features.iloc[test_idx].copy(deep=True)
        y_test = target.iloc[test_idx].copy(deep=True)
        x_test_ready = preprocessor.transform(x_test)
        test_prediction = estimator.predict(x_test_ready)
        metric_fn = _classification_metrics if config.task == "classification" else _regression_metrics
        metrics["test"] = metric_fn(y_test.to_numpy(), test_prediction)
        rows.extend(_result_rows("test", test_idx, features.index[test_idx], y_test.to_numpy(), test_prediction))
        session.test_evaluation_count = 1
        record_event("test_metrics_computed_once")
    elif config.task == "clustering":
        if capabilities.get("predict"):
            x_test = features.iloc[test_idx].copy(deep=True)
            x_test_ready = preprocessor.transform(x_test)
            test_prediction = estimator.predict(x_test_ready)
            metrics["test"] = _cluster_metrics(x_test_ready, test_prediction)
            rows.extend(_result_rows("test", test_idx, features.index[test_idx], predicted=test_prediction))
            session.test_evaluation_count = 1
            record_event("test_metrics_computed_once")
        else:
            metrics["test"] = {
                "status": "not_supported",
                "reason": "fitted estimator has no predict; test rows were not fitted or scored",
                "metric_family": "unsupervised; accuracy is not applicable",
            }
            record_event("test_not_evaluated_capability_guard")
    elif config.task == "dimensionality reduction":
        if capabilities.get("transform"):
            x_test = features.iloc[test_idx].copy(deep=True)
            x_test_ready = preprocessor.transform(x_test)
            test_embedding = estimator.transform(x_test_ready)
            metrics["test"] = _trustworthiness_metrics(x_test_ready, test_embedding)
            rows.extend(_result_rows("test", test_idx, features.index[test_idx], values=test_embedding))
            session.test_evaluation_count = 1
            record_event("test_metrics_computed_once")
        else:
            metrics["test"] = {
                "status": "not_supported",
                "reason": "fitted estimator has no transform; test rows were not fitted or scored",
                "metric_family": "unsupervised embedding trustworthiness; accuracy is not applicable",
            }
            record_event("test_not_evaluated_capability_guard")
    else:
        if capabilities.get("predict"):
            x_test = features.iloc[test_idx].copy(deep=True)
            x_test_ready = preprocessor.transform(x_test)
            test_prediction = estimator.predict(x_test_ready)
            test_scores = estimator.score_samples(x_test_ready) if capabilities.get("score_samples") else None
            metrics["test"] = _outlier_metrics(test_prediction, test_scores)
            rows.extend(_result_rows("test", test_idx, features.index[test_idx], predicted=test_prediction))
            session.test_evaluation_count = 1
            record_event("test_metrics_computed_once")
        else:
            metrics["test"] = {
                "status": "not_supported",
                "reason": "fitted detector has no predict; test rows were not fitted or scored",
                "metric_family": "unsupervised outlier rate; accuracy is not applicable",
            }
            record_event("test_not_evaluated_capability_guard")

    result_frame = pd.DataFrame(rows)
    audit = {
        "events": events,
        "split_counts": {name: int(len(values)) for name, values in splits.items()},
        "split_positions": {name: [int(item) for item in values] for name, values in splits.items()},
        "preprocessor_fit_positions": [int(item) for item in splits["train"]],
        "preprocessor_fit_policy": "train+validation only" if session.fit_scope == "train_validation" else "training partition only",
        "fit_scope": session.fit_scope,
        "snapshot_manifest": session.snapshot_manifest,
        "test_evaluation_count": session.test_evaluation_count,
        "test_metrics_computed_after_freeze": events.index("selection_frozen") < next(
            (i for i, item in enumerate(events) if item.startswith("test_")), len(events)
        ),
    }
    sequence_plan = getattr(session, "sequence_plan", None)
    if sequence_plan is not None:
        audit["sequence_plan_manifest"] = sequence_plan.to_dict()
    artifact_paths: dict[str, str] = {}
    if export_artifacts and config.output_dir:
        artifact_paths = _write_artifacts(
            config.output_dir,
            session.source_path,
            session.frozen_config,
            session.frozen_config_sha256,
            metrics,
            result_frame,
            fitted,
            training_curves=_session_training_curves(session),
        )
    result = ExperimentResult(
        config=session.frozen_config,
        metrics=metrics,
        audit=audit,
        results=result_frame,
        model=fitted,
        artifact_paths=artifact_paths,
    )
    session.result = result
    return result


def run_experiment(
    config: ExperimentConfig | dict[str, Any],
    *,
    on_event: Callable[[str], None] | None = None,
) -> ExperimentResult:
    """Convenience API: fit, validate, freeze, then evaluate test once."""
    session = prepare_experiment(config, on_event=on_event)
    return finalize_experiment(session, on_event=on_event)


def load_model(path: str | Path) -> FittedModel:
    """Load a self-contained fitted model artifact produced by run_experiment."""
    source = Path(path).expanduser()
    if source.is_dir():
        source = source / "model.joblib"
    if not source.is_file():
        raise ArtifactError(f"Model artifact does not exist: {source}")
    try:
        model = joblib.load(source)
    except Exception as exc:
        raise ArtifactError(f"Could not load model artifact {source}: {exc}") from exc
    if isinstance(model, dict) and model.get("kind") == "pyml_workbench.extended_cpu_model":
        try:
            from .extended_experiment import load_extended_model

            model = load_extended_model(source)
        except Exception as exc:
            raise ArtifactError(f"Could not load extended model artifact {source}: {exc}") from exc
    if not isinstance(model, FittedModel):
        raise ArtifactError(f"File is not a pyml-workbench fitted model artifact: {source}")
    return model


def predict(model: FittedModel | str | Path, values: Any):
    """Predict with a loaded bundle or artifact path, respecting model capabilities."""
    fitted = load_model(model) if isinstance(model, (str, Path)) else model
    return fitted.predict(values)


def transform(model: FittedModel | str | Path, values: Any):
    """Transform rows with a fitted dimensionality-reduction bundle."""
    fitted = load_model(model) if isinstance(model, (str, Path)) else model
    return fitted.transform(values)
