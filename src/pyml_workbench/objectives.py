"""Task objectives consume validation outputs, never holdout test data."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np
from sklearn.metrics import balanced_accuracy_score, silhouette_score

from .experiment import _json_digest
from .metrics import METRICS, metric_direction, anomaly_metrics


@dataclass
class ObjectiveSpec:
    metric: str | None = None
    direction: str | None = None
    split: str = "validation"
    objective_labels_column: str | None = None
    label_mapping: dict | None = None
    coverage_min: float = .8

    @classmethod
    def from_dict(cls, payload=None):
        return cls(**(payload.to_dict() if isinstance(payload, cls) else payload or {}))

    def to_dict(self):
        return asdict(self)

    def resolve(self, task):
        if task not in METRICS:
            raise ValueError(f"Unknown objective task {task!r}")
        metric = self.metric or next(iter(METRICS[task]))
        direction = metric_direction(task, metric)
        resolved = ObjectiveSpec(**dict(self.to_dict(), metric=metric, direction=self.direction or direction))
        if resolved.direction != direction or resolved.split not in {"validation", "train_exploratory"}:
            raise ValueError("Objective metric/direction/split is incompatible with this task")
        if task == "clustering" and metric in {"ari", "nmi", "ami"} and not resolved.objective_labels_column:
            raise ValueError("External clustering metrics require independent objective_labels_column")
        if not .8 <= resolved.coverage_min <= 1:
            raise ValueError("Cluster coverage threshold must be between .8 and 1")
        if task == "anomaly detection" and not resolved.objective_labels_column:
            raise ValueError("Automatic anomaly search requires independent objective_labels_column")
        return resolved

    def fingerprint(self, snapshot):
        return _json_digest({"objective": self.to_dict(), "snapshot": snapshot.to_dict()})


def cluster_objective(values, labels, coverage_min=.8):
    labels = np.asarray(labels)
    valid = labels != -1
    coverage = float(valid.mean()) if len(labels) else 0.
    groups = len(np.unique(labels[valid]))
    result = {"coverage": coverage, "non_noise_cluster_count": groups, "silhouette": None}
    if coverage < coverage_min or groups < 2 or groups >= int(valid.sum()):
        return None, result
    # scipy sparse matrices support the same boolean row selection.
    result["silhouette"] = float(silhouette_score(values[valid], labels[valid]))
    return result["silhouette"], result


def score_session(session, objective, snapshot):
    if session.config.model_id in {"H01", "H02", "H03", "N01", "N02", "N04", "N06", "N07", "N08", "N09"}:
        from .extended_experiment import score_extended_session

        return score_extended_session(session, objective)
    split = "train" if objective.split == "train_exploratory" else "validation"
    metrics = dict(session.metrics.get(split, {}))
    if metrics.get("status") in {"not_supported", "unsupported"}:
        return None, metrics
    positions = session.splits[split]
    task = session.config.task
    if task == "clustering" and objective.metric == "silhouette":
        rows = [row for row in session.rows if row["split"] == split]
        labels = np.asarray([row["y_pred"] for row in rows])
        if len(labels) != len(positions):
            return None, {"reason": "No labels for the selected score partition"}
        transformed = session.fitted_model.preprocessor.transform(session.features.iloc[positions])
        value, details = cluster_objective(transformed, labels, objective.coverage_min)
        return value, dict(metrics, **details)
    if task == "anomaly detection":
        if snapshot.objective_labels is None:
            raise ValueError("Objective labels were not loaded independently")
        expected = snapshot.objective_labels.iloc[positions]
        if objective.label_mapping:
            expected = expected.map(lambda value: objective.label_mapping.get(str(value)))
        if expected.isna().any() or set(expected.unique()) != {-1, 1}:
            raise ValueError("Each scored anomaly partition needs both -1 anomaly and +1 normal labels; provide label_mapping")
        predicted = [row["y_pred"] for row in session.rows if row["split"] == split]
        estimator = session.fitted_model.estimator
        scores = None
        if callable(getattr(estimator, "score_samples", None)):
            values = session.fitted_model.preprocessor.transform(session.features.iloc[positions])
            scores = estimator.score_samples(values)
        details = dict(metrics, **anomaly_metrics(expected.to_numpy(), predicted, scores))
        return details.get(objective.metric), details
    if session.config.evaluation is not None and objective.split == "validation":
        cv = session.metrics.get("cross_validation", {})
        metrics = dict(cv.get("mean", {}), cross_validation=cv)
    value = metrics.get(objective.metric)
    if not isinstance(value, (float, int)) or not math.isfinite(value):
        return None, metrics
    return float(value), metrics
