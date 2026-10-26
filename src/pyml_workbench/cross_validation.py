"""Leakage-safe outer split and reusable K-fold evaluation primitives.

This module works with stable row positions rather than owning feature or target
containers. Each candidate callback receives only one fold's global training
and validation positions; it must fit every learned transform on training rows
and use validation rows only for transform, predict, and score.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
import hashlib
import json
import math
from numbers import Integral, Real
import time
from typing import Any, Literal

import numpy as np
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split


Task = Literal["classification", "regression"]
InterruptReason = Literal["cancelled", "timeout"]
FoldCallback = Callable[[np.ndarray, np.ndarray, int], "FoldEvaluation"]


class CrossValidationError(ValueError):
    """Raised when a leakage-safe development/test CV plan cannot be formed."""


class CrossValidationInterrupted(RuntimeError):
    """A candidate did not finish every fold before cancellation or its deadline."""

    def __init__(self, reason: InterruptReason, completed_folds: int):
        self.reason = reason
        self.completed_folds = completed_folds
        super().__init__(
            f"Cross-validation candidate interrupted ({reason}) after "
            f"{completed_folds} of its folds; partial candidates are not complete."
        )


@dataclass(frozen=True)
class FoldEvaluation:
    """Outputs from one fold callback.

    For classification, score_columns identifies the class represented by each
    score-matrix column. The evaluator aligns columns to CVPlan.class_labels.
    """

    primary_score: float
    predictions: Any
    scores: Any | None = None
    score_columns: tuple[Any, ...] | None = None
    scores_unavailable_reason: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class CVPlan:
    """Immutable global row positions shared by every candidate in one search."""

    task: Task
    row_count: int
    seed: int
    n_splits: int
    test_fraction: float
    development_indices: np.ndarray
    test_indices: np.ndarray
    folds: tuple[tuple[np.ndarray, np.ndarray], ...]
    fold_id_by_row: tuple[int, ...]
    split_sha256: str
    class_labels: tuple[Any, ...] | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return the split metadata without feature or target data."""
        return {
            "evaluation_method": "kfold",
            "task": self.task,
            "row_count": self.row_count,
            "seed": self.seed,
            "n_splits": self.n_splits,
            "test_fraction": self.test_fraction,
            "development_indices": self.development_indices.tolist(),
            "test_indices": self.test_indices.tolist(),
            "folds": [
                {"train": train.tolist(), "validation": valid.tolist()}
                for train, valid in self.folds
            ],
            "fold_id_by_row": list(self.fold_id_by_row),
            "class_order": (
                [_json_safe_label(label) for label in self.class_labels]
                if self.class_labels is not None
                else None
            ),
            "oof_semantics": (
                "Each development row is validated exactly once; metrics and "
                "curves built from these predictions are cross-validation OOF."
            ),
            "split_sha256": self.split_sha256,
        }


@dataclass(frozen=True)
class CVResult:
    """Complete fold scores and OOF outputs aligned to development row positions."""

    fold_scores: tuple[float, ...]
    mean_score: float
    fold_std_ddof1: float
    oof_row_positions: np.ndarray
    oof_predictions: np.ndarray
    oof_scores: np.ndarray | None
    score_columns: tuple[Any, ...] | None
    scores_unavailable_reason: str | None
    fold_metadata: tuple[Mapping[str, Any], ...]
    primary_metric_name: str | None = None


def _readonly_int_array(values: Any) -> np.ndarray:
    result = np.asarray(values, dtype=np.int64).copy()
    if result.ndim != 1:
        raise CrossValidationError("Row positions must be one-dimensional.")
    result.setflags(write=False)
    return result


def _readonly_array(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values).copy()
    result.setflags(write=False)
    return result


def _json_safe_label(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return {"type": f"{type(value).__module__}.{type(value).__qualname__}", "repr": repr(value)}


def _split_digest(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _is_missing_scalar(value: Any) -> bool:
    if value is None or value is np.ma.masked:
        return True
    if isinstance(value, (float, np.floating)):
        return bool(np.isnan(value))
    if isinstance(value, (np.datetime64, np.timedelta64)):
        return bool(np.isnat(value))
    if type(value).__name__ in {"NAType", "NaTType"}:
        return True
    try:
        comparison = value != value
        if isinstance(comparison, (bool, np.bool_)):
            return bool(comparison)
    except (TypeError, ValueError):
        pass
    return False


def _missing_mask(values: np.ndarray) -> np.ndarray:
    if values.dtype.kind in {"f", "c"}:
        return np.isnan(values)
    if values.dtype.kind in {"m", "M"}:
        return np.isnat(values)
    if values.dtype.kind == "O":
        return np.fromiter(
            (_is_missing_scalar(value) for value in values),
            dtype=np.bool_,
            count=len(values),
        )
    return np.zeros(len(values), dtype=np.bool_)


def _python_scalar(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


def _labels_equal(left: Any, right: Any) -> bool:
    try:
        result = left == right
        return bool(result)
    except (TypeError, ValueError):
        return False


def _validated_class_order(
    target: np.ndarray,
    class_order: Sequence[Any] | None,
) -> tuple[Any, ...]:
    try:
        observed = tuple(_python_scalar(value) for value in np.unique(target).tolist())
    except (TypeError, ValueError) as exc:
        raise CrossValidationError(
            "Classification labels must have one comparable scalar type."
        ) from exc

    try:
        observed_set = set(observed)
    except TypeError as exc:
        raise CrossValidationError("Classification labels must be hashable scalar values.") from exc

    if class_order is None:
        return observed
    if isinstance(class_order, (str, bytes)):
        raise CrossValidationError("class_order must be a sequence of labels, not a string.")
    requested = tuple(_python_scalar(value) for value in class_order)
    try:
        requested_set = set(requested)
    except TypeError as exc:
        raise CrossValidationError("class_order labels must be hashable scalar values.") from exc
    if len(requested_set) != len(requested) or requested_set != observed_set:
        raise CrossValidationError(
            "class_order must contain every observed target class exactly once."
        )
    return requested


def make_cv_plan(
    row_count: int,
    target: Any,
    *,
    task: Task,
    n_splits: int = 5,
    seed: int = 42,
    test_fraction: float = 0.2,
    class_order: Sequence[Any] | None = None,
) -> CVPlan:
    """Create a shared 80/20 outer split and development-only shuffled folds.

    Classification uses a stratified final test split and StratifiedKFold.
    Regression uses a shuffled final test split and KFold. Returned row
    positions are sorted, stable positions into the caller's original data.
    """
    if isinstance(row_count, bool) or not isinstance(row_count, Integral) or row_count < 3:
        raise CrossValidationError("At least 3 rows are required for a development/test split.")
    row_count = int(row_count)
    if task not in {"classification", "regression"}:
        raise CrossValidationError("K-fold is supported only for tabular classification or regression.")
    if isinstance(seed, bool) or not isinstance(seed, Integral) or seed < 0:
        raise CrossValidationError("The split seed must be a non-negative integer.")
    seed = int(seed)
    if isinstance(n_splits, bool) or not isinstance(n_splits, Integral) or not 2 <= n_splits <= 10:
        raise CrossValidationError("K-fold count must be an integer from 2 through 10.")
    n_splits = int(n_splits)
    if (
        isinstance(test_fraction, bool)
        or not isinstance(test_fraction, Real)
        or not math.isfinite(float(test_fraction))
        or not 0 < float(test_fraction) < 1
    ):
        raise CrossValidationError("The final test fraction must be between 0 and 1.")
    test_fraction = float(test_fraction)

    y = np.asarray(target)
    if y.ndim != 1 or len(y) != row_count:
        raise CrossValidationError("The target must be one-dimensional and match the feature row count.")
    if _missing_mask(y).any():
        raise CrossValidationError("The target contains missing labels; resolve them before K-fold evaluation.")
    if task == "regression" and class_order is not None:
        raise CrossValidationError("class_order is only valid for classification.")

    n_test = int(math.ceil(row_count * test_fraction))
    n_development = row_count - n_test
    if n_development < n_splits:
        raise CrossValidationError(
            f"K={n_splits} needs at least {n_splits} development rows after reserving the final test set; "
            f"only {n_development} remain. Reduce K or add rows."
        )

    labels: tuple[Any, ...] | None = None
    stratify: np.ndarray | None = None
    if task == "classification":
        labels = _validated_class_order(y, class_order)
        try:
            observed_classes, counts = np.unique(y, return_counts=True)
        except (TypeError, ValueError) as exc:
            raise CrossValidationError(
                "Classification labels must have one comparable scalar type."
            ) from exc
        n_classes = len(labels)
        if len(counts) != n_classes:
            raise CrossValidationError("Could not count the target classes consistently.")
        counts_by_label = {
            _python_scalar(label): int(count)
            for label, count in zip(observed_classes.tolist(), counts, strict=True)
        }
        too_small = {
            repr(label): counts_by_label[label]
            for label in labels
            if counts_by_label[label] < 2
        }
        if too_small:
            raise CrossValidationError(
                "The final test split must be able to retain every class; "
                f"classes with fewer than 2 rows: {too_small}. Add data or simplify labels."
            )
        if n_test < n_classes or n_development < n_classes:
            raise CrossValidationError(
                "The dataset is too small to place every class in both development and final test "
                f"partitions ({n_classes} classes, {n_development} development rows, {n_test} test rows)."
            )
        stratify = y

    positions = np.arange(row_count, dtype=np.int64)
    try:
        development, test = train_test_split(
            positions,
            test_size=n_test,
            random_state=seed,
            shuffle=True,
            stratify=stratify,
        )
    except ValueError as exc:
        if task == "classification":
            raise CrossValidationError(
                "Could not stratify the independent final test split. Add examples per class "
                f"or simplify labels. scikit-learn detail: {exc}"
            ) from exc
        raise CrossValidationError(f"Could not form an independent final test split: {exc}") from exc

    development = np.sort(np.asarray(development, dtype=np.int64))
    test = np.sort(np.asarray(test, dtype=np.int64))
    y_development = y[development]
    if task == "classification":
        try:
            dev_classes, dev_counts = np.unique(y_development, return_counts=True)
            _, test_counts = np.unique(y[test], return_counts=True)
        except (TypeError, ValueError) as exc:
            raise CrossValidationError(
                "Classification labels must have one comparable scalar type."
            ) from exc
        dev_counts_by_label = {
            _python_scalar(label): int(count)
            for label, count in zip(dev_classes.tolist(), dev_counts, strict=True)
        }
        small_dev = {
            repr(label): dev_counts_by_label[label]
            for label in labels or ()
            if dev_counts_by_label[label] < n_splits
        }
        if small_dev:
            raise CrossValidationError(
                f"Each class needs at least K={n_splits} development rows for stratified folds; "
                f"classes below K: {small_dev}. Reduce K or add examples."
            )
        if len(test_counts) != len(labels or ()):
            raise CrossValidationError(
                "The final test split is missing one or more target classes. Add examples per class "
                "or simplify labels."
            )

    splitter = (
        StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        if task == "classification"
        else KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    )
    fold_pairs: list[tuple[np.ndarray, np.ndarray]] = []
    fold_id = np.full(row_count, -2, dtype=np.int64)
    fold_id[development] = -1
    try:
        split_iterator = splitter.split(
            np.zeros(len(development), dtype=np.uint8),
            y_development if task == "classification" else None,
        )
        for fold_number, (train_local, valid_local) in enumerate(split_iterator):
            train_global = development[np.asarray(train_local, dtype=np.int64)]
            valid_global = development[np.asarray(valid_local, dtype=np.int64)]
            if np.intersect1d(train_global, valid_global).size:
                raise CrossValidationError("A development row appears in both fold training and validation.")
            if np.isin(train_global, test).any() or np.isin(valid_global, test).any():
                raise CrossValidationError("Final test rows cannot appear in a cross-validation fold.")
            if np.any(fold_id[valid_global] != -1):
                raise CrossValidationError("A development row appears in more than one OOF validation fold.")
            fold_id[valid_global] = fold_number
            fold_pairs.append((_readonly_int_array(train_global), _readonly_int_array(valid_global)))
    except ValueError as exc:
        raise CrossValidationError(f"Could not form K={n_splits} folds: {exc}") from exc

    if len(fold_pairs) != n_splits or np.any(fold_id[development] < 0):
        raise CrossValidationError(
            "The folds do not provide exactly one OOF validation assignment per development row."
        )

    split_payload = {
        "task": task,
        "row_count": row_count,
        "seed": seed,
        "n_splits": n_splits,
        "test_fraction": test_fraction,
        "development_indices": [int(value) for value in development],
        "test_indices": [int(value) for value in test],
        "folds": [
            {
                "train": [int(value) for value in train],
                "validation": [int(value) for value in valid],
            }
            for train, valid in fold_pairs
        ],
    }
    return CVPlan(
        task=task,
        row_count=row_count,
        seed=seed,
        n_splits=n_splits,
        test_fraction=test_fraction,
        development_indices=_readonly_int_array(development),
        test_indices=_readonly_int_array(test),
        folds=tuple(fold_pairs),
        fold_id_by_row=tuple(int(value) for value in fold_id),
        split_sha256=_split_digest(split_payload),
        class_labels=labels,
    )


def _validated_score_columns(
    columns: Sequence[Any] | None,
    width: int,
) -> tuple[Any, ...] | None:
    if columns is None:
        return None
    if isinstance(columns, (str, bytes)):
        return None
    try:
        normalized = tuple(_python_scalar(value) for value in columns)
    except TypeError:
        return None
    if len(normalized) != width:
        return None
    try:
        if len(set(normalized)) != len(normalized):
            return None
    except TypeError:
        return None
    return normalized


def _align_class_scores(
    scores: np.ndarray,
    columns: tuple[Any, ...],
    class_labels: tuple[Any, ...],
) -> tuple[np.ndarray | None, str | None]:
    if len(columns) != len(class_labels):
        return None, "Score columns do not include every target class."
    order: list[int] = []
    for label in class_labels:
        matches = [
            index for index, column_label in enumerate(columns)
            if _labels_equal(column_label, label)
        ]
        if len(matches) != 1:
            return None, f"Scores cannot be aligned to target class {label!r}."
        order.append(matches[0])
    if any(
        not any(_labels_equal(column_label, label) for label in class_labels)
        for column_label in columns
    ):
        return None, "Score columns contain labels outside the target class order."
    return scores[:, order], None


def _fold_score_matrix(
    evaluation: FoldEvaluation,
    valid_count: int,
    plan: CVPlan,
) -> tuple[np.ndarray | None, tuple[Any, ...] | None, str | None]:
    if evaluation.scores is None:
        reason = evaluation.scores_unavailable_reason or (
            "The fold callback did not return probability or decision scores."
        )
        return None, None, reason
    if evaluation.scores_unavailable_reason:
        return None, None, evaluation.scores_unavailable_reason
    try:
        scores = np.asarray(evaluation.scores, dtype=np.float64)
    except (TypeError, ValueError):
        return None, None, "Scores are not a numeric matrix."
    if scores.ndim != 2:
        return None, None, "Scores must be a two-dimensional matrix."
    if scores.shape[0] != valid_count:
        return None, None, "Score rows do not match the fold validation positions."
    if not np.isfinite(scores).all():
        return None, None, "Scores contain NaN or infinite values."

    columns = _validated_score_columns(evaluation.score_columns, scores.shape[1])
    if columns is None:
        return None, None, "Score columns are missing, duplicated, or do not match the score matrix."
    if plan.task == "classification":
        if plan.class_labels is None:
            return None, None, "The CV plan has no target class order for score alignment."
        if (len(plan.class_labels) == 2 and scores.shape[1] == 1
                and columns == (plan.class_labels[1],)
                and evaluation.metadata.get("score_semantics") == "binary_positive_class_margin"):
            # Retain the estimator's genuine positive-class margin. It is not
            # a probability and does not imply a fabricated negative column.
            return scores, columns, None
        scores, reason = _align_class_scores(scores, columns, plan.class_labels)
        if reason is not None:
            return None, None, reason
        return scores, plan.class_labels, None
    return scores, columns, None


def _prediction_dtype(predictions: Sequence[np.ndarray]) -> np.dtype[Any]:
    dtypes = [array.dtype for array in predictions]
    kinds = {dtype.kind for dtype in dtypes}
    if kinds <= set("biufc"):
        return np.result_type(*dtypes)
    if len(kinds) == 1:
        return np.result_type(*dtypes)
    return np.dtype(object)


def evaluate_cv_candidate(
    plan: CVPlan,
    fit_and_score: FoldCallback,
    *,
    should_cancel: Callable[[], bool] | None = None,
    deadline: float | None = None,
    clock: Callable[[], float] = time.monotonic,
    primary_metric_name: str | None = None,
) -> CVResult:
    """Run all shared folds and assemble exactly-once OOF outputs.

    The callback signature is train_positions, valid_positions, fold_number.
    Positions are global indices into the caller's original rows, and never
    include final-test rows. The callback owns feature/target slicing and must
    fit every learned preprocessing step only on train_positions.
    """
    if len(plan.folds) != plan.n_splits:
        raise CrossValidationError("The CV plan does not contain the requested number of folds.")
    if len(plan.fold_id_by_row) != plan.row_count:
        raise CrossValidationError("The CV plan has an invalid fold-id map.")
    if plan.development_indices.ndim != 1 or plan.test_indices.ndim != 1:
        raise CrossValidationError("The CV plan row positions must be one-dimensional.")
    if len(np.intersect1d(plan.development_indices, plan.test_indices)):
        raise CrossValidationError("Development and final-test rows overlap.")
    if isinstance(deadline, bool) or (
        deadline is not None and (not isinstance(deadline, Real) or not math.isfinite(float(deadline)))
    ):
        raise CrossValidationError("deadline must be a finite monotonic-clock value.")
    if not callable(fit_and_score):
        raise CrossValidationError("fit_and_score must be a fold callback.")

    local_index = np.full(plan.row_count, -1, dtype=np.int64)
    local_index[plan.development_indices] = np.arange(
        len(plan.development_indices), dtype=np.int64
    )
    assigned = np.zeros(len(plan.development_indices), dtype=np.bool_)
    prediction_chunks: list[tuple[np.ndarray, np.ndarray]] = []
    score_chunks: list[tuple[np.ndarray, np.ndarray]] = []
    score_reasons: list[str] = []
    fold_scores: list[float] = []
    metadata: list[Mapping[str, Any]] = []
    output_score_columns: tuple[Any, ...] | None = None

    def interrupted_reason() -> InterruptReason | None:
        if should_cancel is not None and should_cancel():
            return "cancelled"
        if deadline is not None and clock() >= float(deadline):
            return "timeout"
        return None

    for fold_number, (train_positions, valid_positions) in enumerate(plan.folds):
        reason = interrupted_reason()
        if reason is not None:
            raise CrossValidationInterrupted(reason, len(fold_scores))

        train_copy = _readonly_int_array(train_positions)
        valid_copy = _readonly_int_array(valid_positions)
        try:
            evaluation = fit_and_score(train_copy, valid_copy, fold_number)
        except Exception as exc:
            reason = interrupted_reason()
            if reason is not None:
                raise CrossValidationInterrupted(reason, len(fold_scores)) from exc
            raise

        reason = interrupted_reason()
        if reason is not None:
            raise CrossValidationInterrupted(reason, fold_number + 1)
        if not isinstance(evaluation, FoldEvaluation):
            raise CrossValidationError("Fold callback must return a FoldEvaluation.")

        try:
            score = float(evaluation.primary_score)
        except (TypeError, ValueError, OverflowError) as exc:
            raise CrossValidationError(
                f"Fold {fold_number + 1} returned an invalid primary score."
            ) from exc
        if not math.isfinite(score):
            raise CrossValidationError(f"Fold {fold_number + 1} returned a non-finite primary score.")

        predictions = np.asarray(evaluation.predictions)
        if predictions.ndim == 0 or predictions.shape[0] != len(valid_positions):
            raise CrossValidationError(
                f"Fold {fold_number + 1} prediction rows do not match its validation positions."
            )
        if predictions.dtype.kind in "biufc" and not np.isfinite(predictions).all():
            raise CrossValidationError(
                f"Fold {fold_number + 1} predictions contain NaN or infinite values."
            )

        local_positions = local_index[valid_positions]
        if np.any(local_positions < 0):
            raise CrossValidationError("A validation position is outside the development partition.")
        if assigned[local_positions].any():
            raise CrossValidationError("OOF predictions would overwrite an already-covered row.")
        assigned[local_positions] = True
        prediction_chunks.append((local_positions.copy(), predictions.copy()))

        fold_scores.append(score)
        if not isinstance(evaluation.metadata, Mapping):
            raise CrossValidationError("Fold metadata must be a mapping.")
        metadata.append(dict(evaluation.metadata))

        aligned_scores, columns, score_reason = _fold_score_matrix(
            evaluation,
            len(valid_positions),
            plan,
        )
        if score_reason is not None:
            score_reasons.append(f"Fold {fold_number + 1}: {score_reason}")
        elif aligned_scores is not None:
            if output_score_columns is None:
                output_score_columns = columns
            elif columns != output_score_columns:
                score_reasons.append(
                    f"Fold {fold_number + 1}: aligned score columns changed between folds."
                )
            score_chunks.append((local_positions.copy(), aligned_scores.copy()))

        reason = interrupted_reason()
        if reason is not None:
            raise CrossValidationInterrupted(reason, len(fold_scores))

    if len(fold_scores) != plan.n_splits or not assigned.all():
        raise CrossValidationError(
            "The candidate did not produce one complete OOF prediction per development row."
        )

    prediction_shape = prediction_chunks[0][1].shape[1:]
    if any(chunk.shape[1:] != prediction_shape for _, chunk in prediction_chunks):
        raise CrossValidationError("Prediction shape changed between folds.")
    prediction_result = np.empty(
        (len(plan.development_indices),) + prediction_shape,
        dtype=_prediction_dtype([chunk for _, chunk in prediction_chunks]),
    )
    for local_positions, chunk in prediction_chunks:
        prediction_result[local_positions] = chunk

    if len(score_chunks) == plan.n_splits and not score_reasons:
        score_shape = score_chunks[0][1].shape[1:]
        if any(chunk.shape[1:] != score_shape for _, chunk in score_chunks):
            score_reasons.append("Score shape changed between folds.")
            oof_scores = None
            output_score_columns = None
        else:
            oof_scores = np.empty(
                (len(plan.development_indices),) + score_shape,
                dtype=np.float64,
            )
            for local_positions, chunk in score_chunks:
                oof_scores[local_positions] = chunk
    else:
        oof_scores = None
        if not score_reasons:
            score_reasons.append("Scores were not available for every fold.")
        output_score_columns = None

    if should_cancel is not None and should_cancel():
        raise CrossValidationInterrupted("cancelled", len(fold_scores))
    if deadline is not None and clock() >= float(deadline):
        raise CrossValidationInterrupted("timeout", len(fold_scores))

    return CVResult(
        fold_scores=tuple(fold_scores),
        mean_score=float(np.mean(fold_scores)),
        fold_std_ddof1=float(np.std(fold_scores, ddof=1)),
        oof_row_positions=_readonly_int_array(plan.development_indices),
        oof_predictions=_readonly_array(prediction_result),
        oof_scores=_readonly_array(oof_scores) if oof_scores is not None else None,
        score_columns=output_score_columns,
        scores_unavailable_reason="; ".join(score_reasons) if score_reasons else None,
        fold_metadata=tuple(metadata),
        primary_metric_name=primary_metric_name,
    )
