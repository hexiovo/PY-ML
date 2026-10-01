"""Raw-row inference for fitted sequence-model bundles."""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from .extended_experiment import ExtendedExperimentError, ExtendedFittedModel, predict_extended
from .sequence import (
    SequenceConfig,
    SequenceError,
    _group_keys,
    _ordered_groups,
    _time_values,
)


_SEQUENCE_MODEL_IDS = frozenset({"H01", "H02", "H03", "N04", "N06"})
_HMM_MODEL_IDS = frozenset({"H01", "H02", "H03"})


class SequenceInferenceError(ValueError):
    """Raised when raw rows do not satisfy a fitted sequence bundle's protocol."""


def sequence_inference_frame(bundle: ExtendedFittedModel, frame: pd.DataFrame) -> pd.DataFrame:
    """Predict from raw rows and return predictions with source-row mappings.

    ``frame`` is a caller-loaded CSV/DataFrame. The input is never sorted or
    modified in place. Groups and their rows are ordered with the same stable
    rules used to build a training ``SequencePlan``. HMM results have one row
    per observation; N04/N06 results have one row per forecast target that has
    enough preceding context for the bundle's frozen window and horizon.
    """
    sequence_config, feature_columns, observation_columns, target_column = _bundle_protocol(bundle, frame)
    groups = _ordered_input_groups(frame, sequence_config)
    ordered_positions = np.asarray(
        [int(position) for group in groups for position in group["positions"]],
        dtype=np.int64,
    )
    if not len(ordered_positions):
        raise SequenceInferenceError("Sequence inference requires at least one source row")
    source_indexes = frame.index.tolist()
    group_by_position = {
        int(position): group["label"]
        for group in groups
        for position in group["positions"]
    }

    if bundle.model_id in _HMM_MODEL_IDS:
        return _hmm_inference_frame(
            bundle,
            frame,
            groups,
            ordered_positions,
            source_indexes,
            group_by_position,
            observation_columns,
        )
    return _window_inference_frame(
        bundle,
        frame,
        groups,
        source_indexes,
        group_by_position,
        feature_columns,
        sequence_config,
        target_column,
    )


def _bundle_protocol(
    bundle: ExtendedFittedModel,
    frame: pd.DataFrame,
) -> tuple[SequenceConfig, tuple[str, ...], tuple[str, ...], str | None]:
    if not isinstance(bundle, ExtendedFittedModel):
        raise TypeError("bundle must be a loaded or fitted ExtendedFittedModel")
    if bundle.model_id not in _SEQUENCE_MODEL_IDS:
        raise SequenceInferenceError(f"{bundle.model_id!r} is not a supported sequence model")
    if bundle.estimator is None:
        raise SequenceInferenceError("Fitted sequence bundle is missing its estimator")
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("frame must be a pandas DataFrame loaded from raw CSV/DataFrame rows")
    if frame.empty:
        raise SequenceInferenceError("Sequence inference source frame is empty")
    if frame.columns.has_duplicates:
        raise SequenceInferenceError("Sequence inference source has duplicate column names")
    if any(not isinstance(column, str) for column in frame.columns):
        raise SequenceInferenceError("Sequence inference column names must be strings")

    frozen_config = bundle.config
    if not isinstance(frozen_config, Mapping):
        raise SequenceInferenceError("Fitted bundle is missing its frozen experiment configuration")
    if frozen_config.get("model_id") != bundle.model_id or frozen_config.get("task") != bundle.task:
        raise SequenceInferenceError("Fitted bundle identity does not match its frozen configuration")
    dataset = frozen_config.get("dataset")
    if not isinstance(dataset, Mapping):
        raise SequenceInferenceError("Fitted bundle configuration is missing dataset settings")
    target_column = dataset.get("target_column")

    feature_columns = tuple(bundle.feature_columns)
    if not feature_columns or len(set(feature_columns)) != len(feature_columns):
        raise SequenceInferenceError("Fitted bundle has invalid feature columns")
    configured_features = dataset.get("feature_columns")
    if configured_features is not None and tuple(configured_features) != feature_columns:
        raise SequenceInferenceError("Fitted feature columns do not match the frozen configuration")

    if bundle.sequence_config is None:
        raise SequenceInferenceError("Fitted sequence bundle is missing its frozen sequence configuration")
    available_columns = list(frame.columns)
    # The supervised target is part of the frozen training identity, but new
    # prediction rows need not contain labels.
    if target_column is not None and target_column not in available_columns:
        available_columns.append(target_column)
    try:
        sequence_config = SequenceConfig.from_dict(bundle.sequence_config)
        config_sequence = SequenceConfig.from_dict(frozen_config.get("sequence") or {})
        if sequence_config.to_dict() != config_sequence.to_dict():
            raise SequenceInferenceError("Fitted sequence settings do not match the frozen configuration")
        sequence_config.validate(
            task=bundle.task,
            model_id=bundle.model_id,
            feature_columns=feature_columns,
            target_column=target_column,
            available_columns=available_columns,
        )
    except SequenceError as exc:
        raise SequenceInferenceError(str(exc)) from exc

    observation_columns = sequence_config.observation_columns or feature_columns
    if bundle.model_id == "H03" and len(observation_columns) != 1:
        raise SequenceInferenceError("H03 requires exactly one frozen observation column")
    return sequence_config, feature_columns, tuple(observation_columns), target_column


def _ordered_input_groups(frame: pd.DataFrame, config: SequenceConfig) -> list[dict[str, Any]]:
    try:
        group_keys, group_values = _group_keys(frame, config.group_column)
        time_values = _time_values(frame, config, group_keys)
        groups = _ordered_groups(group_keys, group_values, time_values)
    except SequenceError as exc:
        raise SequenceInferenceError(str(exc)) from exc
    if not groups:
        raise SequenceInferenceError("Sequence inference source frame has no groups")
    return groups


def _hmm_inference_frame(
    bundle: ExtendedFittedModel,
    frame: pd.DataFrame,
    groups: list[dict[str, Any]],
    ordered_positions: np.ndarray,
    source_indexes: list[Any],
    group_by_position: dict[int, Any],
    observation_columns: tuple[str, ...],
) -> pd.DataFrame:
    observations = frame.loc[:, list(observation_columns)].to_numpy(dtype=object, copy=True)[ordered_positions]
    lengths = [len(group["positions"]) for group in groups]
    try:
        states = np.asarray(
            predict_extended(
                bundle,
                {"observations": observations, "lengths": lengths},
                operation="predict",
            )
        )
        posterior = np.asarray(
            predict_extended(
                bundle,
                {"observations": observations, "lengths": lengths},
                operation="posterior",
            ),
            dtype=np.float64,
        )
    except ExtendedExperimentError:
        # In particular, preserve the fitted H03 alphabet's unknown-category
        # error so callers can report the exact mapping failure.
        raise
    if states.ndim != 1 or len(states) != len(ordered_positions):
        raise SequenceInferenceError("Fitted HMM returned an invalid hidden-state vector")
    if (
        posterior.ndim != 2
        or posterior.shape[0] != len(ordered_positions)
        or posterior.shape[1] < 1
        or not np.isfinite(posterior).all()
    ):
        raise SequenceInferenceError("Fitted HMM returned an invalid posterior matrix")

    records: list[dict[str, Any]] = []
    for output_position, source_position in enumerate(ordered_positions):
        position = int(source_position)
        record = {
            "source_row_position": position,
            "source_index": source_indexes[position],
            "group_id": group_by_position[position],
            "hidden_state": states[output_position].item()
            if isinstance(states[output_position], np.generic)
            else states[output_position],
        }
        record.update(
            {f"posterior_{state}": float(value) for state, value in enumerate(posterior[output_position])}
        )
        records.append(record)
    return pd.DataFrame.from_records(records)


def _window_inference_frame(
    bundle: ExtendedFittedModel,
    frame: pd.DataFrame,
    groups: list[dict[str, Any]],
    source_indexes: list[Any],
    group_by_position: dict[int, Any],
    feature_columns: tuple[str, ...],
    config: SequenceConfig,
    target_column: str | None,
) -> pd.DataFrame:
    if target_column is None:
        raise SequenceInferenceError("N04/N06 fitted configuration is missing its target column")
    features = frame.loc[:, list(feature_columns)].to_numpy(dtype=object, copy=True)
    source_maps: list[np.ndarray] = []
    target_positions: list[int] = []
    needed = config.window + config.horizon
    for group in groups:
        group_positions = np.asarray(group["positions"], dtype=np.int64)
        if len(group_positions) < needed:
            raise SequenceInferenceError(
                f"N04/N06 group {group['label']!r} has {len(group_positions)} rows; "
                f"window={config.window}, horizon={config.horizon} requires at least {needed}"
            )
        window_count = len(group_positions) - needed + 1
        for start in range(window_count):
            source_maps.append(group_positions[start : start + config.window])
            target_positions.append(int(group_positions[start + needed - 1]))
    if not source_maps:
        raise SequenceInferenceError("N04/N06 inference source has no valid forecast windows")

    source_map_array = np.stack(source_maps).astype(np.int64, copy=False)
    windows = features[source_map_array]
    predictions = _predict_fitted_windows(bundle, windows, feature_columns)
    if len(predictions) != len(target_positions):
        raise SequenceInferenceError("Fitted sequence regressor returned the wrong number of predictions")
    if not np.isfinite(predictions).all():
        raise SequenceInferenceError("Fitted sequence regressor returned non-finite predictions")

    records = []
    for window_number, target_position in enumerate(target_positions):
        position = int(target_position)
        source_positions = [int(item) for item in source_map_array[window_number]]
        records.append(
            {
                "source_row_position": position,
                "source_index": source_indexes[position],
                "group_id": group_by_position[position],
                "prediction": float(predictions[window_number]),
                "window_source_row_positions": source_positions,
                "window_source_indexes": [source_indexes[item] for item in source_positions],
            }
        )
    return pd.DataFrame.from_records(records)


def _predict_fitted_windows(
    bundle: ExtendedFittedModel,
    windows: np.ndarray,
    feature_columns: tuple[str, ...],
) -> np.ndarray:
    if bundle.preprocessor is None or not callable(getattr(bundle.preprocessor, "transform", None)):
        raise SequenceInferenceError("Fitted N04/N06 bundle is missing its fitted preprocessor")
    count, window, feature_count = windows.shape
    if feature_count != len(feature_columns):
        raise SequenceInferenceError("N04/N06 window feature width differs from the frozen bundle")
    raw_frame = pd.DataFrame(windows.reshape(-1, feature_count), columns=feature_columns)
    try:
        transformed = bundle.preprocessor.transform(raw_frame)
        if callable(getattr(transformed, "toarray", None)):
            transformed = transformed.toarray()
        transformed = np.asarray(transformed, dtype=np.float32)
    except (TypeError, ValueError, OverflowError) as exc:
        raise SequenceInferenceError(f"Fitted feature preprocessing failed for inference windows: {exc}") from exc
    if (
        transformed.ndim != 2
        or transformed.shape[0] != count * window
        or transformed.shape[1] < 1
        or not np.isfinite(transformed).all()
    ):
        raise SequenceInferenceError("Transformed inference windows must contain finite fitted features")
    transformed_windows = np.ascontiguousarray(
        transformed.reshape(count, window, transformed.shape[1]), dtype=np.float32
    )

    # Use the shared CPU inference helper on the already-fitted estimator. The
    # raw-row adapter transforms once here so fitted categorical preprocessors
    # work too; predict_extended's tensor-window input contract is float-only.
    from .sequence_models import predict_deep

    try:
        predictions = np.asarray(predict_deep(bundle.estimator, bundle.model_id, transformed_windows), dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise SequenceInferenceError(f"Fitted {bundle.model_id} inference failed: {exc}") from exc
    if predictions.ndim == 2 and predictions.shape[1] == 1:
        return predictions[:, 0]
    if predictions.ndim != 1:
        raise SequenceInferenceError("Fitted sequence regressor must return one value per forecast window")
    return predictions
