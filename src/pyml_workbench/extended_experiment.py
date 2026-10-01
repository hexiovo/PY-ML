"""Validation, refit, test, and CPU bundle helpers for S03 models.

Core code remains the owner of session lifecycle and persistence. This module
fits extended estimators and returns pure metrics/rows for the final test call.
Optional numerical packages are reached only inside the requested model path.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Callable, Mapping

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import ExperimentConfig
from .data import DatasetError
from .experiment import (
    DatasetSnapshot,
    ExperimentError,
    ExperimentSession,
    FittedModel,
    _canonical_config,
    _classification_metrics,
    _json_digest,
    _regression_metrics,
    _result_rows,
)
from .preprocessing import CategoricalStringifier
from .sequence import SequenceError, SequencePlan, _json_text, _normal_scalar
from .sequence_models import (
    DEEP_MODEL_IDS,
    HMM_MODEL_IDS,
    WINDOW_DEEP_MODEL_IDS,
    ExtendedModelError,
    build_deep_from_state,
    build_hmm_estimator,
    deep_estimator_state,
    fit_deep_validation,
    fit_hmm,
    hmm_lengths,
    hmm_test_outputs,
    predict_deep,
    predict_proba_deep,
    refit_deep_fixed_epochs,
    resolve_hmm_seed,
    resolve_deep_seed,
    score_hmm,
    validate_deep_inputs,
    validate_deep_targets,
    validate_extended_parameters,
)


_EXTENDED_MODEL_IDS = HMM_MODEL_IDS | DEEP_MODEL_IDS
_BUNDLE_KIND = "pyml_workbench.extended_cpu_model"
_PROVENANCE_SCHEMA = "pyml_workbench.refit_provenance/1"


class ExtendedExperimentError(ExperimentError):
    """Raised when an extended fit, score, refit, or inference contract fails."""


@dataclass(frozen=True)
class ExtendedTestComputation:
    """Pure final-test metrics and result rows; contains no lifecycle state."""

    metrics: dict[str, Any]
    rows: list[dict[str, Any]]


@dataclass
class ExtendedExperimentSession(ExperimentSession):
    """Existing session fields plus the plan and train-only mapping provenance."""

    sequence_plan: SequencePlan | None = None
    training_metadata: dict[str, Any] = field(default_factory=dict, repr=False)

    @property
    def extended_training_metadata(self) -> dict[str, Any]:
        return self.training_metadata

    @property
    def refit_provenance(self) -> dict[str, Any] | None:
        value = self.training_metadata.get("refit_provenance")
        return None if value is None else _json_copy(value)


class ExtendedFittedModel(FittedModel):
    """FittedModel-compatible model with strict sequence and class-map routing."""

    def __init__(
        self,
        *,
        estimator: Any,
        preprocessor: Any,
        model_id: str,
        task: str,
        feature_columns: list[str],
        preprocessing: dict[str, Any],
        config: dict[str, Any],
        frozen_config_sha256: str,
        parameters: Mapping[str, Any],
        class_mapping: list[dict[str, Any]] | None = None,
        categorical_alphabet: list[Any] | tuple[Any, ...] | None = None,
        sequence_config: Mapping[str, Any] | None = None,
        snapshot_manifest_sha256: str,
        sequence_plan_manifest_sha256: str | None,
        model_state_dict: Mapping[str, Any] | None = None,
    ):
        super().__init__(
            model_id=model_id,
            task=task,
            estimator=estimator,
            preprocessor=preprocessor,
            feature_columns=feature_columns,
            preprocessing=preprocessing,
            config=config,
            frozen_config_sha256=frozen_config_sha256,
        )
        self.parameters = dict(parameters)
        self.class_mapping = None if class_mapping is None else _json_copy(class_mapping)
        self.categorical_alphabet = None if categorical_alphabet is None else _json_copy(list(categorical_alphabet))
        self.sequence_config = None if sequence_config is None else _json_copy(dict(sequence_config))
        self.snapshot_manifest_sha256 = snapshot_manifest_sha256
        self.sequence_plan_manifest_sha256 = sequence_plan_manifest_sha256
        self.model_state_dict = None if model_state_dict is None else _copy_state(model_state_dict)
        if model_id in HMM_MODEL_IDS:
            self.capabilities = {"predict": True, "predict_proba": False, "decision_function": False, "score_samples": False, "transform": False, "fit_predict": False, "fit_transform": False}
            self.preprocessing["operation_protocol"] = "observations plus explicit sequence lengths"
        elif model_id == "N01":
            self.capabilities = {"predict": True, "predict_proba": True, "decision_function": False, "score_samples": False, "transform": False, "fit_predict": False, "fit_transform": False}
        else:
            self.capabilities = {"predict": True, "predict_proba": False, "decision_function": False, "score_samples": False, "transform": False, "fit_predict": False, "fit_transform": False}

    def manifest(self) -> dict[str, Any]:
        return {
            "schema_version": 1,
            "kind": _BUNDLE_KIND,
            "model_id": self.model_id,
            "task": self.task,
            "feature_columns": [str(column) for column in self.feature_columns],
            "capabilities": dict(self.capabilities),
            "preprocessing": dict(self.preprocessing),
            "frozen_config_sha256": self.frozen_config_sha256,
            "snapshot_manifest_sha256": self.snapshot_manifest_sha256,
            "sequence_plan_manifest_sha256": self.sequence_plan_manifest_sha256,
            "serialization": "CPU-only extended bundle; no source rows, plan, validation data, or optimizer",
        }


def prepare_extended_experiment(
    config: ExperimentConfig | dict[str, Any],
    *,
    snapshot: DatasetSnapshot | None = None,
    sequence_plan: SequencePlan | None = None,
    on_event: Callable[..., None] | None = None,
    fit_scope: str = "train",
    selected_epochs: int | None = None,
    refit_provenance: Mapping[str, Any] | None = None,
) -> ExtendedExperimentSession:
    """Fit one extended estimator on allowed rows/windows and score validation."""
    config = _as_config(config)
    if config.model_id not in _EXTENDED_MODEL_IDS:
        raise ExtendedExperimentError(f"{config.model_id!r} is not an extended model")
    if fit_scope not in {"train", "train_validation"}:
        raise ExtendedExperimentError("fit_scope must be 'train' or 'train_validation'")
    if snapshot is None:
        raise ExtendedExperimentError("An owned DatasetSnapshot is required before an extended fit")
    snapshot.verify(config)
    _validate_model_task(config.model_id, config.task)
    if config.model_id in HMM_MODEL_IDS | WINDOW_DEEP_MODEL_IDS:
        if sequence_plan is None:
            raise ExtendedExperimentError(f"{config.model_id} requires a verified SequencePlan before fit")
        sequence_plan.verify(config=config, snapshot=snapshot)
    elif sequence_plan is not None:
        raise ExtendedExperimentError("N01/N02 use the DatasetSnapshot split and must not receive a SequencePlan")

    frozen_config, frozen_sha = _canonical_config(config)
    stored_config = ExperimentConfig.from_dict(frozen_config)
    parameters = validate_extended_parameters(config.model_id, config.parameters)
    if config.model_id in DEEP_MODEL_IDS:
        effective_training_seed = resolve_deep_seed(parameters, config.split.seed)
    elif config.model_id in HMM_MODEL_IDS:
        effective_training_seed = resolve_hmm_seed(parameters, config.split.seed)
    else:
        effective_training_seed = int(config.split.seed)
    if fit_scope == "train_validation" and config.model_id in DEEP_MODEL_IDS:
        if isinstance(selected_epochs, bool) or not isinstance(selected_epochs, (int, np.integer)):
            raise ExtendedExperimentError("Final deep refit requires explicit selected_epochs")
        if not 1 <= int(selected_epochs) <= parameters["max_epochs"] <= 100:
            raise ExtendedExperimentError("Final refit requires 1 <= selected_epochs <= max_epochs <= 100")
    snapshot_sha = _json_digest(snapshot.to_dict())
    plan_sha = None if sequence_plan is None else sequence_plan.plan_sha256
    feature_columns = list(snapshot.features.columns)
    class_mapping: list[dict[str, Any]] | None = None
    categorical_alphabet = None
    sequence_config = None if sequence_plan is None else sequence_plan.sequence_config.to_dict()
    preprocessor = None
    training_curves: dict[str, Any] | None = None
    preprocessing: dict[str, Any] = {"fit_policy": "train-only"}
    actual_splits = {name: np.asarray(values, dtype=np.int64).copy() for name, values in snapshot.splits.items()}

    if config.model_id in HMM_MODEL_IDS:
        assert sequence_plan is not None
        categorical_alphabet = list(sequence_plan.categorical_alphabet) if config.model_id == "H03" else None
        if fit_scope == "train_validation":
            _validate_refit_provenance(
                config.model_id,
                refit_provenance,
                selected_epochs=None,
                snapshot_sha=snapshot_sha,
                plan_sha=plan_sha,
                class_mapping=None,
                categorical_alphabet=categorical_alphabet,
                effective_training_seed=effective_training_seed,
            )
        estimator = build_hmm_estimator(
            config.model_id,
            parameters,
            seed=effective_training_seed,
            categorical_n_features=sequence_plan.categorical_n_features,
        )
        fit_partition = _combined_partition(sequence_plan, fit_scope)
        fit_hmm(estimator, config.model_id, fit_partition)
        metrics, rows = _hmm_train_validation_outputs(
            estimator, config.model_id, sequence_plan, fit_scope, fit_partition
        )
        provenance = _make_refit_provenance(
            model_id=config.model_id,
            parameters=parameters,
            snapshot_sha=snapshot_sha,
            plan_sha=plan_sha,
            categorical_alphabet=categorical_alphabet,
            effective_training_seed=effective_training_seed,
        )
        preprocessing.update({"input_protocol": "HMM observations with exact lengths", "scaler": "none"})
    elif config.model_id in WINDOW_DEEP_MODEL_IDS:
        assert sequence_plan is not None
        fit_rows = _fit_source_rows(sequence_plan, fit_scope)
        preprocessor, preprocessing = _fit_preprocessor(snapshot.features, fit_rows)
        train_x, train_y = _window_arrays(sequence_plan.train, preprocessor, feature_columns, name="train")
        if fit_scope == "train":
            validation_x, validation_y = _window_arrays(sequence_plan.validation, preprocessor, feature_columns, name="validation")
            deep_fit = fit_deep_validation(
                config.model_id,
                parameters,
                seed=effective_training_seed,
                train_x=train_x,
                train_y=train_y,
                validation_x=validation_x,
                validation_y=validation_y,
            )
            estimator = deep_fit.estimator
            training_curves = deep_fit.training_curves
            metrics, rows = _deep_train_validation_outputs(
                estimator,
                config.model_id,
                sequence_plan,
                snapshot.features,
                preprocessor,
                None,
                fit_scope,
            )
            selected_epochs = deep_fit.selected_epochs
            best_epoch = deep_fit.best_epoch
            best_weight_policy = deep_fit.best_weight_policy
            provenance = _make_refit_provenance(
                model_id=config.model_id,
                parameters=parameters,
                snapshot_sha=snapshot_sha,
                plan_sha=plan_sha,
                selected_epochs=selected_epochs,
                best_epoch=best_epoch,
                best_weight_policy=best_weight_policy,
                effective_training_seed=effective_training_seed,
            )
        else:
            if refit_provenance is None:
                raise ExtendedExperimentError("Final deep refit requires the winner's frozen refit_provenance")
            _validate_refit_provenance(
                config.model_id,
                refit_provenance,
                selected_epochs=selected_epochs,
                snapshot_sha=snapshot_sha,
                plan_sha=plan_sha,
                class_mapping=None,
                categorical_alphabet=None,
                effective_training_seed=effective_training_seed,
            )
            selected_epochs = int(selected_epochs)
            train_x, train_y = _combined_window_arrays(sequence_plan, preprocessor, feature_columns)
            estimator = refit_deep_fixed_epochs(
                config.model_id,
                parameters,
                seed=effective_training_seed,
                train_x=train_x,
                train_y=train_y,
                selected_epochs=selected_epochs,
            )
            training_curves = estimator._pyml_training_curves
            metrics, rows = _deep_refit_outputs(estimator, config.model_id, train_x, train_y)
            provenance = _json_copy(dict(refit_provenance))
        preprocessing["input_protocol"] = "float32 (batch, window, features) windows"
    else:
        assert config.model_id in {"N01", "N02"}
        if snapshot.target is None:
            raise ExtendedExperimentError(f"{config.model_id} requires a supervised target")
        train_rows = actual_splits["train"]
        mapping_override = None
        if fit_scope == "train_validation":
            if refit_provenance is None:
                raise ExtendedExperimentError("Final deep refit requires the winner's frozen refit_provenance")
            mapping_override = refit_provenance.get("class_mapping") if config.model_id == "N01" else None
        class_mapping = _class_mapping(snapshot.target.iloc[train_rows]) if config.model_id == "N01" else None
        if mapping_override is not None:
            if _normalize_class_mapping(mapping_override) != class_mapping:
                raise ExtendedExperimentError("Final refit class_mapping differs from the original train-only mapping")
            _validate_refit_provenance(
                config.model_id,
                refit_provenance,
                selected_epochs=selected_epochs,
                snapshot_sha=snapshot_sha,
                plan_sha=None,
                class_mapping=class_mapping,
                categorical_alphabet=None,
                effective_training_seed=effective_training_seed,
            )
        elif fit_scope == "train_validation":
            _validate_refit_provenance(
                config.model_id,
                refit_provenance,
                selected_epochs=selected_epochs,
                snapshot_sha=snapshot_sha,
                plan_sha=None,
                class_mapping=None,
                categorical_alphabet=None,
                effective_training_seed=effective_training_seed,
            )
        preprocessor, preprocessing = _fit_preprocessor(snapshot.features, train_rows if fit_scope == "train" else np.concatenate([actual_splits["train"], actual_splits["validation"]]))
        train_x = _transform_frame(preprocessor, snapshot.features.iloc[train_rows], name="train")
        if config.model_id == "N01":
            train_y = _encode_labels(snapshot.target.iloc[train_rows], class_mapping, context="train")
        else:
            train_y = _regression_targets(snapshot.target.iloc[train_rows], context="train")
        if fit_scope == "train":
            validation_rows = actual_splits["validation"]
            validation_x = _transform_frame(preprocessor, snapshot.features.iloc[validation_rows], name="validation")
            if config.model_id == "N01":
                validation_y = _encode_labels(snapshot.target.iloc[validation_rows], class_mapping, context="validation")
            else:
                validation_y = _regression_targets(snapshot.target.iloc[validation_rows], context="validation")
            deep_fit = fit_deep_validation(
                config.model_id,
                parameters,
                seed=effective_training_seed,
                train_x=train_x,
                train_y=train_y,
                validation_x=validation_x,
                validation_y=validation_y,
                class_count=len(class_mapping) if class_mapping is not None else None,
            )
            estimator = deep_fit.estimator
            training_curves = deep_fit.training_curves
            metrics, rows = _deep_table_train_validation_outputs(
                estimator,
                config.model_id,
                snapshot.features,
                snapshot.target,
                actual_splits,
                preprocessor,
                class_mapping,
            )
            selected_epochs = deep_fit.selected_epochs
            provenance = _make_refit_provenance(
                model_id=config.model_id,
                parameters=parameters,
                snapshot_sha=snapshot_sha,
                plan_sha=None,
                selected_epochs=selected_epochs,
                best_epoch=deep_fit.best_epoch,
                best_weight_policy=deep_fit.best_weight_policy,
                class_mapping=class_mapping,
                effective_training_seed=effective_training_seed,
            )
        else:
            assert refit_provenance is not None and selected_epochs is not None
            refit_rows = np.concatenate([actual_splits["train"], actual_splits["validation"]])
            train_x = _transform_frame(preprocessor, snapshot.features.iloc[refit_rows], name="refit")
            if config.model_id == "N01":
                train_y = _encode_labels(snapshot.target.iloc[refit_rows], class_mapping, context="refit train+validation")
            else:
                train_y = _regression_targets(snapshot.target.iloc[refit_rows], context="refit train+validation")
            estimator = refit_deep_fixed_epochs(
                config.model_id,
                parameters,
                seed=effective_training_seed,
                train_x=train_x,
                train_y=train_y,
                selected_epochs=selected_epochs,
                class_count=len(class_mapping) if class_mapping is not None else None,
            )
            training_curves = estimator._pyml_training_curves
            metrics, rows = _deep_refit_outputs(estimator, config.model_id, train_x, train_y, class_mapping=class_mapping)
            provenance = _json_copy(dict(refit_provenance))
        preprocessing["input_protocol"] = "float32 (batch, features)"

    if config.model_id in DEEP_MODEL_IDS and fit_scope == "train":
        actual_splits = {name: values.copy() for name, values in snapshot.splits.items()}
    elif fit_scope == "train_validation":
        actual_splits["train"] = np.concatenate([actual_splits["train"], actual_splits["validation"]])
        actual_splits["validation"] = np.asarray([], dtype=np.int64)

    fitted = ExtendedFittedModel(
        estimator=estimator,
        preprocessor=preprocessor,
        model_id=config.model_id,
        task=config.task,
        feature_columns=feature_columns,
        preprocessing=preprocessing,
        config=frozen_config,
        frozen_config_sha256=frozen_sha,
        parameters=parameters,
        class_mapping=class_mapping,
        categorical_alphabet=categorical_alphabet,
        sequence_config=sequence_config,
        snapshot_manifest_sha256=snapshot_sha,
        sequence_plan_manifest_sha256=plan_sha,
        model_state_dict=(deep_estimator_state(estimator) if config.model_id in DEEP_MODEL_IDS and fit_scope == "train_validation" else getattr(estimator, "_pyml_best_state_dict", None)),
    )
    training_metadata = {
        "refit_provenance": _json_copy(provenance),
        "selected_epochs": selected_epochs,
        "best_epoch": provenance.get("best_epoch"),
        "best_weight_policy": provenance.get("best_weight_policy"),
        "effective_training_seed": effective_training_seed,
    }
    if training_curves is not None:
        training_metadata["training_curves"] = _json_copy(training_curves)
    return ExtendedExperimentSession(
        config=stored_config,
        frozen_config=frozen_config,
        frozen_config_sha256=frozen_sha,
        source_path=snapshot.source_path,
        features=snapshot.features.copy(deep=True),
        target=snapshot.target.copy(deep=True) if snapshot.target is not None else None,
        splits=actual_splits,
        fitted_model=fitted,
        metrics=metrics,
        rows=rows,
        events=[],
        sequence_plan=sequence_plan,
        training_metadata=training_metadata,
        snapshot_manifest=snapshot.to_dict(),
        fit_scope=fit_scope,
    )


def score_extended_session(session: ExtendedExperimentSession, objective: Any) -> tuple[float | None, dict[str, Any]]:
    """Return the finite validation objective and its metrics for search adapters."""
    if not isinstance(session, ExtendedExperimentSession):
        raise TypeError("session must be an ExtendedExperimentSession")
    metric = getattr(objective, "metric", None)
    split = getattr(objective, "split", "validation")
    if split not in {"validation", "train_exploratory"}:
        raise ExtendedExperimentError("Extended objectives may score validation or train_exploratory only")
    partition_name = "train" if split == "train_exploratory" else "validation"
    if session.fit_scope == "train_validation" and partition_name == "validation":
        return None, {"reason": "A final-refit session has no validation score partition"}
    model_id = session.config.model_id
    if model_id in HMM_MODEL_IDS:
        if metric != "log_likelihood_per_observation":
            raise ExtendedExperimentError("HMM comparison requires log_likelihood_per_observation")
        assert session.sequence_plan is not None
        partition = getattr(session.sequence_plan, partition_name)
        value = score_hmm(session.fitted_model.estimator, model_id, partition)
        details = _hmm_metrics(value, partition)
    else:
        details = dict(session.metrics.get(partition_name, {}))
        value = details.get(metric)
    if isinstance(value, (float, int, np.number)) and math.isfinite(float(value)):
        return float(value), details
    return None, details


def refit_extended_session(
    config: ExperimentConfig | dict[str, Any],
    *,
    snapshot: DatasetSnapshot,
    parameters: Mapping[str, Any],
    selected_epochs: int | None = None,
    refit_provenance: Mapping[str, Any] | None = None,
    sequence_plan: SequencePlan | None = None,
    on_event: Callable[..., None] | None = None,
) -> ExtendedExperimentSession:
    """Build a fresh winner model using only original train+validation data."""
    config = _as_config(config)
    frozen = config.to_dict()
    frozen["parameters"] = dict(parameters)
    selected_config = ExperimentConfig.from_dict(frozen)
    if refit_provenance is None:
        raise ExtendedExperimentError("Winner refit requires the explicitly frozen refit_provenance")
    if selected_config.model_id in DEEP_MODEL_IDS and selected_epochs is None:
        raise ExtendedExperimentError("Neural winner refit requires explicit selected_epochs")
    if selected_config.model_id in HMM_MODEL_IDS and selected_epochs is not None:
        raise ExtendedExperimentError("HMM refit does not accept selected_epochs")
    return prepare_extended_experiment(
        selected_config,
        snapshot=snapshot,
        sequence_plan=sequence_plan,
        on_event=on_event,
        fit_scope="train_validation",
        selected_epochs=selected_epochs,
        refit_provenance=refit_provenance,
    )


def evaluate_extended_test(session: ExtendedExperimentSession) -> ExtendedTestComputation:
    """Compute final test metrics/rows only; core owns all once-only state."""
    if not isinstance(session, ExtendedExperimentSession):
        raise TypeError("session must be an ExtendedExperimentSession")
    if session.fitted_model.frozen_config_sha256 != session.frozen_config_sha256:
        raise ExtendedExperimentError("Extended fitted model does not match the session config hash")
    model_id = session.config.model_id
    if model_id in HMM_MODEL_IDS:
        if session.sequence_plan is None:
            raise ExtendedExperimentError("HMM final test needs the frozen SequencePlan")
        partition = session.sequence_plan.test
        per_observation, states, posterior = hmm_test_outputs(
            session.fitted_model.estimator,
            model_id,
            partition,
            categorical_alphabet=session.fitted_model.categorical_alphabet,
        )
        metrics = _hmm_metrics(per_observation, partition)
        rows = _hmm_rows(session, partition, states, posterior)
        return ExtendedTestComputation(metrics=metrics, rows=rows)

    if model_id in WINDOW_DEEP_MODEL_IDS:
        if session.sequence_plan is None:
            raise ExtendedExperimentError(f"{model_id} final test needs the frozen SequencePlan")
        partition = session.sequence_plan.test
        x = _window_inputs_for_model(session, partition, name="test")
        y_true = _regression_targets(partition.window_targets.reshape(-1), context="test")
        prediction = predict_deep(session.fitted_model.estimator, model_id, x).reshape(-1)
        actual = y_true.reshape(-1)
        metrics = _regression_metrics(actual, prediction)
        rows = _window_rows(session, partition, actual, prediction)
        return ExtendedTestComputation(metrics=metrics, rows=rows)

    test_rows = np.asarray(session.splits["test"], dtype=np.int64)
    x_raw = session.features.iloc[test_rows]
    x = _transform_frame(session.fitted_model.preprocessor, x_raw, name="test")
    if model_id == "N01":
        if session.target is None:
            raise ExtendedExperimentError("N01 final test requires its target column")
        y_true_ids = _encode_labels(session.target.iloc[test_rows], session.fitted_model.class_mapping, context="test")
        logits = predict_deep(
            session.fitted_model.estimator,
            model_id,
            x,
            class_count=len(session.fitted_model.class_mapping or []),
        )
        predicted_ids = np.argmax(logits, axis=1)
        metrics = _classification_metrics(y_true_ids, predicted_ids)
        y_true = _decode_labels(y_true_ids, session.fitted_model.class_mapping)
        y_pred = _decode_labels(predicted_ids, session.fitted_model.class_mapping)
        rows = _result_rows("test", test_rows, session.features.index[test_rows], y_true, y_pred)
        return ExtendedTestComputation(metrics=metrics, rows=rows)
    if session.target is None:
        raise ExtendedExperimentError("N02 final test requires its target column")
    y_true = _regression_targets(session.target.iloc[test_rows], context="test").reshape(-1)
    prediction = predict_deep(session.fitted_model.estimator, model_id, x).reshape(-1)
    metrics = _regression_metrics(y_true, prediction)
    rows = _result_rows("test", test_rows, session.features.index[test_rows], y_true, prediction)
    return ExtendedTestComputation(metrics=metrics, rows=rows)


def predict_extended(bundle: ExtendedFittedModel, inputs: Any, *, operation: str = "predict") -> Any:
    """Run a supported new-data operation through the fitted model's frozen protocol."""
    if not isinstance(bundle, ExtendedFittedModel):
        raise TypeError("bundle must be a loaded or fitted ExtendedFittedModel")
    model_id = bundle.model_id
    if model_id in HMM_MODEL_IDS:
        if operation == "predict_proba":
            raise ExtendedExperimentError("HMM posterior states are available with operation='posterior', not predict_proba")
        if operation not in {"predict", "decode", "posterior", "log_likelihood"}:
            raise ExtendedExperimentError(f"Unsupported HMM operation {operation!r}")
        if not isinstance(inputs, Mapping) or set(inputs) != {"observations", "lengths"}:
            raise ExtendedExperimentError("HMM inference requires exactly {'observations', 'lengths'}")
        lengths = _validate_lengths_array(inputs["lengths"], inputs["observations"])
        observations = _encode_inference_hmm_observations(bundle, inputs["observations"])
        estimator = bundle.estimator
        if operation == "predict":
            return estimator.predict(observations, lengths=lengths)
        if operation == "decode":
            return estimator.decode(observations, lengths=lengths)
        if operation == "posterior":
            return estimator.predict_proba(observations, lengths=lengths)
        return float(estimator.score(observations, lengths=lengths)) / len(observations)
    if model_id in WINDOW_DEEP_MODEL_IDS:
        windows = validate_deep_inputs(model_id, inputs, name="inference windows")
        transformed = _transform_window_array(bundle, windows, name="inference")
        if operation != "predict":
            raise ExtendedExperimentError(f"{model_id} supports predict only")
        return predict_deep(bundle.estimator, model_id, transformed).reshape(-1)
    frame = _inference_frame(bundle, inputs)
    transformed = _transform_frame(bundle.preprocessor, frame, name="inference")
    if model_id == "N01":
        count = len(bundle.class_mapping or [])
        logits = predict_deep(bundle.estimator, model_id, transformed, class_count=count)
        if operation == "predict_proba":
            return predict_proba_deep(bundle.estimator, model_id, transformed, class_count=count)
        if operation != "predict":
            raise ExtendedExperimentError("N01 supports predict and predict_proba")
        return _decode_labels(np.argmax(logits, axis=1), bundle.class_mapping)
    if operation != "predict":
        raise ExtendedExperimentError(f"{model_id} supports predict only")
    return predict_deep(bundle.estimator, model_id, transformed).reshape(-1)


def export_extended_model(bundle_or_session: ExtendedFittedModel | ExtendedExperimentSession, output_dir: str | Path) -> str:
    """Write a CPU-only reconstruction bundle without source/split/optimizer data."""
    bundle = bundle_or_session.fitted_model if isinstance(bundle_or_session, ExtendedExperimentSession) else bundle_or_session
    if not isinstance(bundle, ExtendedFittedModel):
        raise TypeError("export_extended_model requires an extended model or session")
    target = Path(output_dir).expanduser().resolve()
    if target.suffix.lower() not in {".joblib", ".pkl"}:
        target.mkdir(parents=True, exist_ok=True)
        target = target / "model.joblib"
    else:
        target.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "schema_version": 1,
        "kind": _BUNDLE_KIND,
        "model_id": bundle.model_id,
        "task": bundle.task,
        "config": bundle.config,
        "config_sha256": bundle.frozen_config_sha256,
        "parameters": dict(bundle.parameters),
        "feature_columns": list(bundle.feature_columns),
        "preprocessor": bundle.preprocessor,
        "preprocessing": dict(bundle.preprocessing),
        "class_mapping": _json_copy(bundle.class_mapping),
        "categorical_alphabet": _json_copy(bundle.categorical_alphabet),
        "sequence_config": _json_copy(bundle.sequence_config),
        "snapshot_manifest_sha256": bundle.snapshot_manifest_sha256,
        "sequence_plan_manifest_sha256": bundle.sequence_plan_manifest_sha256,
        "model_kind": "hmm_estimator" if bundle.model_id in HMM_MODEL_IDS else "deep_state_dict",
        "model_state_sha256": _model_state_digest(bundle),
        "package_versions": _package_versions(bundle.model_id),
    }
    if bundle.model_id in HMM_MODEL_IDS:
        payload["estimator"] = bundle.estimator
        payload["categorical_n_features"] = len(bundle.categorical_alphabet or []) or None
    else:
        state = bundle.model_state_dict or deep_estimator_state(bundle.estimator)
        payload["state_dict"] = _copy_state(state)
        payload["input_size"] = _model_input_size(bundle)
        payload["output_size"] = len(bundle.class_mapping or []) if bundle.model_id == "N01" else 1
    descriptor, temporary = tempfile.mkstemp(prefix=target.name + ".", suffix=".tmp", dir=target.parent)
    os.close(descriptor)
    try:
        joblib.dump(payload, temporary, compress=3)
        with open(temporary, "r+b") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return str(target)


def load_extended_model(path: str | Path) -> ExtendedFittedModel:
    """Load and verify a raw-data-free extended CPU bundle."""
    target = Path(path).expanduser().resolve()
    try:
        payload = joblib.load(target)
    except Exception as exc:
        raise ExtendedExperimentError(f"Extended model bundle could not be loaded: {target}") from exc
    if not isinstance(payload, dict) or payload.get("kind") != _BUNDLE_KIND or payload.get("schema_version") != 1:
        raise ExtendedExperimentError("Unsupported extended model bundle schema")
    model_id = payload.get("model_id")
    if model_id not in _EXTENDED_MODEL_IDS:
        raise ExtendedExperimentError("Extended model bundle contains an unsupported model id")
    forbidden = {"snapshot", "sequence_plan", "features", "target", "validation_dataset", "optimizer", "train_x", "train_y", "test_x", "test_y"}
    if forbidden.intersection(payload):
        raise ExtendedExperimentError("Extended CPU bundle contains forbidden training or test data")
    config = payload.get("config")
    if not isinstance(config, dict) or _json_digest(config) != payload.get("config_sha256"):
        raise ExtendedExperimentError("Extended model bundle config checksum does not match")
    if payload.get("model_kind") == "hmm_estimator" and model_id in HMM_MODEL_IDS:
        estimator = payload.get("estimator")
        if estimator is None:
            raise ExtendedExperimentError("HMM model bundle is missing its fitted estimator")
    elif payload.get("model_kind") == "deep_state_dict" and model_id in DEEP_MODEL_IDS:
        estimator = build_deep_from_state(
            model_id,
            payload["parameters"],
            input_size=payload["input_size"],
            output_size=payload["output_size"],
            state_dict=payload["state_dict"],
        )
    else:
        raise ExtendedExperimentError("Extended model bundle model kind does not match its model id")
    shell = ExtendedFittedModel(
        estimator=estimator,
        preprocessor=payload.get("preprocessor"),
        model_id=model_id,
        task=payload["task"],
        feature_columns=list(payload["feature_columns"]),
        preprocessing=dict(payload["preprocessing"]),
        config=config,
        frozen_config_sha256=payload["config_sha256"],
        parameters=payload["parameters"],
        class_mapping=payload.get("class_mapping"),
        categorical_alphabet=payload.get("categorical_alphabet"),
        sequence_config=payload.get("sequence_config"),
        snapshot_manifest_sha256=payload["snapshot_manifest_sha256"],
        sequence_plan_manifest_sha256=payload.get("sequence_plan_manifest_sha256"),
        model_state_dict=payload.get("state_dict"),
    )
    if _model_state_digest(shell) != payload.get("model_state_sha256"):
        raise ExtendedExperimentError("Extended model state checksum does not match")
    return shell


# The sequence-named API remains a thin compatibility alias for the five HMM
# and temporal-window models; all lifecycle and validation behavior is shared.
prepare_sequence_experiment = prepare_extended_experiment
score_sequence_session = score_extended_session
evaluate_sequence_test = evaluate_extended_test
predict_sequence = predict_extended
export_sequence_model = export_extended_model
load_sequence_model = load_extended_model


def _as_config(config: ExperimentConfig | dict[str, Any]) -> ExperimentConfig:
    if isinstance(config, dict):
        try:
            return ExperimentConfig.from_dict(config)
        except Exception as exc:
            raise ExtendedExperimentError(f"Invalid extended experiment configuration: {exc}") from exc
    if not isinstance(config, ExperimentConfig):
        raise TypeError("config must be ExperimentConfig or its JSON-compatible dictionary")
    config.validate()
    return config


def _validate_model_task(model_id: str, task: str) -> None:
    expected = "sequence_modeling" if model_id in HMM_MODEL_IDS else "classification" if model_id == "N01" else "regression"
    if task != expected:
        raise ExtendedExperimentError(f"{model_id} requires task={expected!r}, received {task!r}")


def _fit_preprocessor(features: pd.DataFrame, fit_positions: np.ndarray) -> tuple[Any, dict[str, Any]]:
    train = features.iloc[np.asarray(fit_positions, dtype=np.int64)].copy(deep=True)
    if train.empty:
        raise ExtendedExperimentError("The allowed training partition for preprocessing is empty")
    numeric: list[str] = []
    categorical: list[str] = []
    for column in features.columns:
        series = train[column]
        if series.isna().all():
            raise DatasetError(f"Training feature column {column!r} is entirely missing")
        if pd.api.types.is_datetime64_any_dtype(series.dtype) or pd.api.types.is_timedelta64_dtype(series.dtype):
            raise DatasetError(f"Date/time feature column {column!r} needs explicit conversion")
        nonmissing = series.dropna().tolist()
        all_numeric = bool(nonmissing) and all(
            isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, (bool, np.bool_))
            for value in nonmissing
        )
        if (pd.api.types.is_numeric_dtype(series.dtype) and not pd.api.types.is_bool_dtype(series.dtype)) or all_numeric:
            numeric.append(column)
        else:
            categorical.append(column)
    transformers = []
    if numeric:
        transformers.append(("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), numeric))
    if categorical:
        transformers.append(("categorical", Pipeline([
            ("stringify", CategoricalStringifier()),
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="error", sparse_output=False)),
        ]), categorical))
    if not transformers:
        raise DatasetError("No supported train-only numeric or categorical features were selected")
    preprocessor = ColumnTransformer(transformers, remainder="drop", sparse_threshold=0.0, verbose_feature_names_out=False)
    try:
        preprocessor.fit(train)
    except (TypeError, ValueError) as exc:
        raise ExtendedExperimentError(f"Train-only feature preprocessing failed: {exc}") from exc
    details = {
        "numeric_columns": [str(item) for item in numeric],
        "categorical_columns": [str(item) for item in categorical],
        "numeric_scaler": "standard",
        "fit_policy": "fit on permitted training source rows only",
        "input_adapter": "numeric median imputation and standardization; train-only one-hot categories",
    }
    return preprocessor, details


def _transform_frame(preprocessor: Any, frame: pd.DataFrame, *, name: str) -> np.ndarray:
    try:
        result = np.asarray(preprocessor.transform(frame), dtype=np.float32)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ExtendedExperimentError(f"Feature preprocessing failed for {name}: {exc}") from exc
    if result.ndim != 2 or result.shape[0] != len(frame) or result.shape[1] < 1 or not np.isfinite(result).all():
        raise ExtendedExperimentError(f"Transformed {name} features must be finite float32 rows")
    return np.ascontiguousarray(result, dtype=np.float32)


def _class_mapping(target: pd.Series) -> list[dict[str, Any]]:
    labels: dict[str, Any] = {}
    for position, value in enumerate(target.tolist()):
        safe = _normal_scalar(value, context=f"training class label at position {position}")
        labels[_json_text(safe)] = safe
    if len(labels) < 2:
        raise ExtendedExperimentError("N01 train-only class mapping requires at least two classes")
    return [{"label": labels[key], "encoded": index} for index, key in enumerate(sorted(labels))]


def _normalize_class_mapping(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list) or len(value) < 2:
        raise ExtendedExperimentError("N01 refit provenance needs a deterministic class_mapping list")
    result = []
    seen_labels = set()
    for expected_index, row in enumerate(value):
        if (
            not isinstance(row, Mapping)
            or set(row) != {"label", "encoded"}
            or isinstance(row["encoded"], bool)
            or not isinstance(row["encoded"], int)
            or row["encoded"] != expected_index
        ):
            raise ExtendedExperimentError("N01 class_mapping must be [{label, encoded}] with contiguous integer codes")
        label = _normal_scalar(row["label"], context="refit class mapping label")
        key = _json_text(label)
        if key in seen_labels:
            raise ExtendedExperimentError("N01 class_mapping contains duplicate labels")
        seen_labels.add(key)
        result.append({"label": label, "encoded": expected_index})
    if result != sorted(result, key=lambda row: _json_text(row["label"])):
        raise ExtendedExperimentError("N01 class_mapping is not in deterministic sorted order")
    return result


def _encode_labels(target: pd.Series, mapping: list[dict[str, Any]] | None, *, context: str) -> np.ndarray:
    if mapping is None:
        raise ExtendedExperimentError("A frozen train-only class_mapping is required")
    lookup = {_json_text(row["label"]): int(row["encoded"]) for row in mapping}
    result = []
    for position, value in enumerate(target.tolist()):
        safe = _normal_scalar(value, context=f"{context} class label at position {position}")
        key = _json_text(safe)
        if key not in lookup:
            raise ExtendedExperimentError(f"{context} class {safe!r} is unknown to the frozen train-only mapping")
        result.append(lookup[key])
    return np.asarray(result, dtype=np.int64)


def _decode_labels(codes: Any, mapping: list[dict[str, Any]] | None) -> np.ndarray:
    if mapping is None:
        raise ExtendedExperimentError("The fitted classifier is missing its frozen class_mapping")
    labels = [row["label"] for row in mapping]
    codes = np.asarray(codes, dtype=np.int64).reshape(-1)
    if np.any(codes < 0) or np.any(codes >= len(labels)):
        raise ExtendedExperimentError("Classifier output contains a class id outside the frozen mapping")
    return np.asarray([labels[int(code)] for code in codes], dtype=object)


def _regression_targets(target: Any, *, context: str) -> np.ndarray:
    try:
        values = pd.to_numeric(pd.Series(target), errors="raise").to_numpy(dtype=np.float32)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ExtendedExperimentError(f"Regression {context} targets must be numeric and fit float32") from exc
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ExtendedExperimentError(f"Regression {context} targets must be finite float32 values")
    return np.ascontiguousarray(values.reshape(-1, 1), dtype=np.float32)


def _window_inputs_for_model(session: ExtendedExperimentSession, partition: Any, *, name: str) -> np.ndarray:
    assert session.sequence_plan is not None
    raw = np.asarray(partition.window_inputs, dtype=object)
    if raw.ndim != 3 or raw.shape[1] != session.sequence_plan.sequence_config.window:
        raise ExtendedExperimentError(f"{name} window inputs do not match the frozen SequencePlan")
    n_windows, window, n_features = raw.shape
    flat = pd.DataFrame(raw.reshape(-1, n_features), columns=session.sequence_plan.feature_columns)
    transformed = _transform_frame(session.fitted_model.preprocessor, flat, name=f"{name} windows")
    return transformed.reshape(n_windows, window, transformed.shape[1])


def _window_arrays(partition: Any, preprocessor: Any, feature_columns: list[str], *, name: str) -> tuple[np.ndarray, np.ndarray]:
    raw = np.asarray(partition.window_inputs, dtype=object)
    if raw.ndim != 3:
        raise ExtendedExperimentError(f"{name} windows must have shape (batch, window, features)")
    batch, window, features = raw.shape
    frame = pd.DataFrame(raw.reshape(-1, features), columns=feature_columns)
    transformed = _transform_frame(preprocessor, frame, name=f"{name} windows")
    x = transformed.reshape(batch, window, transformed.shape[1])
    y = _regression_targets(np.asarray(partition.window_targets).reshape(-1), context=name)
    if len(x) != len(y):
        raise ExtendedExperimentError(f"{name} window and target counts differ")
    return x, y


def _fit_source_rows(plan: SequencePlan, fit_scope: str) -> np.ndarray:
    train = plan.train.row_positions
    if fit_scope == "train":
        return np.asarray(train, dtype=np.int64)
    return np.concatenate([np.asarray(train, dtype=np.int64), np.asarray(plan.validation.row_positions, dtype=np.int64)])


def _combined_window_arrays(plan: SequencePlan, preprocessor: Any, feature_columns: list[str]) -> tuple[np.ndarray, np.ndarray]:
    train_x, train_y = _window_arrays(plan.train, preprocessor, feature_columns, name="train refit")
    validation_x, validation_y = _window_arrays(plan.validation, preprocessor, feature_columns, name="validation refit")
    return np.concatenate([train_x, validation_x]), np.concatenate([train_y, validation_y])


def _combined_partition(plan: SequencePlan, fit_scope: str):
    from types import SimpleNamespace

    partitions = [plan.train] if fit_scope == "train" else [plan.train, plan.validation]
    observations = np.concatenate([np.asarray(item.observations) for item in partitions], axis=0)
    lengths = np.concatenate([np.asarray(item.lengths, dtype=np.int64) for item in partitions])
    encodings = {item.observation_encoding for item in partitions}
    if len(encodings) != 1:
        raise ExtendedExperimentError("Combined HMM refit partitions use different observation encodings")
    return SimpleNamespace(name=fit_scope, observations=observations, lengths=lengths, observation_encoding=next(iter(encodings)))


def _hmm_metrics(value: float, partition: Any) -> dict[str, Any]:
    lengths = hmm_lengths(partition)
    return {
        "log_likelihood_per_observation": float(value),
        "n_observations": int(sum(lengths)),
        "n_sequences": int(len(lengths)),
    }


def _hmm_train_validation_outputs(estimator: Any, model_id: str, plan: SequencePlan, fit_scope: str, fit_partition: Any):
    metrics: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    if fit_scope == "train_validation":
        value = score_hmm(estimator, model_id, fit_partition)
        metrics["refit"] = _hmm_metrics(value, fit_partition)
    else:
        for name in ("train", "validation"):
            part = getattr(plan, name)
            value = score_hmm(estimator, model_id, part)
            metrics[name] = _hmm_metrics(value, part)
    return metrics, rows


def _deep_table_train_validation_outputs(estimator, model_id, features, target, splits, preprocessor, class_mapping):
    metrics: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    for name in ("train", "validation"):
        positions = np.asarray(splits[name], dtype=np.int64)
        if not len(positions):
            continue
        x = _transform_frame(preprocessor, features.iloc[positions], name=name)
        if model_id == "N01":
            actual = _encode_labels(target.iloc[positions], class_mapping, context=name)
            logits = predict_deep(estimator, model_id, x, class_count=len(class_mapping or []))
            predicted = np.argmax(logits, axis=1)
            metrics[name] = _classification_metrics(actual, predicted)
            rows.extend(_result_rows(name, positions, features.index[positions], _decode_labels(actual, class_mapping), _decode_labels(predicted, class_mapping)))
        else:
            actual = _regression_targets(target.iloc[positions], context=name).reshape(-1)
            predicted = predict_deep(estimator, model_id, x).reshape(-1)
            metrics[name] = _regression_metrics(actual, predicted)
            rows.extend(_result_rows(name, positions, features.index[positions], actual, predicted))
    return metrics, rows


def _deep_train_validation_outputs(estimator, model_id, plan, features, preprocessor, class_mapping, fit_scope):
    metrics: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    for name in (("train", "validation") if fit_scope == "train" else ("train",)):
        partition = getattr(plan, name)
        x, y = _window_arrays(partition, preprocessor, list(plan.feature_columns), name=name)
        prediction = predict_deep(estimator, model_id, x).reshape(-1)
        actual = y.reshape(-1)
        metrics["refit" if fit_scope == "train_validation" else name] = _regression_metrics(actual, prediction)
        rows.extend(_window_rows_for_plan(plan, partition, features, actual, prediction, "refit" if fit_scope == "train_validation" else name))
    return metrics, rows


def _deep_refit_outputs(estimator, model_id, inputs, targets, *, class_mapping=None):
    if model_id == "N01":
        logits = predict_deep(estimator, model_id, inputs, class_count=len(class_mapping or []))
        predicted = np.argmax(logits, axis=1)
        actual = np.asarray(targets, dtype=np.int64)
        return {"refit": _classification_metrics(actual, predicted)}, []
    predicted = predict_deep(estimator, model_id, inputs).reshape(-1)
    actual = np.asarray(targets, dtype=np.float32).reshape(-1)
    return {"refit": _regression_metrics(actual, predicted)}, []


def _hmm_rows(session, partition, states, posterior):
    positions = np.asarray(partition.row_positions, dtype=np.int64)
    result = []
    state_count = posterior.shape[1]
    group_by_row = _group_by_row(partition)
    for index, position in enumerate(positions):
        row = {
            "split": "test",
            "row_position": int(position),
            "source_index": str(session.features.index[int(position)]),
            "hidden_state": int(states[index]),
            "group_id": _json_safe(group_by_row[int(position)]),
        }
        for state in range(state_count):
            row[f"posterior_{state + 1}"] = float(posterior[index, state])
        result.append(row)
    return result


def _window_rows(session, partition, actual, predicted):
    assert session.sequence_plan is not None
    return _window_rows_for_plan(session.sequence_plan, partition, session.features, actual, predicted, "test")


def _window_rows_for_plan(plan, partition, features, actual, predicted, split):
    result = []
    for index, target_position in enumerate(partition.window_target_positions):
        target_position = int(target_position)
        row = {
            "split": split,
            "row_position": target_position,
            "source_index": str(features.index[target_position]),
            "y_true": float(actual[index]),
            "y_pred": float(predicted[index]),
            "sequence_source_row_positions": [int(item) for item in partition.window_source_positions[index]],
            "sequence_target_row_position": target_position,
            "group_id": _json_safe(plan._group_values_by_row[target_position]),
        }
        result.append(row)
    return result


def _group_by_row(partition):
    mapping = {}
    cursor = 0
    for label, length in zip(partition.group_ids, partition.lengths):
        for position in partition.row_positions[cursor : cursor + int(length)]:
            mapping[int(position)] = label
        cursor += int(length)
    return mapping


def _make_refit_provenance(
    *, model_id, parameters, snapshot_sha, plan_sha, selected_epochs=None, best_epoch=None,
    best_weight_policy=None, class_mapping=None, categorical_alphabet=None, effective_training_seed=None,
):
    result: dict[str, Any] = {
        "schema": _PROVENANCE_SCHEMA,
        "model_id": model_id,
        "snapshot_manifest_sha256": snapshot_sha,
    }
    if plan_sha is not None:
        result["sequence_plan_manifest_sha256"] = plan_sha
    if model_id in HMM_MODEL_IDS | DEEP_MODEL_IDS:
        result["effective_training_seed"] = int(effective_training_seed)
    if model_id in DEEP_MODEL_IDS:
        result.update(
            selected_epochs=int(selected_epochs),
            best_epoch=int(best_epoch),
            early_stopping={"monitor": "valid_loss", "patience": int(parameters["patience"]), "load_best": True},
            best_weight_policy=best_weight_policy,
        )
    if class_mapping is not None:
        result["class_mapping"] = _json_copy(_normalize_class_mapping(class_mapping))
    if categorical_alphabet is not None:
        result["categorical_alphabet"] = _json_copy(list(categorical_alphabet))
    _assert_json(result, "refit_provenance")
    return result


def _validate_refit_provenance(
    model_id, provenance, *, selected_epochs, snapshot_sha, plan_sha, class_mapping, categorical_alphabet,
    effective_training_seed=None,
):
    if not isinstance(provenance, Mapping):
        raise ExtendedExperimentError("refit_provenance must be a frozen JSON object")
    payload = _json_copy(dict(provenance))
    if payload.get("schema") != _PROVENANCE_SCHEMA or payload.get("model_id") != model_id:
        raise ExtendedExperimentError("refit_provenance schema/model_id does not match the winner")
    if payload.get("snapshot_manifest_sha256") != snapshot_sha:
        raise ExtendedExperimentError("refit_provenance snapshot digest does not match this refit snapshot")
    if payload.get("sequence_plan_manifest_sha256") != plan_sha:
        raise ExtendedExperimentError("refit_provenance SequencePlan digest does not match this refit plan")
    if model_id in HMM_MODEL_IDS | DEEP_MODEL_IDS:
        recorded_seed = payload.get("effective_training_seed")
        if (
            isinstance(recorded_seed, bool)
            or not isinstance(recorded_seed, int)
            or not 0 <= recorded_seed <= (1 << 32) - 1
            or recorded_seed != effective_training_seed
        ):
            raise ExtendedExperimentError("refit_provenance effective_training_seed does not match this refit")
    if model_id in DEEP_MODEL_IDS:
        epoch = payload.get("selected_epochs")
        best_epoch = payload.get("best_epoch")
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch <= 0 or epoch != selected_epochs:
            raise ExtendedExperimentError("refit_provenance selected_epochs does not match the explicit refit argument")
        if isinstance(best_epoch, bool) or not isinstance(best_epoch, int) or best_epoch != epoch:
            raise ExtendedExperimentError("refit_provenance best_epoch must match selected_epochs")
        early_stopping = payload.get("early_stopping")
        if not isinstance(early_stopping, Mapping) or set(early_stopping) != {"monitor", "patience", "load_best"}:
            raise ExtendedExperimentError("refit_provenance must record valid_loss/load_best early stopping")
        if early_stopping.get("monitor") != "valid_loss" or early_stopping.get("load_best") is not True:
            raise ExtendedExperimentError("refit_provenance must record valid_loss/load_best early stopping")
        if isinstance(early_stopping.get("patience"), bool) or not isinstance(early_stopping.get("patience"), int) or early_stopping["patience"] <= 0:
            raise ExtendedExperimentError("refit_provenance early_stopping patience is invalid")
        if payload.get("best_weight_policy") != "early_stopping_abs_threshold_zero_load_best_valid_loss":
            raise ExtendedExperimentError("refit_provenance best_weight_policy does not match the actual restored-weight rule")
    elif model_id in HMM_MODEL_IDS and any(name in payload for name in ("selected_epochs", "best_epoch", "early_stopping", "best_weight_policy")):
        raise ExtendedExperimentError("HMM refit provenance must omit neural epoch and early-stopping fields")
    if class_mapping is not None:
        if _normalize_class_mapping(payload.get("class_mapping")) != _normalize_class_mapping(class_mapping):
            raise ExtendedExperimentError("refit_provenance class_mapping differs from the frozen train-only mapping")
    elif "class_mapping" in payload and model_id != "N01":
        raise ExtendedExperimentError("refit_provenance contains an inapplicable class_mapping")
    if categorical_alphabet is not None:
        if payload.get("categorical_alphabet") != _json_copy(list(categorical_alphabet)):
            raise ExtendedExperimentError("refit_provenance categorical_alphabet differs from the frozen train-only alphabet")
    elif "categorical_alphabet" in payload and model_id != "H03":
        raise ExtendedExperimentError("refit_provenance contains an inapplicable categorical_alphabet")
    return payload


def _validate_lengths_array(lengths: Any, observations: Any) -> list[int]:
    raw = np.asarray(observations)
    values = np.asarray(lengths)
    if values.ndim != 1 or values.size == 0 or not np.issubdtype(values.dtype, np.integer) or np.any(values <= 0) or int(values.sum()) != len(raw):
        raise ExtendedExperimentError("HMM inference lengths must be positive integers summing to observation rows")
    return [int(item) for item in values]


def _encode_inference_hmm_observations(bundle, values):
    if bundle.model_id == "H03":
        if bundle.categorical_alphabet is None:
            raise ExtendedExperimentError("H03 bundle is missing its frozen categorical alphabet")
        lookup = {_json_text(value): index for index, value in enumerate(bundle.categorical_alphabet)}
        encoded = []
        for position, value in enumerate(np.asarray(values, dtype=object).reshape(-1)):
            safe = _normal_scalar(value, context=f"inference category at position {position}")
            key = _json_text(safe)
            if key not in lookup:
                raise ExtendedExperimentError(f"Inference category {safe!r} is unknown to the frozen train-only alphabet")
            encoded.append(lookup[key])
        return np.asarray(encoded, dtype=np.int64).reshape(-1, 1)
    try:
        result = np.asarray(values, dtype=np.float32)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ExtendedExperimentError("HMM inference observations must be numeric float32") from exc
    if result.ndim != 2 or result.shape[1] < 1 or not np.isfinite(result).all():
        raise ExtendedExperimentError("HMM inference observations must be finite with shape (n, features)")
    return result


def _inference_frame(bundle, values) -> pd.DataFrame:
    columns = list(bundle.feature_columns)
    if isinstance(values, pd.DataFrame):
        frame = values.loc[:, columns].copy(deep=False)
    elif isinstance(values, Mapping):
        frame = pd.DataFrame([dict(values)])
    else:
        array = np.asarray(values, dtype=object)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        if array.ndim != 2 or array.shape[1] != len(columns):
            raise ExtendedExperimentError(f"Expected raw feature rows with shape (batch, {len(columns)})")
        frame = pd.DataFrame(array, columns=columns)
    missing = [column for column in columns if column not in frame]
    if missing:
        raise ExtendedExperimentError(f"Inference data is missing fitted feature columns: {missing}")
    return frame.loc[:, columns]


def _transform_window_array(bundle, windows, *, name):
    raw_features = len(bundle.feature_columns)
    if windows.shape[-1] != raw_features:
        raise ExtendedExperimentError(f"{bundle.model_id} {name} features must match the fitted raw feature width {raw_features}")
    batch, window, count = windows.shape
    flat = pd.DataFrame(windows.reshape(-1, count), columns=bundle.feature_columns)
    return _transform_frame(bundle.preprocessor, flat, name=f"{name} windows").reshape(batch, window, -1)


def _json_copy(value):
    if value is None:
        return None
    return json.loads(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False))


def _assert_json(value, label):
    try:
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ExtendedExperimentError(f"{label} must contain only finite JSON values") from exc


def _json_safe(value):
    if isinstance(value, np.generic):
        value = value.item()
    if value is pd.NA or value is None:
        return None
    if isinstance(value, (str, bool, int, float)):
        if isinstance(value, float) and not math.isfinite(value):
            raise ExtendedExperimentError("Result values must be finite")
        return value
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(value).isoformat()
    return str(value)


def _copy_state(state):
    if state is None:
        return None
    return {str(name): np.array(value, copy=True) for name, value in state.items()}


def _model_input_size(bundle):
    if bundle.preprocessor is not None and hasattr(bundle.preprocessor, "transformers_"):
        try:
            return int(bundle.preprocessor.transformers_[-1][2].shape[1])
        except (IndexError, AttributeError, TypeError):
            pass
    if bundle.model_state_dict:
        for name, array in bundle.model_state_dict.items():
            if name.endswith("weight_ih_l0"):
                return int(np.asarray(array).shape[1])
            if name.endswith("0.weight") and np.asarray(array).ndim == 2:
                return int(np.asarray(array).shape[1])
    dummy = _transform_frame(bundle.preprocessor, pd.DataFrame([_dummy_feature_row(bundle.feature_columns)]), name="bundle shape probe")
    return int(dummy.shape[1])


def _dummy_feature_row(columns):
    return {column: 0.0 for column in columns}


def _model_state_digest(bundle):
    digest = hashlib.sha256()
    if bundle.model_id in HMM_MODEL_IDS:
        estimator = bundle.estimator
        names = ("startprob_", "transmat_", "means_", "covars_", "weights_", "emissionprob_")
        for name in names:
            if hasattr(estimator, name):
                value = np.asarray(getattr(estimator, name))
                digest.update(name.encode("utf-8"))
                digest.update(value.dtype.str.encode("ascii"))
                digest.update(json.dumps(list(value.shape)).encode("ascii"))
                digest.update(np.ascontiguousarray(value).tobytes())
    else:
        state = bundle.model_state_dict or deep_estimator_state(bundle.estimator)
        for name, value in sorted(state.items()):
            array = np.asarray(value)
            digest.update(str(name).encode("utf-8"))
            digest.update(array.dtype.str.encode("ascii"))
            digest.update(json.dumps(list(array.shape)).encode("ascii"))
            digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def _package_versions(model_id):
    names = ["pyml-workbench", "numpy", "scikit-learn"]
    names.extend(["hmmlearn"] if model_id in HMM_MODEL_IDS else ["torch", "skorch"])
    result = {}
    for name in names:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = "not-installed"
    return result
