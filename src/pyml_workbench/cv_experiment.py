"""Development-only CV sessions and a separate frozen development refit."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from .catalog import build_estimator, get_model
from .config import ConfigError, ExperimentConfig
from .cross_validation import CVPlan, FoldEvaluation, evaluate_cv_candidate, make_cv_plan
from .experiment import (
    DatasetSnapshot, ExperimentError, ExperimentSession, FittedModel,
    _canonical_config, _classification_explanation_cache, _classification_metrics,
    _classification_score_cache, _json_digest, _regression_metrics, _result_rows,
    _table_digest, _target_for_task, build_snapshot, configured_preprocessor,
)
from .preprocessing import fit_transform_preprocessor


@dataclass
class CrossValidationSnapshot(DatasetSnapshot):
    plan: CVPlan

    def verify(self, config=None):
        super().verify(config)
        split_config = self.manifest["split_config"]
        expected = make_cv_plan(len(self.features), self.target, task=self.manifest["task"],
                                n_splits=split_config["n_splits"], seed=split_config["seed"])
        if self.plan.to_dict() != expected.to_dict() or self.manifest.get("cv_plan") != expected.to_dict():
            raise ExperimentError("CV snapshot fold assignments were modified")
        if set(self.splits) != {"development", "test"} or any(
            not np.array_equal(self.splits[name], positions)
            for name, positions in (("development", expected.development_indices), ("test", expected.test_indices))
        ):
            raise ExperimentError("CV snapshot development/test positions were modified")


@dataclass
class CrossValidationSession(ExperimentSession):
    cv_metadata: dict[str, Any] | None = None
    cv_payload_sha256: str = ""
    cv_snapshot: CrossValidationSnapshot | None = None
    cv_refit_session: ExperimentSession | None = None


def _cv_payload(session):
    return {"rows": session.rows, "plot_scores": session.plot_scores,
            "metrics": session.metrics, "cv_metadata": session.cv_metadata}


def _refit_oof_payload(session):
    return {"rows": [row for row in session.rows if row["split"] == "oof"],
            "plot_scores": {"oof": session.plot_scores["oof"]} if "oof" in session.plot_scores else {},
            "cv_metadata": getattr(session, "cv_metadata", None),
            "validation": session.metrics.get("selection_validation"), "oof": session.metrics.get("oof")}


def seal_cv_refit_session(session):
    """Bind retained OOF cache independently of later final-test metrics."""
    session.cv_oof_payload_sha256 = _json_digest(_refit_oof_payload(session))
    return session


def verify_cv_refit_session(session, snapshot=None, config=None):
    """Check a final model's development fit and retained OOF cache without inference."""
    if not isinstance(session, ExperimentSession) or session.fitted_model is None or session.fit_scope != "train_validation":
        raise ExperimentError("Expected the final development-refit CV model session")
    expected_config = ExperimentConfig.from_dict(config) if isinstance(config, dict) else config or session.config
    frozen, digest = _canonical_config(expected_config)
    if (session.config.to_dict() != frozen or session.frozen_config != frozen
            or session.frozen_config_sha256 != digest or session.fitted_model.config != frozen
            or session.fitted_model.frozen_config_sha256 != digest):
        raise ExperimentError("Final CV model configuration identity was modified")
    snapshot = snapshot if snapshot is not None else getattr(session, "cv_snapshot", None)
    if not isinstance(snapshot, CrossValidationSnapshot):
        raise ExperimentError("Final CV session is missing its owned CV snapshot")
    snapshot.verify(expected_config)
    if (session.snapshot_manifest != snapshot.to_dict() or set(session.splits) != {"train", "validation", "test"}
            or not np.array_equal(session.splits["train"], snapshot.plan.development_indices)
            or len(session.splits["validation"]) or not np.array_equal(session.splits["test"], snapshot.plan.test_indices)):
        raise ExperimentError("Final CV session development/test identity was modified")
    payload = _refit_oof_payload(session)
    if not getattr(session, "cv_oof_payload_sha256", None) or _json_digest(payload) != session.cv_oof_payload_sha256:
        raise ExperimentError("Final CV session OOF cache was modified or is incomplete")
    if session.metrics.get("cv") != payload["cv_metadata"]:
        raise ExperimentError("Final CV metadata differs from its retained fold evaluation")
    # Reuse semantic validation against the original snapshot using a model-free
    # view. No final-fit/train/test metric enters this OOF evaluation view.
    view = CrossValidationSession(
        config=expected_config, frozen_config=frozen, frozen_config_sha256=digest,
        source_path=session.source_path, features=session.features, target=session.target,
        splits={name: values for name, values in snapshot.splits.items()}, fitted_model=None,
        metrics={"validation": payload["validation"], "oof": payload["oof"], "cv": payload["cv_metadata"]},
        rows=payload["rows"], events=[], snapshot_manifest=session.snapshot_manifest,
        fit_scope="cv_development", plot_scores=payload["plot_scores"], cv_metadata=payload["cv_metadata"], cv_snapshot=snapshot,
    )
    view.cv_payload_sha256 = _json_digest(_cv_payload(view))
    verify_cv_session(view, snapshot, expected_config)
    return session


def verify_cv_session(session, snapshot=None, config=None):
    """Verify a complete evaluation-only candidate without inventing a model."""
    if not isinstance(session, CrossValidationSession) or session.fitted_model is not None:
        raise ExperimentError("Expected a CV evaluation session without a final fitted model")
    snapshot = snapshot if snapshot is not None else session.cv_snapshot
    expected_config = ExperimentConfig.from_dict(config) if isinstance(config, dict) else config or session.config
    frozen, digest = _canonical_config(expected_config)
    if (session.frozen_config != frozen or session.frozen_config_sha256 != digest
            or session.config.to_dict() != frozen or session.fit_scope != "cv_development"):
        raise ExperimentError("CV session configuration identity was modified")
    if session.frozen or session.finalized or session.result is not None or session.test_evaluation_count or "test" in session.metrics:
        raise ExperimentError("CV evaluation session has entered final test evaluation")
    if not isinstance(session.cv_metadata, dict) or _json_digest(_cv_payload(session)) != session.cv_payload_sha256:
        raise ExperimentError("CV evaluation payload was modified")
    if snapshot is not None:
        if not isinstance(snapshot, CrossValidationSnapshot):
            raise ExperimentError("CV candidate requires a CV snapshot")
        snapshot.verify(expected_config)
        if (session.snapshot_manifest != snapshot.to_dict()
                or _table_digest(session.features) != _table_digest(snapshot.features)
                or _table_digest(session.target) != _table_digest(_target_for_task(snapshot.target, expected_config.task, build_estimator(expected_config.model_id, expected_config.parameters, seed=expected_config.split.seed)))):
            raise ExperimentError("CV session data/snapshot identity was modified")
    observed_data = {"features": _table_digest(session.features), "target": _table_digest(session.target), "objective_labels": None}
    # Classification targets retain their original type. Numeric regression
    # normalization can change dtype, so its original snapshot hash is checked
    # above and a separate session data fingerprint is always bound below.
    if session.cv_metadata.get("session_data_sha256") != _json_digest(observed_data):
        raise ExperimentError("CV session source rows were modified")
    plan = make_cv_plan(len(session.features), session.target, task=expected_config.task,
                        n_splits=expected_config.split.n_splits, seed=expected_config.split.seed)
    if session.snapshot_manifest.get("cv_plan") != plan.to_dict():
        raise ExperimentError("CV session split/fold plan differs from its snapshot")
    if set(session.splits) != {"development", "test"} or not np.array_equal(session.splits["development"], plan.development_indices) or not np.array_equal(session.splits["test"], plan.test_indices):
        raise ExperimentError("CV session development/test positions were modified")
    positions = [row.get("row_position") for row in session.rows]
    if positions != plan.development_indices.tolist() or any(row.get("split") != "oof" for row in session.rows):
        raise ExperimentError("CV OOF rows do not cover every development row exactly once in stable order")
    metric_fn = _classification_metrics if expected_config.task == "classification" else _regression_metrics
    actual = session.target.iloc[plan.development_indices].to_numpy()
    recorded_actual = np.asarray([row["y_true"] for row in session.rows])
    if not np.array_equal(actual, recorded_actual):
        raise ExperimentError("CV OOF true labels differ from their development rows")
    aggregate = metric_fn(actual, np.asarray([row["y_pred"] for row in session.rows]))
    if aggregate != session.metrics.get("oof") or aggregate != session.cv_metadata.get("oof_metrics"):
        raise ExperimentError("CV OOF aggregate metrics differ from their predictions")
    fold_metrics = session.cv_metadata.get("fold_metrics", [])
    if len(fold_metrics) != plan.n_splits:
        raise ExperimentError("CV candidate has incomplete fold metrics")
    prediction = np.asarray([row["y_pred"] for row in session.rows])
    local_positions = {int(position): index for index, position in enumerate(plan.development_indices)}
    for number, (_, validation) in enumerate(plan.folds):
        local = [local_positions[int(position)] for position in validation]
        expected_fold = _finite_metrics(metric_fn(actual[local], prediction[local]))
        if expected_fold != fold_metrics[number]:
            raise ExperimentError("CV fold metrics differ from their OOF rows")
    mean, std = _summarize_metrics(fold_metrics)
    if (mean != session.metrics.get("validation") or mean != session.cv_metadata.get("fold_metric_mean")
            or std != session.cv_metadata.get("fold_metric_std_ddof1")
            or session.cv_metadata.get("validation_source") != "cv_fold_mean"
            or session.cv_metadata.get("oof_source") != "cv_oof"):
        raise ExperimentError("CV fold mean/std/source metadata is inconsistent")
    for key, value in plan.to_dict().items():
        if session.cv_metadata.get(key) != value:
            raise ExperimentError("CV metadata differs from its deterministic fold plan")
    return session


def _finite_metrics(metrics):
    return {name: (None if isinstance(value, (int, float)) and not np.isfinite(value) else value)
            for name, value in metrics.items()}


def _summarize_metrics(folds):
    mean, std = {}, {}
    for name in folds[0]:
        values = [fold.get(name) for fold in folds]
        if all(isinstance(value, (int, float)) and np.isfinite(value) for value in values):
            mean[name] = float(np.mean(values))
            std[name] = float(np.std(values, ddof=1))
    return mean, std


def _validated_snapshot(config, snapshot):
    snapshot = build_snapshot(config) if snapshot is None else snapshot
    if not isinstance(snapshot, CrossValidationSnapshot):
        raise ExperimentError("K-fold strategy requires its own development/test snapshot")
    snapshot.verify(config)
    if snapshot.objective_labels is not None:
        raise ConfigError("CV classification/regression does not accept independent objective labels")
    if get_model(config.model_id)["task"] != config.task:
        raise ConfigError("Model/task mismatch")
    return snapshot


def _fresh_fit(config, features, target, positions, fit_scope):
    """Instantiate every learned stage anew and fit only the provided rows."""
    estimator = build_estimator(config.model_id, config.parameters, seed=config.split.seed)
    fit_frame = features.iloc[positions].copy(deep=True)
    transformer, specification, options = configured_preprocessor(fit_frame, config)
    ready, learned_metadata = fit_transform_preprocessor(transformer, fit_frame, fit_scope=fit_scope)
    specification.update(fit_policy="fit only on permitted training rows", fit_scope=fit_scope,
                         fit_positions=[int(value) for value in positions], configuration=options)
    if learned_metadata is not None:
        specification["learned_transform_metadata"] = learned_metadata
    estimator.fit(ready, target.iloc[positions].to_numpy())
    return estimator, transformer, specification, ready


def prepare_cv_experiment(config, *, snapshot=None, on_event=None, should_cancel=None, deadline=None):
    frozen, digest = _canonical_config(config)
    config = ExperimentConfig.from_dict(frozen)
    snapshot = _validated_snapshot(config, snapshot)
    features = snapshot.features.copy(deep=True)
    target = _target_for_task(snapshot.target.copy(deep=True), config.task,
                              build_estimator(config.model_id, config.parameters, seed=config.split.seed))
    plan = snapshot.plan
    metric_fn = _classification_metrics if config.task == "classification" else _regression_metrics
    primary = "balanced_accuracy" if config.task == "classification" else "rmse"
    fold_metrics, fold_preprocessing, score_descriptions = [], [], []
    events = ["dataset_loaded", "target_validated", "cv_split_created"]

    def callback(train, valid, fold_number):
        estimator, preprocessor, preprocessing, _ = _fresh_fit(config, features, target, train, "cv_fold_train")
        transformed = preprocessor.transform(features.iloc[valid].copy(deep=True))
        prediction = np.asarray(estimator.predict(transformed)).reshape(-1)
        metrics = _finite_metrics(metric_fn(target.iloc[valid].to_numpy(), prediction))
        fold_metrics.append(metrics)
        fold_preprocessing.append(preprocessing)
        score = _classification_score_cache(estimator, transformed, valid) if config.task == "classification" else {"available": False, "reason": "Regression has no class scores"}
        score_descriptions.append(score)
        name = f"cv_fold_{fold_number + 1}_completed"
        events.append(name)
        if on_event:
            on_event(name)
        columns = tuple(item["class_label"] for item in score.get("score_column_map", []))
        # No estimator/preprocessor is retained by the completed fold callback.
        return FoldEvaluation(primary_score=metrics[primary], predictions=prediction,
                              scores=score.get("values") if score.get("available") else None,
                              score_columns=columns or None,
                              scores_unavailable_reason=score.get("reason"),
                              metadata={"metrics": metrics, "preprocessing": preprocessing,
                                        "score_type": score.get("score_type"), "score_semantics": score.get("semantics")})

    result = evaluate_cv_candidate(plan, callback, should_cancel=should_cancel, deadline=deadline,
                                   primary_metric_name=primary)
    mean, std = _summarize_metrics(fold_metrics)
    development = plan.development_indices
    oof = metric_fn(target.iloc[development].to_numpy(), result.oof_predictions)
    metadata = dict(plan.to_dict(), strategy="kfold", primary_metric=primary,
                    primary_direction="max" if config.task == "classification" else "min",
                    parameters=config.parameters, preprocess_spec=fold_preprocessing[0].get("configuration", {}),
                    fold_metrics=fold_metrics, fold_metric_mean=mean, fold_metric_std_ddof1=std,
                    fold_preprocessing=fold_preprocessing, validation_source="cv_fold_mean",
                    oof_source="cv_oof", oof_metrics=oof,
                    session_data_sha256=_json_digest({"features": _table_digest(features), "target": _table_digest(target), "objective_labels": None}))
    plot_scores = {}
    if config.task == "classification":
        descriptions = {(value.get("score_type"), value.get("semantics")) for value in score_descriptions}
        available = result.oof_scores is not None and len(descriptions) == 1
        first = score_descriptions[0]
        score_payload = {"available": available, "row_positions": development.tolist(),
                         "class_order": list(plan.class_labels), "source": "cv_oof"}
        if available:
            score_payload.update(score_type=first["score_type"], semantics=first["semantics"],
                                 score_column_map=[{"column_index": index, "class_label": label} for index, label in enumerate(result.score_columns)],
                                 values=result.oof_scores.tolist())
        else:
            score_payload["reason"] = result.scores_unavailable_reason or "Score semantics changed between folds; OOF curves are unavailable."
        plot_scores["oof"] = score_payload
    session = CrossValidationSession(
        config=config, frozen_config=frozen, frozen_config_sha256=digest, source_path=snapshot.source_path,
        features=features, target=target, splits={name: values.copy() for name, values in snapshot.splits.items()},
        fitted_model=None, metrics={"validation": mean, "oof": oof, "cv": metadata,
                                   "selection": {"mode": "development-only K-fold; no final model yet", "frozen_model_id": config.model_id,
                                                 "frozen_parameters": config.parameters, "frozen_config_sha256": digest, "validation_source": "cv_fold_mean"}},
        rows=_result_rows("oof", development, features.index[development], target.iloc[development].to_numpy(), result.oof_predictions),
        events=events + ["cv_candidate_completed", "selection_frozen"], snapshot_manifest=snapshot.to_dict(),
        fit_scope="cv_development", plot_scores=plot_scores, cv_metadata=metadata, cv_snapshot=snapshot,
    )
    session.cv_payload_sha256 = _json_digest(_cv_payload(session))
    return verify_cv_session(session, snapshot, config)


def refit_cv_experiment(config, *, snapshot=None, on_event=None):
    """Fit the whole 80% development set; test rows are untouched until evaluate_test."""
    frozen, digest = _canonical_config(config)
    config = ExperimentConfig.from_dict(frozen)
    snapshot = _validated_snapshot(config, snapshot)
    features = snapshot.features.copy(deep=True)
    target = _target_for_task(snapshot.target.copy(deep=True), config.task,
                              build_estimator(config.model_id, config.parameters, seed=config.split.seed))
    development = snapshot.plan.development_indices
    estimator, transformer, specification, ready = _fresh_fit(config, features, target, development, "development_refit")
    fitted = FittedModel(model_id=config.model_id, task=config.task, estimator=estimator, preprocessor=transformer,
                         feature_columns=list(features.columns), preprocessing=specification, config=frozen, frozen_config_sha256=digest)
    prediction = np.asarray(estimator.predict(ready)).reshape(-1)
    metric_fn = _classification_metrics if config.task == "classification" else _regression_metrics
    events = ["dataset_loaded", "target_validated", "cv_split_created", "preprocessor_fit_development_only",
              "estimator_fit_development_only", "refit_metrics_computed", "selection_frozen", "refit_development_only"]
    if on_event:
        for name in events:
            on_event(name)
    session = ExperimentSession(config, frozen, digest, snapshot.source_path, features, target,
                                {"train": development.copy(), "validation": np.array([], dtype=np.int64), "test": snapshot.plan.test_indices.copy()},
                                fitted, {"refit": metric_fn(target.iloc[development].to_numpy(), prediction),
                                         "selection": {"mode": "frozen CV parameters; development refit", "validation_source": "cv_fold_mean"}},
                                _result_rows("refit", development, features.index[development], target.iloc[development].to_numpy(), prediction), events,
                                snapshot_manifest=snapshot.to_dict(), fit_scope="train_validation")
    if config.task == "classification":
        session.plot_scores["refit"] = _classification_score_cache(estimator, ready, development)
    session.plot_explanations = _classification_explanation_cache(estimator, transformer, task=config.task,
                                                                 output_label=config.dataset.target_column)
    session.cv_metadata = snapshot.plan.to_dict()
    session.cv_snapshot = snapshot
    return session


def refit_cv_session(session, expected_config=None, on_event=None):
    """Freeze a single configured CV candidate and return its one final refit.

    Durable GUI/worker callers replace their evaluation session with this
    returned frozen ExperimentSession before invoking the one-test lifecycle.
    Repeated calls on the same evaluation object reuse the final model/result.
    """
    from .experiment import freeze_experiment

    verify_cv_session(session, config=expected_config)
    if session.cv_refit_session is not None:
        freeze_experiment(session.cv_refit_session, expected_config or session.config)
        return session.cv_refit_session
    final = refit_cv_experiment(session.config, snapshot=session.cv_snapshot, on_event=on_event)
    final.cv_metadata = dict(session.cv_metadata)
    final.metrics["cv"] = dict(session.cv_metadata)
    final.metrics["oof"] = dict(session.metrics["oof"])
    final.metrics["selection_validation"] = dict(session.metrics["validation"])
    final.rows = [dict(row) for row in session.rows] + final.rows
    final.plot_scores.update(session.plot_scores)
    seal_cv_refit_session(final)
    freeze_experiment(final, expected_config or session.config)
    session.cv_refit_session = final
    return final
