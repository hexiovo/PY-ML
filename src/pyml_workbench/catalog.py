"""Approved model catalog and runtime estimator capability detection."""
from __future__ import annotations

import importlib
import importlib.util
import inspect
import json
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Any

_EXTENDED_MODEL_IDS = frozenset({"H01", "H02", "H03", "N01", "N02", "N04", "N06"})
_LABEL_ENCODED_CLASSIFIERS = frozenset({"C25"})
_CATBOOST_MODEL_IDS = frozenset({"C27", "R26"})
_CATBOOST_PARAMETER_ALIASES = {
    "random_state": ("random_seed", lambda value: value),
    "n_jobs": ("thread_count", lambda value: value),
    "silent": ("verbose", lambda value: None if value is None else not value),
}
_CATBOOST_FALLBACK_PARAMETERS = (
    "iterations", "learning_rate", "depth", "loss_function", "border_count",
    "l2_leaf_reg", "random_strength", "random_seed", "thread_count", "verbose",
    "logging_level", "allow_writing_files", "task_type", "devices", "bootstrap_type",
    "subsample", "class_weights", "auto_class_weights", "classes_count", "eval_metric",
    "custom_loss", "custom_metric", "one_hot_max_size", "grow_policy", "min_data_in_leaf",
    "max_leaves", "ignored_features", "train_dir", "save_snapshot", "snapshot_file",
    "snapshot_interval", "use_best_model", "best_model_min_trees", "od_type", "od_wait",
    "od_pval", "metric_period", "verbose_eval", "has_time", "allow_const_label",
    "feature_weights", "monotone_constraints", "text_features", "embedding_features",
)


class ModelNotAvailableError(ValueError):
    """Raised when a planned, but not implemented, model is selected."""


class OptionalModelDependencyError(ImportError):
    """Raised when an optional estimator library is missing or cannot load."""


@lru_cache(maxsize=1)
def _catalog() -> dict[str, Any]:
    path = Path(__file__).with_name("model_catalog.json")
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _with_runtime_availability(item: dict[str, Any]) -> dict[str, Any]:
    extra = item.get("optional_extra")
    module_names = item.get("optional_modules", ())
    if isinstance(module_names, str):
        module_names = (module_names,)
    elif not isinstance(module_names, (list, tuple)):
        module_names = ()
    legacy_module = item.get("optional_module")
    if legacy_module:
        module_names = (legacy_module, *module_names)
    module_names = tuple(dict.fromkeys(
        module_name for module_name in module_names
        if isinstance(module_name, str) and module_name
    ))
    if not extra or not module_names:
        return item
    missing = []
    for module_name in module_names:
        try:
            present = importlib.util.find_spec(module_name) is not None
        except (ImportError, ModuleNotFoundError, ValueError):
            present = False
        if not present:
            missing.append(module_name)
    item["runtime_available"] = not missing
    if missing:
        item["unavailable_modules"] = missing
        item["unavailable_reason"] = (
            f"当前环境缺少可选依赖 {', '.join(missing)}；"
            f"安装项目 extra '{extra}' 后可使用。"
        )
    else:
        item.pop("unavailable_modules", None)
        item.pop("unavailable_reason", None)
    return item


def list_models(task: str | None = None, *, include_deferred: bool = False) -> list[dict[str, Any]]:
    """Return runnable entries, optionally including still-deferred catalog entries."""
    data = _catalog()
    items = list(data["active_models"])
    available_deferred = [item for item in data["deferred_models"] if item.get("implementation_status") == "available"]
    items.extend(available_deferred)
    if include_deferred:
        items.extend(item for item in data["deferred_models"] if item.get("implementation_status") != "available")
    if task is not None:
        items = [item for item in items if item["task"] == task]
    return [_with_runtime_availability(item) for item in deepcopy(items)]


def get_model(model_id: str) -> dict[str, Any]:
    for item in _catalog()["active_models"]:
        if item["id"] == model_id:
            return _with_runtime_availability(deepcopy(item))
    for item in _catalog()["deferred_models"]:
        if item["id"] == model_id:
            if item.get("implementation_status") == "available":
                return _with_runtime_availability(deepcopy(item))
            raise ModelNotAvailableError(f"{model_id} is planned but not implemented")
    raise ValueError(f"Unknown model id: {model_id}")


def estimator_class(model_id: str) -> type:
    spec = get_model(model_id)
    module_name, class_name = spec["implementation"].rsplit(".", 1)
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        extra = spec.get("optional_extra")
        if extra:
            raise OptionalModelDependencyError(
                f"{model_id} requires the optional '{extra}' extra ({module_name}); "
                f"the dependency is missing or could not be loaded: {exc}"
            ) from exc
        raise
    estimator_type = getattr(module, class_name)
    if not isinstance(estimator_type, type):
        raise TypeError(f"Catalog entry {model_id} does not resolve to a class")
    return estimator_type


def build_estimator(model_id: str, parameters: dict[str, Any] | None = None, *, seed: int = 42):
    """Instantiate one registered sklearn estimator with validated parameters."""
    if model_id in _EXTENDED_MODEL_IDS:
        raise ValueError(
            f"{model_id} requires the extended sequence/deep model builder; "
            "it cannot be constructed as a sklearn estimator"
        )
    spec = get_model(model_id)
    estimator_type = estimator_class(model_id)
    estimator = estimator_type()
    if spec.get("target_adapter") == "label_encoded":
        from .label_encoding import LabelEncodedClassifier

        estimator = LabelEncodedClassifier(estimator)
    requested = dict(parameters or {})
    if model_id in _CATBOOST_MODEL_IDS:
        requested = _normalize_catboost_parameters(requested)
        valid_names = set(estimator_parameter_names(model_id, estimator))
        valid_names.update(name for name, (canonical, _convert) in _CATBOOST_PARAMETER_ALIASES.items()
                           if canonical in valid_names)
        valid = set(estimator.get_params(deep=True)) | valid_names
    else:
        valid = estimator.get_params(deep=True)
    unknown = sorted(set(requested) - set(valid))
    if unknown:
        raise ValueError(f"Unsupported parameters for {model_id}: {', '.join(unknown)}")
    seed_parameter = spec.get("seed_parameter", "random_state")
    thread_parameter = spec.get("thread_parameter", "n_jobs")
    if seed_parameter in valid:
        requested.setdefault(seed_parameter, seed)
    if thread_parameter in valid:
        requested.setdefault(thread_parameter, 1)
    for name, value in spec.get("builder_defaults", {}).items():
        if model_id in _CATBOOST_MODEL_IDS and name == "verbose" and "logging_level" in requested:
            # CatBoost treats logging_level and verbose as mutually exclusive ways
            # to select output verbosity; keep the user's explicit choice.
            continue
        if name in valid:
            requested.setdefault(name, value)
    if requested:
        estimator.set_params(**requested)
    return estimator


def estimator_parameter_names(model_id: str, estimator=None) -> tuple[str, ...]:
    """Return the explicit parameter contract used by the parameter interface."""
    if estimator is None:
        estimator = build_estimator(model_id)
    configured = estimator.get_params(deep=True)
    if model_id not in _CATBOOST_MODEL_IDS:
        return tuple(configured)

    names: list[str] = []
    try:
        signature = inspect.signature(type(estimator).__init__)
        names = [
            name
            for name, parameter in signature.parameters.items()
            if name != "self"
            and parameter.kind not in {
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            }
        ]
    except (TypeError, ValueError):
        pass
    if not names:
        names = list(_CATBOOST_FALLBACK_PARAMETERS)
    for name in configured:
        if name not in names:
            names.append(name)
    return tuple(names)


def _normalize_catboost_parameters(parameters: dict[str, Any]) -> dict[str, Any]:
    """Translate common sklearn aliases and reject contradictory CatBoost aliases."""
    normalized = dict(parameters)
    if "logging_level" in normalized and ("verbose" in normalized or "silent" in normalized):
        raise ValueError("CatBoost parameters 'logging_level' and 'verbose'/'silent' conflict")
    for alias, (canonical, convert) in _CATBOOST_PARAMETER_ALIASES.items():
        if alias not in normalized:
            continue
        alias_value = convert(normalized.pop(alias))
        if canonical in normalized:
            try:
                agrees = bool(normalized[canonical] == alias_value)
            except (TypeError, ValueError):
                agrees = normalized[canonical] is alias_value
            if not agrees:
                raise ValueError(
                    f"Conflicting CatBoost parameters '{alias}' and '{canonical}'"
                )
        else:
            normalized[canonical] = alias_value
    return normalized


def model_capabilities(model_id: str, estimator=None) -> dict[str, bool]:
    """Report methods supported by the actual estimator instance and its settings."""
    if estimator is None:
        estimator = build_estimator(model_id)
    operations = {
        "predict": callable(getattr(estimator, "predict", None)),
        "predict_proba": callable(getattr(estimator, "predict_proba", None)),
        "decision_function": callable(getattr(estimator, "decision_function", None)),
        "score_samples": callable(getattr(estimator, "score_samples", None)),
        "transform": callable(getattr(estimator, "transform", None)),
        "fit_predict": callable(getattr(estimator, "fit_predict", None)),
        "fit_transform": callable(getattr(estimator, "fit_transform", None)),
    }
    # LocalOutlierFactor exposes these methods but raises unless novelty=True.
    if model_id == "A03" and not bool(getattr(estimator, "novelty", False)):
        operations.update(predict=False, decision_function=False, score_samples=False)
    # SVC has a predict_proba method even when probability estimates are disabled.
    if model_id == "C07" and not bool(getattr(estimator, "probability", False)):
        operations["predict_proba"] = False
    return operations
