"""Approved model catalog and runtime estimator capability detection."""
from __future__ import annotations

import importlib
import json
from copy import deepcopy
from functools import lru_cache
from pathlib import Path
from typing import Any

_EXTENDED_MODEL_IDS = frozenset({"H01", "H02", "H03", "N01", "N02", "N04", "N06"})


class ModelNotAvailableError(ValueError):
    """Raised when a planned, but not implemented, model is selected."""


@lru_cache(maxsize=1)
def _catalog() -> dict[str, Any]:
    path = Path(__file__).with_name("model_catalog.json")
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


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
    return deepcopy(items)


def get_model(model_id: str) -> dict[str, Any]:
    for item in _catalog()["active_models"]:
        if item["id"] == model_id:
            return deepcopy(item)
    for item in _catalog()["deferred_models"]:
        if item["id"] == model_id:
            if item.get("implementation_status") == "available":
                return deepcopy(item)
            raise ModelNotAvailableError(f"{model_id} is planned but not implemented")
    raise ValueError(f"Unknown model id: {model_id}")


def estimator_class(model_id: str) -> type:
    spec = get_model(model_id)
    module_name, class_name = spec["implementation"].rsplit(".", 1)
    module = importlib.import_module(module_name)
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
    estimator_type = estimator_class(model_id)
    estimator = estimator_type()
    requested = dict(parameters or {})
    valid = estimator.get_params(deep=True)
    unknown = sorted(set(requested) - set(valid))
    if unknown:
        raise ValueError(f"Unsupported parameters for {model_id}: {', '.join(unknown)}")
    if "random_state" in valid:
        requested.setdefault("random_state", seed)
    if "n_jobs" in valid:
        requested.setdefault("n_jobs", 1)
    if requested:
        estimator.set_params(**requested)
    return estimator


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
