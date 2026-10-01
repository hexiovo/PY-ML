"""Explicit typed spaces; inactive conditional parameters are never fitted."""
from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
from typing import Any

import numpy as np
from sklearn.model_selection import ParameterGrid

from .catalog import build_estimator
from .sequence_models import EXTENDED_MODEL_IDS, HMM_MODEL_IDS, validate_extended_parameters


def json_copy(value):
    return json.loads(json.dumps(value, allow_nan=False))


@dataclass
class SearchSpace:
    fields: dict[str, dict[str, Any]] = field(default_factory=dict)
    fixed: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, payload):
        if isinstance(payload, cls):
            return cls.from_dict(payload.to_dict())
        if not isinstance(payload, dict) or set(payload) - {"fields", "fixed"}:
            raise ValueError("Space must contain fields and optional fixed parameters")
        space = cls(json_copy(payload.get("fields", {})), json_copy(payload.get("fixed", {})))
        space.validate()
        return space

    def to_dict(self):
        return json_copy({"fields": self.fields, "fixed": self.fixed})

    def order(self):
        pending, result = list(self.fields), []
        while pending:
            ready = [name for name in pending if all(parent not in pending for parent in self.fields[name].get("when", {}))]
            if not ready:
                raise ValueError("Conditional parameter dependencies contain a cycle")
            result.extend(ready)
            pending = [name for name in pending if name not in ready]
        return result

    def validate(self):
        if not isinstance(self.fields, dict) or not isinstance(self.fixed, dict):
            raise ValueError("Search fields and fixed parameters must be mappings")
        if set(self.fields) & set(self.fixed):
            raise ValueError("A parameter cannot be both searched and fixed")
        for name, spec in self.fields.items():
            if not isinstance(spec, dict) or set(spec) - {"type", "low", "high", "values", "log", "when"}:
                raise ValueError(f"Invalid field description for {name}")
            kind = spec.get("type")
            if kind not in {"real", "integer", "choice"}:
                raise ValueError(f"Unknown parameter type for {name}")
            if kind == "choice":
                if not isinstance(spec.get("values"), list) or not spec["values"]:
                    raise ValueError(f"Choice {name} needs nonempty values")
                if any(value is not None and not isinstance(value, (str, int, float, bool)) for value in spec["values"]):
                    raise ValueError("Choice values must be JSON scalars; structured parameters can be fixed")
            else:
                low, high = spec.get("low"), spec.get("high")
                if any(isinstance(v, bool) or not isinstance(v, (float, int)) or not math.isfinite(v) for v in (low, high)) or low >= high:
                    raise ValueError(f"{name} needs finite low < high")
                if kind == "integer" and any(not isinstance(v, int) for v in (low, high)):
                    raise ValueError(f"Integer {name} needs integer bounds")
                if spec.get("log") and (kind != "real" or low <= 0):
                    raise ValueError("Log spaces require positive real bounds")
                if "values" in spec and (not isinstance(spec["values"], list) or not spec["values"]):
                    raise ValueError(f"Grid values for {name} must be nonempty")
            when = spec.get("when", {})
            if not isinstance(when, dict):
                raise ValueError("when must map parent parameters to allowed values")
            for parent, values in when.items():
                if parent == name or parent not in self.fields and parent not in self.fixed or not isinstance(values, list) or not values:
                    raise ValueError(f"Invalid conditional parent for {name}")
                if parent in self.fields:
                    for value in values:
                        self._check(parent, value)
            for value in spec.get("values", []):
                self._check(name, value)
        self.order()
        for name, value in self.fixed.items():
            if name.endswith("n_jobs") and value != 1:
                raise ValueError("Search workers require n_jobs=1")
        if any(name.endswith("n_jobs") or name.endswith("random_state") for name in self.fields):
            raise ValueError("Resource threads and random seeds cannot be search variables")

    def _check(self, name, value):
        spec = self.fields[name]
        if spec["type"] == "choice":
            if value not in spec["values"]:
                raise ValueError(f"Choice {name} is out of range")
        else:
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"{name} must be finite numeric")
            if spec["type"] == "integer" and not isinstance(value, int):
                raise ValueError(f"{name} must be integer")
            if not spec["low"] <= value <= spec["high"]:
                raise ValueError(f"{name} is out of bounds")

    def canonical(self, candidate, base=None):
        result = json_copy(base or {})
        result.update(self.fixed)
        for name in self.order():
            spec = self.fields[name]
            if not all(result.get(parent) in values for parent, values in spec.get("when", {}).items()):
                result.pop(name, None)
                continue
            if name not in candidate:
                raise ValueError(f"Missing active parameter {name}")
            value = candidate[name]
            if isinstance(value, np.generic):
                value = value.item()
            self._check(name, value)
            result[name] = value
        if any(name.endswith("n_jobs") and value != 1 for name, value in result.items()):
            raise ValueError("Search workers require n_jobs=1")
        return result

    def validate_for(self, config, snapshot=None):
        self.validate()
        if config.model_id in EXTENDED_MODEL_IDS:
            try:
                names = set(validate_extended_parameters(config.model_id, config.parameters | self.fixed))
            except ValueError as exc:
                raise ValueError(str(exc)) from exc
            if config.model_id in HMM_MODEL_IDS:
                names.update({"algorithm", "implementation", "params", "init_params", "verbose"})
            max_epochs = self.fields.get("max_epochs")
            if max_epochs is not None:
                if max_epochs.get("type") not in {"integer", "choice"}:
                    raise ValueError("Deep max_epochs must use an integer or choice search field")
                limits = [max_epochs.get("high")] if max_epochs.get("type") != "choice" else max_epochs.get("values", [])
                if any(isinstance(value, bool) or not isinstance(value, int) or value > 100 for value in limits):
                    raise ValueError("Deep max_epochs search values cannot exceed 100")
        else:
            names = set(build_estimator(config.model_id).get_params(deep=True))
        unknown = set(self.fields) | set(self.fixed) | set(config.parameters)
        if unknown - names:
            raise ValueError(f"Unknown estimator parameter(s): {sorted(unknown - names)}")
        if any(name.endswith("n_jobs") and value != 1 for name, value in config.parameters.items()):
            raise ValueError("Search workers require n_jobs=1")

    def grid(self):
        choices = {}
        for name, spec in self.fields.items():
            if "values" not in spec:
                raise ValueError(f"Grid parameter {name} needs explicit finite values")
            choices[name] = spec["values"]
        return ParameterGrid(choices)

    def sample(self, rng):
        result = {}
        for name, spec in self.fields.items():
            if spec["type"] == "choice":
                result[name] = spec["values"][int(rng.integers(len(spec["values"])))]
            elif spec["type"] == "integer":
                result[name] = int(rng.integers(spec["low"], spec["high"] + 1))
            elif spec.get("log"):
                result[name] = float(np.exp(rng.uniform(np.log(spec["low"]), np.log(spec["high"]))))
            else:
                result[name] = float(rng.uniform(spec["low"], spec["high"]))
        return result


def recommended_space(model_id):
    """Small explicit defaults; no guessed ranges for arbitrary estimator fields."""
    if model_id in EXTENDED_MODEL_IDS:
        return SearchSpace()
    params = build_estimator(model_id).get_params()
    if "C" in params:
        return SearchSpace.from_dict({"fields": {"C": {"type": "real", "low": .1, "high": 10., "log": True, "values": [.1, 1., 10.]}}})
    if "alpha" in params:
        return SearchSpace.from_dict({"fields": {"alpha": {"type": "real", "low": .01, "high": 10., "log": True, "values": [.01, 1., 10.]}}})
    if "n_clusters" in params:
        return SearchSpace.from_dict({"fields": {"n_clusters": {"type": "integer", "low": 2, "high": 4, "values": [2, 3, 4]}}})
    raise ValueError("No reviewed default search space for this model; provide a typed space")
