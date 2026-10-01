"""Thin task adapter that exposes only operations supported by sklearn."""
from __future__ import annotations

from typing import Any

from .catalog import build_estimator, model_capabilities


class UnsupportedOperationError(RuntimeError):
    """Raised instead of claiming an estimator supports an unavailable method."""


class EstimatorAdapter:
    def __init__(self, model_id: str, parameters: dict[str, Any] | None = None, *, seed: int = 42):
        self.model_id = model_id
        self.estimator = build_estimator(model_id, parameters, seed=seed)
        self._fitted = False

    @property
    def capabilities(self) -> dict[str, bool]:
        return model_capabilities(self.model_id, self.estimator)

    def fit(self, X, y=None) -> "EstimatorAdapter":
        if y is None:
            self.estimator.fit(X)
        else:
            self.estimator.fit(X, y)
        self._fitted = True
        return self

    def fit_predict(self, X, y=None):
        self._require("fit_predict")
        result = self.estimator.fit_predict(X) if y is None else self.estimator.fit_predict(X, y)
        self._fitted = True
        return result

    def fit_transform(self, X, y=None):
        self._require("fit_transform")
        result = self.estimator.fit_transform(X) if y is None else self.estimator.fit_transform(X, y)
        self._fitted = True
        return result

    def predict(self, X):
        self._require_fitted()
        self._require("predict")
        return self.estimator.predict(X)

    def predict_proba(self, X):
        self._require_fitted()
        self._require("predict_proba")
        return self.estimator.predict_proba(X)

    def transform(self, X):
        self._require_fitted()
        self._require("transform")
        return self.estimator.transform(X)

    def decision_function(self, X):
        self._require_fitted()
        self._require("decision_function")
        return self.estimator.decision_function(X)

    def score_samples(self, X):
        self._require_fitted()
        self._require("score_samples")
        return self.estimator.score_samples(X)

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError(f"{self.model_id} must be fitted before inference")

    def _require(self, operation: str) -> None:
        if not self.capabilities.get(operation, False):
            raise UnsupportedOperationError(f"{self.model_id} does not support {operation} for this configuration")
