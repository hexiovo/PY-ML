"""Estimator adapters for external classifiers with stricter target-label APIs."""
from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.metaestimators import available_if


class LabelEncodedClassifier(ClassifierMixin, BaseEstimator):
    """Fit an estimator on contiguous integers while exposing original labels.

    XGBoost's scikit-learn classifier requires integer-encoded class labels. This
    wrapper keeps that implementation detail inside the persisted estimator so
    training, search, reload, and prediction all use the same mapping.
    """

    def __init__(self, estimator: Any):
        self.estimator = estimator

    def get_params(self, deep: bool = True) -> dict[str, Any]:
        if not deep:
            return {"estimator": self.estimator}
        return self.estimator.get_params(deep=True)

    def set_params(self, **params: Any) -> "LabelEncodedClassifier":
        if not params:
            return self
        params = dict(params)
        estimator = params.pop("estimator", None)
        if estimator is not None:
            self.estimator = estimator
        nested = {
            key[len("estimator__"):]: value
            for key, value in params.items()
            if key.startswith("estimator__")
        }
        direct = {key: value for key, value in params.items() if not key.startswith("estimator__")}
        self.estimator.set_params(**(direct | nested))
        return self

    def _validate_params(self) -> None:
        validate = getattr(self.estimator, "_validate_params", None)
        if callable(validate):
            validate()

    def fit(self, X, y, **fit_params: Any) -> "LabelEncodedClassifier":
        labels = np.asarray(y)
        if labels.ndim != 1:
            labels = labels.reshape(-1)
        self.label_encoder_ = LabelEncoder()
        try:
            encoded = self.label_encoder_.fit_transform(labels)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"XGBoost classification labels cannot be encoded: {exc}") from exc
        self.estimator.fit(X, encoded, **fit_params)
        self.classes_ = self.label_encoder_.classes_.copy()
        base_classes = np.asarray(getattr(self.estimator, "classes_", ()), dtype=object).reshape(-1)
        if base_classes.size:
            try:
                base_ids = base_classes.astype(np.int64)
            except (TypeError, ValueError) as exc:
                raise ValueError("XGBoost returned non-integer encoded classes after fitting") from exc
        else:
            base_ids = np.arange(len(self.classes_), dtype=np.int64)
        self._score_column_order_ = tuple(
            int(np.flatnonzero(base_ids == class_id)[0])
            for class_id in range(len(self.classes_))
            if np.count_nonzero(base_ids == class_id) == 1
        )
        if len(self._score_column_order_) != len(self.classes_):
            raise ValueError("XGBoost encoded class columns do not match the fitted original labels")
        return self

    def _require_fitted(self) -> None:
        if not hasattr(self, "label_encoder_"):
            raise RuntimeError("LabelEncodedClassifier must be fitted before inference")

    def predict(self, X):
        self._require_fitted()
        encoded = np.asarray(self.estimator.predict(X)).reshape(-1)
        try:
            integer_labels = encoded.astype(np.int64)
        except (TypeError, ValueError) as exc:
            raise ValueError("XGBoost predictions are not encoded class IDs") from exc
        if not np.array_equal(encoded, integer_labels):
            raise ValueError("XGBoost predictions contain non-integer encoded class IDs")
        return self.label_encoder_.inverse_transform(integer_labels)

    def predict_proba(self, X):
        self._require_fitted()
        scores = np.asarray(self.estimator.predict_proba(X))
        if scores.ndim != 2 or scores.shape[1] != len(self._score_column_order_):
            raise ValueError("XGBoost probability columns do not match the fitted original labels")
        return scores[:, self._score_column_order_]

    @available_if(lambda self: callable(getattr(self.estimator, "decision_function", None)))
    def decision_function(self, X):
        self._require_fitted()
        scores = np.asarray(self.estimator.decision_function(X))
        if scores.ndim == 2 and scores.shape[1] == len(self._score_column_order_):
            return scores[:, self._score_column_order_]
        # Preserve OvO or otherwise non-class-column scores as-is. The plotting
        # layer must decline them rather than infer OvR from coincidental shape.
        return scores

    @property
    def feature_importances_(self):
        return self.estimator.feature_importances_

    @property
    def coef_(self):
        return self.estimator.coef_

