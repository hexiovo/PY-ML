"""JSON-configurable ensembles with preprocessing inside each member pipeline."""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin
from sklearn.pipeline import Pipeline
from sklearn import ensemble
from sklearn.utils.metaestimators import available_if


class _Ensemble:
    def _build(self, X, *, cv_override=None):
        from .catalog import build_estimator, get_model
        from .preprocessing import build_preprocessor

        if not isinstance(self.members, (list, tuple)) or len(self.members) < 2 or len(set(self.members)) != len(self.members):
            raise ValueError("members must contain at least two different model IDs")
        if self.n_jobs != 1:
            raise ValueError("Ensembles require n_jobs=1")
        overrides = self.member_parameters or {}
        if not isinstance(overrides, dict) or set(overrides) - set(self.members):
            raise ValueError("member_parameters must map member model IDs to parameter objects")
        estimators = []
        task = "classification" if isinstance(self, ClassifierMixin) else "regression"
        for name in self.members:
            spec = get_model(name)
            if spec["task"] != task or spec.get("raw_features") or name.startswith(("N", "T")):
                raise ValueError(f"{name} cannot be used as a {task} ensemble member")
            params = overrides.get(name, {})
            transformer, _ = build_preprocessor(X, name, params)
            estimators.append((name, Pipeline([("preprocess", transformer), ("model", build_estimator(name, params, seed=self.random_state))])))
        if self.kind == "voting":
            cls = ensemble.VotingClassifier if task == "classification" else ensemble.VotingRegressor
            extra = {"voting": self.voting} if task == "classification" else {}
            return cls(estimators=estimators, weights=self.weights, n_jobs=1, **extra)
        if self.kind != "stacking":
            raise ValueError("kind must be voting or stacking")
        if isinstance(self.cv, bool) or not isinstance(self.cv, int) or self.cv < 2:
            raise ValueError("Stacking cv must be an integer >= 2")
        final = build_estimator(self.final_model, seed=self.random_state)
        if get_model(self.final_model)["task"] != task or get_model(self.final_model).get("raw_features") or self.final_model.startswith(("T", "N")):
            raise ValueError("final_model must be a compatible ordinary tabular estimator")
        cls = ensemble.StackingClassifier if task == "classification" else ensemble.StackingRegressor
        return cls(estimators=estimators, final_estimator=final,
                   cv=self.cv if cv_override is None else cv_override, n_jobs=1)

    def fit(self, X, y, *, groups=None):
        cv_override = None
        self.inner_cv_audit_ = []
        if self.kind == "stacking" and groups is not None:
            from sklearn.model_selection import GroupKFold

            values = np.asarray(groups).reshape(-1)
            labels = np.asarray(y).reshape(-1)
            if len(values) != len(X):
                raise ValueError("Stacking groups must align with training rows")
            if isinstance(self.cv, bool) or not isinstance(self.cv, int) or self.cv < 2:
                raise ValueError("Stacking cv must be an integer >= 2")
            if len(np.unique(values)) < self.cv:
                raise ValueError("Stacking needs at least cv distinct training groups")
            cv_override = list(GroupKFold(n_splits=self.cv).split(X, labels, values))
            for train, valid in cv_override:
                train_groups = np.unique(values[train])
                valid_groups = np.unique(values[valid])
                if np.intersect1d(train_groups, valid_groups).size:
                    raise ValueError("Stacking inner training and validation groups overlap")
                if isinstance(self, ClassifierMixin) and set(labels[train]) != set(labels):
                    raise ValueError("Stacking inner training fold is missing a class")
                self.inner_cv_audit_.append({
                    "train_indices": train.tolist(), "validation_indices": valid.tolist(),
                    "train_groups": [str(v) for v in train_groups],
                    "validation_groups": [str(v) for v in valid_groups],
                })
        self.estimator_ = self._build(X, cv_override=cv_override)
        self.estimator_.fit(X, np.asarray(y).reshape(-1))
        self.n_features_in_ = X.shape[1]
        if hasattr(self.estimator_, "classes_"):
            self.classes_ = self.estimator_.classes_
        return self

    def predict(self, X):
        return self.estimator_.predict(X)

    @available_if(lambda self: self.kind == "stacking" or self.voting == "soft")
    def predict_proba(self, X):
        return self.estimator_.predict_proba(X)


class ConfigurableClassifier(ClassifierMixin, _Ensemble, BaseEstimator):
    def __init__(self, kind="voting", members=("C01", "C16"), member_parameters=None,
                 voting="soft", weights=None, final_model="C01", cv=5, n_jobs=1, random_state=42):
        self.kind = kind
        self.members = members
        self.member_parameters = member_parameters
        self.voting = voting
        self.weights = weights
        self.final_model = final_model
        self.cv = cv
        self.n_jobs = n_jobs
        self.random_state = random_state


class ConfigurableRegressor(RegressorMixin, _Ensemble, BaseEstimator):
    def __init__(self, kind="voting", members=("R01", "R14"), member_parameters=None,
                 weights=None, final_model="R02", cv=5, n_jobs=1, random_state=42):
        self.kind = kind
        self.members = members
        self.member_parameters = member_parameters
        self.weights = weights
        self.final_model = final_model
        self.cv = cv
        self.n_jobs = n_jobs
        self.random_state = random_state

    # Regressors do not advertise probability inference.
    predict_proba = None
