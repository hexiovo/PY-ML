"""Train-fitted feature preprocessing and estimator-specific input adaptation."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import FastICA, KernelPCA, PCA, TruncatedSVD
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, KBinsDiscretizer, MinMaxScaler, OneHotEncoder, StandardScaler
from sklearn.random_projection import GaussianRandomProjection

from .catalog import build_estimator, get_model
from .config import SUPPORTED_REDUCTION_METHODS
from .data import DatasetError


class CategoricalStringifier(BaseEstimator, TransformerMixin):
    """Normalize mixed scalar category values while preserving missing cells."""

    def fit(self, values, y=None):
        array = np.asarray(values, dtype=object)
        self.n_features_in_ = array.shape[1] if array.ndim == 2 else 1
        return self

    def transform(self, values):
        array = np.asarray(values, dtype=object)
        if array.ndim == 1:
            array = array.reshape(-1, 1)
        normalized = np.empty(array.shape, dtype=object)
        for index, value in np.ndenumerate(array):
            normalized[index] = np.nan if pd.isna(value) else str(value)
        return normalized

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return np.asarray([f"x{index}" for index in range(self.n_features_in_)], dtype=object)
        return np.asarray(input_features, dtype=object)


class IntegerizeCategories(BaseEstimator, TransformerMixin):
    """Convert ordinal bin IDs to the integer domain required by CategoricalNB."""

    def fit(self, values, y=None):
        array = np.asarray(values)
        self.n_features_in_ = array.shape[1]
        return self

    def transform(self, values):
        return np.asarray(values, dtype=np.int64)

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return np.asarray([f"x{index}" for index in range(self.n_features_in_)], dtype=object)
        return np.asarray(input_features, dtype=object)


class _LegacyPcaMethod(str):
    """Mark PCA selected through the original pca_components option."""


def _resolve_reduction_settings(
    *,
    pca_components: int | None,
    reduction_method: str | None,
    reduction_components: int | None,
) -> tuple[str | None, int | None]:
    if pca_components is not None and (
        type(pca_components) is not int or pca_components < 1
    ):
        raise DatasetError("pca_components must be a positive integer or None")
    has_method = reduction_method is not None
    has_components = reduction_components is not None
    if has_method != has_components:
        raise DatasetError(
            "reduction_method and reduction_components must be provided together"
        )
    if has_method:
        if not isinstance(reduction_method, str) or reduction_method not in SUPPORTED_REDUCTION_METHODS:
            raise DatasetError(f"Unsupported reduction method: {reduction_method!r}")
        if type(reduction_components) is not int or reduction_components < 1:
            raise DatasetError("reduction_components must be a positive integer")
        if pca_components is not None and (
            reduction_method != "pca" or reduction_components != pca_components
        ):
            raise DatasetError(
                "pca_components conflicts with reduction_method/reduction_components"
            )
        return reduction_method, reduction_components
    if pca_components is not None:
        return _LegacyPcaMethod("pca"), pca_components
    return None, None


def _make_reducer(method: str, n_components: int, *, seed: int) -> Any:
    if method == "pca":
        return PCA(n_components=n_components, random_state=seed)
    if method == "truncated_svd":
        return TruncatedSVD(
            n_components=n_components,
            algorithm="randomized",
            random_state=seed,
        )
    if method == "fast_ica":
        return FastICA(
            n_components=n_components,
            whiten="unit-variance",
            max_iter=500,
            random_state=seed,
        )
    if method == "gaussian_random_projection":
        return GaussianRandomProjection(
            n_components=n_components,
            random_state=seed,
        )
    if method == "kernel_pca":
        return KernelPCA(
            n_components=n_components,
            kernel="rbf",
            gamma=None,
            random_state=seed,
        )
    raise DatasetError(f"Unsupported reduction method: {method!r}")


def append_reducer(
    preprocessor: Any,
    method: str | None,
    n_components: int | None,
    *,
    seed: int = 42,
) -> Any:
    """Append a selected train-fitted dimensionality reduction stage."""
    if method is None and n_components is None:
        return preprocessor
    if method not in SUPPORTED_REDUCTION_METHODS:
        raise DatasetError(f"Unsupported reduction method: {method!r}")
    if type(n_components) is not int or n_components < 1:
        raise DatasetError("Reduction component count must be a positive integer")
    if type(seed) is not int or seed < 0:
        raise DatasetError("Reduction seed must be a non-negative integer")
    step_name = "pca" if isinstance(method, _LegacyPcaMethod) else "reduction"
    return Pipeline([
        ("features", preprocessor),
        (step_name, _make_reducer(method, n_components, seed=seed)),
    ])


def append_pca(preprocessor: Any, n_components: int | None) -> Any:
    """Compatibility wrapper for callers that still append legacy PCA."""
    return append_reducer(
        preprocessor,
        _LegacyPcaMethod("pca") if n_components is not None else None,
        n_components,
    )


def _reducer_method(reducer: Any) -> str:
    methods = (
        (PCA, "pca"),
        (TruncatedSVD, "truncated_svd"),
        (FastICA, "fast_ica"),
        (GaussianRandomProjection, "gaussian_random_projection"),
        (KernelPCA, "kernel_pca"),
    )
    for estimator_type, method in methods:
        if isinstance(reducer, estimator_type):
            return method
    raise DatasetError(
        f"Unsupported dimensionality-reduction estimator: {reducer.__class__.__name__}"
    )


def _check_reduction_dimensions(
    values: np.ndarray,
    *,
    method: str,
    n_components: int,
) -> tuple[int, int, int | None]:
    if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] < 1:
        raise DatasetError("Dimensionality reduction requires a non-empty two-dimensional feature matrix")
    rows, columns = values.shape
    if n_components >= columns:
        raise DatasetError(
            f"{method} n_components={n_components} must be less than the "
            f"{columns} preprocessed input features to reduce dimensionality"
        )
    if method in {"pca", "truncated_svd", "fast_ica"} and n_components > min(rows, columns):
        raise DatasetError(
            f"{method} n_components={n_components} exceeds the fit limit "
            f"{min(rows, columns)} (min({rows} fitted rows, {columns} preprocessed features))"
        )
    if method == "kernel_pca" and n_components >= rows:
        raise DatasetError(
            f"kernel_pca n_components={n_components} must be less than the "
            f"{rows} fitted rows because the centered kernel matrix has at most {rows - 1} components"
        )
    input_rank = None
    if method == "fast_ica":
        centered = values - np.mean(values, axis=0, keepdims=True)
        input_rank = int(np.linalg.matrix_rank(centered))
        if n_components > input_rank:
            raise DatasetError(
                f"fast_ica n_components={n_components} exceeds the centered "
                f"training feature rank {input_rank}"
            )
    return rows, columns, input_rank


def _effective_rank(values: Any, *, rows: int, columns: int) -> int:
    singular_values = np.asarray(values, dtype=np.float64).reshape(-1)
    if singular_values.size == 0:
        return 0
    largest = float(np.max(np.abs(singular_values)))
    if not np.isfinite(largest) or largest == 0:
        return 0
    tolerance = np.finfo(np.float64).eps * max(rows, columns) * largest
    return int(np.count_nonzero(np.abs(singular_values) > tolerance))


def _reducer_configuration(method: str, reducer: Any) -> dict[str, Any]:
    configuration: dict[str, Any] = {
        "method": method,
        "components": int(reducer.n_components),
    }
    if method == "pca":
        random_state = reducer.random_state
        configuration["random_state"] = (
            None if random_state is None else int(random_state)
        )
    elif method == "truncated_svd":
        configuration.update(
            algorithm=reducer.algorithm,
            random_state=int(reducer.random_state),
        )
    elif method == "fast_ica":
        configuration.update(
            whiten=reducer.whiten,
            max_iter=int(reducer.max_iter),
            random_state=int(reducer.random_state),
        )
    elif method == "gaussian_random_projection":
        configuration["random_state"] = int(reducer.random_state)
    elif method == "kernel_pca":
        configuration.update(
            kernel=reducer.kernel,
            gamma=reducer.gamma,
            random_state=int(reducer.random_state),
        )
    return configuration


def fit_transform_preprocessor(
    preprocessor: Any,
    values: pd.DataFrame,
    *,
    fit_scope: str,
) -> tuple[np.ndarray, dict[str, Any] | None]:
    """Fit preprocessing and an optional reducer only on permitted fit rows."""
    steps = getattr(preprocessor, "named_steps", {})
    legacy_pca = "pca" in steps and "reduction" not in steps
    reducer = steps.get("reduction")
    if reducer is None:
        # Accept pipelines serialized by earlier PCA-only releases.
        reducer = steps.get("pca")
    if reducer is None:
        return preprocessor.fit_transform(values), None

    features = steps.get("features")
    if features is None or not callable(getattr(features, "fit_transform", None)):
        raise DatasetError("Dimensionality reduction requires a train-fittable feature transformer")
    prepared = np.asarray(features.fit_transform(values))
    if prepared.ndim != 2 or prepared.shape[0] != len(values) or prepared.shape[1] < 1:
        raise DatasetError(
            "Dimensionality reduction input must be a non-empty two-dimensional feature matrix"
        )
    method = _reducer_method(reducer)
    n_components = reducer.n_components
    if type(n_components) is not int or n_components < 1:
        raise DatasetError("Reduction component count must be a positive integer")
    if legacy_pca:
        fit_row_count, input_dimensions = prepared.shape
        if n_components > min(fit_row_count, input_dimensions):
            raise DatasetError(
                f"pca n_components={n_components} exceeds the fit limit "
                f"{min(fit_row_count, input_dimensions)} "
                f"(min({fit_row_count} fitted rows, {input_dimensions} preprocessed features))"
            )
        input_rank = None
    else:
        fit_row_count, input_dimensions, input_rank = _check_reduction_dimensions(
            prepared,
            method=method,
            n_components=n_components,
        )
    try:
        # Fit each already-constructed pipeline stage once. Pipeline.transform()
        # then uses these fitted objects for validation, test, CV and new rows.
        reducer.fit(prepared)
        transformed = np.asarray(reducer.transform(prepared))
    except ValueError as exc:
        raise DatasetError(f"{method} could not fit the permitted training rows: {exc}") from exc
    if transformed.ndim != 2 or transformed.shape[0] != fit_row_count or transformed.shape[1] < 1:
        raise DatasetError(f"{method} did not produce a non-empty two-dimensional feature matrix")

    effective_rank = input_rank
    if method in {"pca", "truncated_svd"}:
        effective_rank = _effective_rank(
            reducer.singular_values_,
            rows=fit_row_count,
            columns=input_dimensions,
        )
    elif method == "kernel_pca":
        effective_rank = _effective_rank(
            reducer.eigenvalues_,
            rows=fit_row_count,
            columns=fit_row_count,
        )
    if not legacy_pca and effective_rank is not None and n_components > effective_rank:
        raise DatasetError(
            f"{method} n_components={n_components} exceeds the fitted effective rank {effective_rank}"
        )

    metadata: dict[str, Any] = {
        "method": method,
        "configuration": _reducer_configuration(method, reducer),
        "requested_components": int(n_components),
        "actual_components": int(transformed.shape[1]),
        "input_dimensions": int(input_dimensions),
        "output_dimensions": int(transformed.shape[1]),
        "fit_scope": str(fit_scope),
        "fit_row_count": int(fit_row_count),
    }
    if effective_rank is not None:
        metadata["effective_rank"] = int(effective_rank)
    variance_ratio = getattr(reducer, "explained_variance_ratio_", None)
    if variance_ratio is not None:
        values = np.asarray(variance_ratio, dtype=np.float64)
        finite = np.isfinite(values)
        metadata["explained_variance_ratio"] = [
            float(value) if bool(is_finite) else None
            for value, is_finite in zip(values, finite)
        ]
        metadata["explained_variance_ratio_sum"] = (
            float(values.sum()) if bool(finite.all()) else None
        )
    if method == "fast_ica":
        metadata["iterations"] = int(reducer.n_iter_)
        metadata["converged"] = bool(reducer.n_iter_ < reducer.max_iter)
    return transformed, metadata


def build_preprocessor(
    features: pd.DataFrame,
    model_id: str,
    parameters: dict[str, Any] | None = None,
    *,
    pca_components: int | None = None,
    reduction_method: str | None = None,
    reduction_components: int | None = None,
    seed: int = 42,
) -> tuple[Any, dict[str, Any]]:
    """Build a dense train-fittable transformer for numeric and categorical columns."""
    if not isinstance(features, pd.DataFrame):
        raise DatasetError("Features must be a pandas DataFrame with named columns")
    if features.shape[1] == 0:
        raise DatasetError("At least one feature column is required")
    if get_model(model_id).get("raw_features"):
        return FunctionTransformer(validate=False), {"input_adapter": "Member-local train-fitted pipelines", "fit_policy": "Each inner fold fits its own preprocessing"}
    estimator = build_estimator(model_id, parameters)
    class_name = estimator.__class__.__name__
    method, components = _resolve_reduction_settings(
        pca_components=pca_components,
        reduction_method=reduction_method,
        reduction_components=reduction_components,
    )
    if method is not None:
        if class_name in {"MultinomialNB", "ComplementNB"}:
            message = (
                "PCA can produce negative components; MultinomialNB and ComplementNB require non-negative inputs"
                if method == "pca"
                else "Dimensionality reduction can produce negative components; MultinomialNB and ComplementNB require non-negative inputs"
            )
            raise DatasetError(
                message
            )
        if class_name == "CategoricalNB":
            message = (
                "PCA produces continuous signed components; CategoricalNB requires non-negative integer category IDs"
                if method == "pca"
                else "Dimensionality reduction produces continuous component values; CategoricalNB requires non-negative integer category IDs"
            )
            raise DatasetError(
                message
            )
    numeric_columns: list[Any] = []
    categorical_columns: list[Any] = []
    unsupported_columns: list[str] = []
    for column in features.columns:
        series = features[column]
        if series.isna().all():
            raise DatasetError(f"Feature column {column!r} is entirely missing")
        dtype = series.dtype
        if pd.api.types.is_datetime64_any_dtype(dtype) or pd.api.types.is_timedelta64_dtype(dtype):
            unsupported_columns.append(str(column))
        elif pd.api.types.is_numeric_dtype(dtype) and not pd.api.types.is_bool_dtype(dtype):
            numeric_columns.append(column)
        else:
            categorical_columns.append(column)
    if unsupported_columns:
        raise DatasetError(
            f"Date/time feature columns need explicit numeric or categorical conversion: {unsupported_columns}"
        )

    nonnegative_models = {"MultinomialNB", "ComplementNB", "NMF"}
    numeric_steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    scaler_name = "minmax_clip" if class_name in nonnegative_models else "standard"
    if class_name in nonnegative_models:
        numeric_steps.append(("scaler", MinMaxScaler(clip=True)))
    else:
        numeric_steps.append(("scaler", StandardScaler()))

    transformers: list[tuple[str, Any, list[Any]]] = []
    if numeric_columns:
        transformers.append(("numeric", Pipeline(numeric_steps), numeric_columns))
    if categorical_columns:
        categorical_pipe = Pipeline([
            ("stringify", CategoricalStringifier()),
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])
        transformers.append(("categorical", categorical_pipe, categorical_columns))
    if not transformers:
        raise DatasetError("No supported numeric or categorical feature columns were selected")
    columns = ColumnTransformer(
        transformers,
        remainder="drop",
        sparse_threshold=0.0,
        verbose_feature_names_out=False,
    )
    if class_name == "CategoricalNB":
        bins = max(2, min(5, len(features)))
        transformer = Pipeline([
            ("columns", columns),
            ("discretize", KBinsDiscretizer(n_bins=bins, encode="ordinal", strategy="quantile", subsample=None)),
            ("integerize", IntegerizeCategories()),
        ])
        input_adapter = f"numeric quantile bins ({bins}) plus one-hot categories, encoded as non-negative integer category IDs"
    else:
        transformer = columns
        if class_name in nonnegative_models:
            input_adapter = "numeric median imputation + train-fitted clipped MinMax scaling; categories one-hot encoded"
        else:
            input_adapter = "numeric median imputation + train-fitted standard scaling; categories one-hot encoded"
    transformer = append_reducer(transformer, method, components, seed=seed)
    details = {
        "numeric_columns": [str(column) for column in numeric_columns],
        "categorical_columns": [str(column) for column in categorical_columns],
        "numeric_scaler": scaler_name,
        "input_adapter": input_adapter,
        "fit_policy": "fit_transform on training partition only",
    }
    if method is not None:
        details["reduction"] = {
            "method": method,
            "requested_components": int(components),
            "seed": int(seed),
        }
    return transformer, details
