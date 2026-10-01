"""Train-fitted feature preprocessing and estimator-specific input adaptation."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import KBinsDiscretizer, MinMaxScaler, OneHotEncoder, StandardScaler

from .catalog import build_estimator, get_model
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


def build_preprocessor(
    features: pd.DataFrame,
    model_id: str,
    parameters: dict[str, Any] | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Build a dense train-fittable transformer for numeric and categorical columns."""
    if not isinstance(features, pd.DataFrame):
        raise DatasetError("Features must be a pandas DataFrame with named columns")
    if features.shape[1] == 0:
        raise DatasetError("At least one feature column is required")
    estimator = build_estimator(model_id, parameters)
    class_name = estimator.__class__.__name__
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
    details = {
        "numeric_columns": [str(column) for column in numeric_columns],
        "categorical_columns": [str(column) for column in categorical_columns],
        "numeric_scaler": scaler_name,
        "input_adapter": input_adapter,
        "fit_policy": "fit_transform on training partition only",
    }
    return transformer, details
