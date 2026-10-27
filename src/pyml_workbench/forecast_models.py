"""Fixed-origin, univariate forecasting adapters with portable sklearn parameters."""
from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted

FORECAST_MODEL_IDS = frozenset({"T01", "T02", "T03", "T04", "T05"})


class ForecastRegressor(RegressorMixin, BaseEstimator):
    """X determines output length; forecasts depend only on fitted target history.

    No validation/test observations are appended during prediction. Input rows
    represent consecutive future steps after the fitted origin.
    """
    def __init__(self, method="naive", seasonal_period=12, trend=None, seasonal=None,
                 order=(1, 1, 0), seasonal_order=(1, 0, 0, 12), max_iter=100):
        self.method = method
        self.seasonal_period = seasonal_period
        self.trend = trend
        self.seasonal = seasonal
        self.order = order
        self.seasonal_order = seasonal_order
        self.max_iter = max_iter

    def fit(self, X, y):
        values = np.asarray(y, dtype=float).reshape(-1)
        if len(values) < 3 or not np.isfinite(values).all():
            raise ValueError("Forecasting needs at least three finite training targets")
        if isinstance(self.seasonal_period, bool) or not isinstance(self.seasonal_period, int) or self.seasonal_period < 1:
            raise ValueError("seasonal_period must be a positive integer")
        if self.method not in {"naive", "seasonal_naive", "exponential", "arima", "sarima"}:
            raise ValueError("Unknown forecast method")
        if self.method == "seasonal_naive" and len(values) < self.seasonal_period:
            raise ValueError("Seasonal naive needs at least seasonal_period training observations")
        self.history_ = values.copy()
        self.n_features_in_ = X.shape[1]
        self.result_ = None
        if self.method == "exponential":
            from statsmodels.tsa.holtwinters import ExponentialSmoothing
            self.result_ = ExponentialSmoothing(values, trend=self.trend, seasonal=self.seasonal,
                                                seasonal_periods=self.seasonal_period if self.seasonal else None,
                                                initialization_method="estimated").fit()
        elif self.method in {"arima", "sarima"}:
            from statsmodels.tsa.arima.model import ARIMA
            seasonal = tuple(self.seasonal_order) if self.method == "sarima" else (0, 0, 0, 0)
            if len(self.order) != 3 or len(seasonal) != 4 or any(isinstance(v, bool) or not isinstance(v, int) or v < 0 for v in (*self.order, *seasonal)):
                raise ValueError("ARIMA orders must contain nonnegative integers")
            self.result_ = ARIMA(values, order=tuple(self.order), seasonal_order=seasonal, trend=self.trend).fit(method_kwargs={"maxiter": self.max_iter})
        return self

    def forecast(self, steps, *, offset=0):
        check_is_fitted(self, "history_")
        if steps < 0 or offset < 0:
            raise ValueError("Forecast steps and offset must be nonnegative")
        total = int(steps + offset)
        if not total:
            return np.empty(0)
        if self.method == "naive":
            values = np.repeat(self.history_[-1], total)
        elif self.method == "seasonal_naive":
            values = np.resize(self.history_[-self.seasonal_period:], total)
        else:
            values = np.asarray(self.result_.forecast(total), dtype=float)
        if not np.isfinite(values).all():
            raise ValueError("Forecast produced non-finite predictions")
        return values[offset:]

    def predict(self, X):
        return self.forecast(len(X))

    @property
    def mase_period_(self):
        if self.method == "seasonal_naive" or self.method == "exponential" and self.seasonal:
            return self.seasonal_period
        if self.method == "sarima" and any(self.seasonal_order[:3]):
            return max(1, int(self.seasonal_order[3]))
        return 1

    def predict_in_sample(self):
        check_is_fitted(self, "history_")
        if self.method in {"naive", "seasonal_naive"}:
            lag = self.seasonal_period if self.method == "seasonal_naive" else 1
            # Leading rows have no historical prediction and are excluded from
            # fitted-value metrics, rather than scored against invented values.
            return np.r_[np.full(lag, np.nan), self.history_[:-lag]]
        return np.asarray(self.result_.fittedvalues, dtype=float)
