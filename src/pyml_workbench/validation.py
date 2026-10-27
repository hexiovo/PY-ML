"""Holdout isolation and fold-local evaluation of the development partition."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit, KFold, StratifiedKFold, GroupKFold, TimeSeriesSplit


def ordered_positions(values, row_count):
    if values is None:
        return np.arange(row_count, dtype=np.int64)
    series = pd.Series(values)
    if series.isna().any():
        raise ValueError("Time ordering column contains missing values")
    if pd.api.types.is_numeric_dtype(series.dtype):
        converted = pd.to_numeric(series, errors="raise").to_numpy(dtype=float)
        if not np.isfinite(converted).all():
            raise ValueError("Time ordering must contain finite values")
    else:
        converted = pd.to_datetime(series, errors="raise", utc=True).astype("int64").to_numpy()
    if len(np.unique(converted)) != len(converted):
        raise ValueError("Chronological splitting requires unique time values; aggregate simultaneous observations first")
    return np.argsort(converted, kind="stable")


def structural_split(frame, split):
    positions = np.arange(len(frame), dtype=np.int64)
    if len(frame) < 10:
        raise ValueError("At least 10 rows are required for a three-way holdout split")
    if split.strategy == "time":
        ordered = ordered_positions(frame[split.time_column] if split.time_column else None, len(frame))
        first, second = int(len(frame) * .6), int(len(frame) * .8)
        return {"train": ordered[:first], "validation": ordered[first:second], "test": ordered[second:]}
    groups = frame[split.group_column]
    if groups.isna().any() or groups.nunique() < 5:
        raise ValueError("Group splitting needs at least five non-missing distinct groups")
    splitter = GroupShuffleSplit(n_splits=1, test_size=.4, random_state=split.seed)
    train, remaining = next(splitter.split(positions, groups=groups))
    splitter = GroupShuffleSplit(n_splits=1, test_size=.5, random_state=split.seed)
    val, test = next(splitter.split(remaining, groups=groups.iloc[remaining]))
    return {"train": train, "validation": remaining[val], "test": remaining[test]}


def cross_validate_snapshot(config, snapshot, *, before_fit=None):
    from .catalog import build_estimator
    from .experiment import _classification_score_cache
    from .metrics import classification_metrics, regression_metrics, prediction_quantile
    from .preprocessing import build_preprocessor

    settings = config.evaluation
    settings.validate()
    dev = np.concatenate([snapshot.splits["train"], snapshot.splits["validation"]])
    scheme = settings.strategy
    if scheme == "auto":
        scheme = config.split.strategy if config.split.strategy != "random" else ("stratified" if config.task == "classification" else "kfold")
    structure = getattr(snapshot, "structure", None)
    if scheme == "time":
        # The snapshot partitions already preserve strict temporal order.
        splitter = TimeSeriesSplit(n_splits=settings.folds, gap=settings.gap)
        folds = splitter.split(dev)
    elif scheme == "group":
        groups = structure[config.split.group_column].iloc[dev]
        if groups.nunique() < settings.folds:
            raise ValueError("Not enough development groups for the requested fold count")
        folds = GroupKFold(n_splits=settings.folds).split(dev, groups=groups)
    elif scheme == "stratified":
        if config.task != "classification":
            raise ValueError("Stratified cross-validation requires classification")
        target = snapshot.target.iloc[dev]
        if target.value_counts().min() < settings.folds:
            raise ValueError("Each development class needs at least one sample per stratified fold")
        folds = StratifiedKFold(n_splits=settings.folds, shuffle=True, random_state=config.split.seed).split(dev, target)
    else:
        folds = KFold(n_splits=settings.folds, shuffle=True, random_state=config.split.seed).split(dev)
    reports = []
    for index, (train_local, val_local) in enumerate(folds, 1):
        train, validation = dev[train_local], dev[val_local]
        estimator = build_estimator(config.model_id, config.parameters, seed=config.split.seed)
        transformer, _ = build_preprocessor(snapshot.features.iloc[train], config.model_id, config.parameters)
        ready_train = transformer.fit_transform(snapshot.features.iloc[train])
        ready_validation = transformer.transform(snapshot.features.iloc[validation])
        actual_train = snapshot.target.iloc[train].to_numpy()
        if before_fit:
            before_fit(index)
        estimator.fit(ready_train, actual_train)
        if config.model_id.startswith("T"):
            prediction = estimator.forecast(len(validation), offset=settings.gap)
        else:
            prediction = np.asarray(estimator.predict(ready_validation)).reshape(-1)
        actual = snapshot.target.iloc[validation].to_numpy()
        if config.task == "classification":
            scores = _classification_score_cache(estimator, ready_validation, validation)
            metrics = classification_metrics(actual, prediction, scores)
        else:
            metrics = regression_metrics(actual, prediction,
                quantile=prediction_quantile(estimator),
                training_series=np.asarray(actual_train, dtype=float) if scheme == "time" else None,
                seasonal_period=getattr(estimator, "mase_period_", 1),
                horizons=np.arange(settings.gap + 1, settings.gap + len(validation) + 1) if config.model_id.startswith("T") else None)
        reports.append({"fold": index, "train_positions": train.tolist(), "validation_positions": validation.tolist(), "metrics": metrics})
    from .metrics import METRICS
    means, deviations = {}, {}
    for metric in METRICS[config.task]:
        values = [report["metrics"].get(metric) for report in reports]
        finite = all(isinstance(value, (int, float)) and np.isfinite(value) for value in values)
        means[metric] = float(np.mean(values)) if finite else None
        deviations[metric] = float(np.std(values)) if finite else None
    return {"scheme": scheme, "fold_count": len(reports), "gap": settings.gap,
            "scope": "train_validation_only; holdout test excluded", "mean": means,
            "std": deviations, "folds": reports, "test_evaluation_count": 0}
