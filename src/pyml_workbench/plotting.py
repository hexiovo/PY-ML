"""Immutable plot inputs, allowlisted chart specifications, and pure payloads.

Matplotlib is deliberately absent from module import and payload construction.  The
renderer at the bottom of this module imports it only when a chart is actually
rendered.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from decimal import Decimal
from enum import Enum
from fractions import Fraction
import math
from numbers import Real
from typing import Any, Iterable
from uuid import UUID

import numpy as np


_SHA_FIELDS = (
    "source_sha256",
    "data_sha256",
    "split_sha256",
    "frozen_config_sha256",
    "batch_snapshot_sha256",
    "receipt_manifest_sha256",
    "receipt_file_sha256",
    "sequence_plan_sha256",
    "cache_payload_sha256",
)
_PARTITIONS = frozenset({"all", "train", "validation", "test"})
_SOURCE_KINDS = frozenset({"loaded_eda", "session_cache", "batch_cache"})
_OWNER_KINDS = frozenset({"loaded_dataset", "session", "batch_job"})


def _is_pandas_missing(value: object) -> bool:
    cls = type(value)
    return cls.__module__.startswith("pandas.") and cls.__name__ in {"NAType", "NaTType"}


def _freeze_scalar(value: object, *, allow_tuple: bool = False) -> object:
    """Normalize NumPy scalars and reject mutable or arbitrary payload objects."""
    if isinstance(value, np.generic):
        value = value.item()
    if allow_tuple and isinstance(value, tuple):
        return tuple(_freeze_scalar(item, allow_tuple=True) for item in value)
    if value is None or isinstance(
        value,
        (bool, int, float, str, bytes, Decimal, Fraction, datetime, date, time, timedelta, UUID),
    ):
        return value
    if _is_pandas_missing(value):
        return value
    raise TypeError(
        "plot values and labels must be immutable Python scalars"
        + (" or tuples of immutable scalars" if allow_tuple else "")
        + f"; got {type(value).__module__}.{type(value).__qualname__}"
    )


def _freeze_tuple(values: Iterable[object], *, allow_tuple: bool = False) -> tuple[object, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError("a string or bytes object is not a plot value sequence")
    return tuple(_freeze_scalar(value, allow_tuple=allow_tuple) for value in values)


def _is_missing(value: object) -> bool:
    if value is None or _is_pandas_missing(value):
        return True
    if isinstance(value, Decimal) and value.is_nan():
        return True
    return isinstance(value, float) and math.isnan(value)


def _typed_key(value: object) -> tuple[type, object]:
    """Make category identity type-sensitive (for example, ``1`` differs from ``True``)."""
    return (type(value), value)


@dataclass(frozen=True, slots=True)
class PlotColumn:
    """A type-preserving immutable column aligned to ``PlotSource.row_positions``."""

    column_id: str
    label: object
    dtype: str
    values: tuple[object, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.column_id, str) or not self.column_id:
            raise ValueError("column_id must be a non-empty string")
        if not isinstance(self.dtype, str):
            raise TypeError("dtype must be a string")
        object.__setattr__(self, "label", _freeze_scalar(self.label, allow_tuple=True))
        object.__setattr__(self, "values", _freeze_tuple(self.values))


@dataclass(frozen=True, slots=True)
class PlotSequenceMap:
    """Immutable sequence alignment metadata for one target row."""

    target_row_position: int
    source_row_positions: tuple[int, ...] = ()
    group_value: object | None = None
    time_value: object | None = None
    order_value_ns: int | None = None

    def __post_init__(self) -> None:
        if isinstance(self.target_row_position, bool) or not isinstance(self.target_row_position, int):
            raise TypeError("target_row_position must be an integer")
        positions = tuple(
            int(item) if isinstance(item, np.integer) and not isinstance(item, np.bool_) else item
            for item in self.source_row_positions
        )
        if any(isinstance(item, bool) or not isinstance(item, int) for item in positions):
            raise TypeError("source_row_positions must contain integers")
        object.__setattr__(self, "source_row_positions", positions)
        object.__setattr__(self, "group_value", _freeze_scalar(self.group_value, allow_tuple=True))
        object.__setattr__(self, "time_value", _freeze_scalar(self.time_value, allow_tuple=True))
        if self.order_value_ns is not None and (
            isinstance(self.order_value_ns, bool) or not isinstance(self.order_value_ns, int)
        ):
            raise TypeError("order_value_ns must be an integer or None")


@dataclass(frozen=True, slots=True)
class PlotProvenance:
    """Optional SHA-256 values; callers compare only matching semantic domains."""

    source_sha256: str | None = None
    data_sha256: str | None = None
    split_sha256: str | None = None
    frozen_config_sha256: str | None = None
    batch_snapshot_sha256: str | None = None
    receipt_manifest_sha256: str | None = None
    receipt_file_sha256: str | None = None
    sequence_plan_sha256: str | None = None
    cache_payload_sha256: str | None = None

    def __post_init__(self) -> None:
        for name in _SHA_FIELDS:
            value = getattr(self, name)
            if value is not None and (
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdefABCDEF" for character in value)
            ):
                raise ValueError(f"{name} must be a 64-character SHA-256 hex digest or None")


@dataclass(frozen=True, slots=True)
class PlotSource:
    """One complete loaded table or one owned experiment partition."""

    schema_version: int
    source_kind: str
    source_id: str
    owner_kind: str
    owner_id: str
    partition: str
    row_positions: tuple[int, ...]
    columns: tuple[PlotColumn, ...] = ()
    outputs: tuple[PlotColumn, ...] = ()
    sequence_map: tuple[PlotSequenceMap, ...] = ()
    provenance: PlotProvenance = field(default_factory=PlotProvenance)
    partition_count: int = 0

    def __post_init__(self) -> None:
        if isinstance(self.schema_version, bool) or self.schema_version != 1:
            raise ValueError("unsupported PlotSource schema_version")
        if self.source_kind not in _SOURCE_KINDS:
            raise ValueError(f"unsupported source_kind: {self.source_kind!r}")
        if self.owner_kind not in _OWNER_KINDS:
            raise ValueError(f"unsupported owner_kind: {self.owner_kind!r}")
        if self.partition not in _PARTITIONS:
            raise ValueError(f"unsupported partition: {self.partition!r}")
        if (
            not isinstance(self.source_id, str)
            or not self.source_id
            or not isinstance(self.owner_id, str)
            or not self.owner_id
        ):
            raise ValueError("source_id and owner_id must be non-empty strings")
        if (self.source_kind, self.owner_kind) not in {
            ("loaded_eda", "loaded_dataset"),
            ("session_cache", "session"),
            ("batch_cache", "batch_job"),
        }:
            raise ValueError("source_kind and owner_kind do not describe a supported owner pair")
        if self.source_kind == "loaded_eda" and self.partition != "all":
            raise ValueError("loaded EDA sources must use partition='all'")
        if self.source_kind != "loaded_eda" and self.partition == "all":
            raise ValueError("experiment cache sources must name one partition")

        positions = tuple(
            int(item) if isinstance(item, np.integer) and not isinstance(item, np.bool_) else item
            for item in self.row_positions
        )
        if any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in positions):
            raise TypeError("row_positions must contain non-negative integers")
        if len(set(positions)) != len(positions):
            raise ValueError("row_positions must be unique source positions")
        if (
            isinstance(self.partition_count, bool)
            or not isinstance(self.partition_count, int)
            or self.partition_count != len(positions)
        ):
            raise ValueError("partition_count must equal the number of row_positions")
        object.__setattr__(self, "row_positions", positions)

        columns = tuple(self.columns)
        outputs = tuple(self.outputs)
        sequence_map = tuple(self.sequence_map)
        if any(not isinstance(column, PlotColumn) for column in columns + outputs):
            raise TypeError("columns and outputs must contain PlotColumn values")
        if any(len(column.values) != len(positions) for column in columns + outputs):
            raise ValueError("every PlotColumn.values tuple must align with row_positions")
        ids = [column.column_id for column in columns + outputs]
        if len(ids) != len(set(ids)):
            raise ValueError("PlotSource column_id values must be unique across columns and outputs")
        if any(not isinstance(item, PlotSequenceMap) for item in sequence_map):
            raise TypeError("sequence_map must contain PlotSequenceMap values")
        target_positions = [item.target_row_position for item in sequence_map]
        if len(target_positions) != len(set(target_positions)):
            raise ValueError("sequence_map target_row_position values must be unique")
        if not set(target_positions).issubset(positions):
            raise ValueError("sequence_map targets must be present in row_positions")
        if not isinstance(self.provenance, PlotProvenance):
            raise TypeError("provenance must be a PlotProvenance")
        object.__setattr__(self, "columns", columns)
        object.__setattr__(self, "outputs", outputs)
        object.__setattr__(self, "sequence_map", sequence_map)


class PlotKind(str, Enum):
    HISTOGRAM = "histogram"
    CATEGORY_COUNTS = "category_counts"
    SCATTER = "scatter"
    CORRELATION = "correlation"
    CLASSIFICATION_CONFUSION = "classification_confusion"
    REGRESSION_ACTUAL_PREDICTED = "regression_actual_predicted"
    REGRESSION_RESIDUAL = "regression_residual"
    CLUSTER_SCATTER = "cluster_scatter"
    HMM_STATE_POSTERIOR = "hmm_state_posterior"


_KIND_TITLES = {
    PlotKind.HISTOGRAM: "数值直方图",
    PlotKind.CATEGORY_COUNTS: "类别计数",
    PlotKind.SCATTER: "数值散点图",
    PlotKind.CORRELATION: "数值相关性",
    PlotKind.CLASSIFICATION_CONFUSION: "真实类别与预测类别",
    PlotKind.REGRESSION_ACTUAL_PREDICTED: "真实值与预测值",
    PlotKind.REGRESSION_RESIDUAL: "回归残差（真实值 − 预测值）",
    PlotKind.CLUSTER_SCATTER: "已有特征坐标上的聚类",
    PlotKind.HMM_STATE_POSTERIOR: "HMM 隐藏状态与后验概率",
}


@dataclass(frozen=True, slots=True)
class PlotSpec:
    """A small allowlisted chart request with explicit column identities."""

    kind: PlotKind
    column_ids: tuple[str, ...] = ()
    bins: int = 20
    top_n: int = 20
    title: str | None = None

    def __post_init__(self) -> None:
        try:
            kind = self.kind if isinstance(self.kind, PlotKind) else PlotKind(self.kind)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"unsupported plot kind: {self.kind!r}") from exc
        object.__setattr__(self, "kind", kind)
        column_ids = tuple(self.column_ids)
        if any(not isinstance(item, str) or not item for item in column_ids):
            raise TypeError("column_ids must contain non-empty strings")
        object.__setattr__(self, "column_ids", column_ids)
        if kind == PlotKind.HISTOGRAM and (
            isinstance(self.bins, bool) or not isinstance(self.bins, int) or not 1 <= self.bins <= 500
        ):
            raise ValueError("histogram bins must be between 1 and 500")
        if kind == PlotKind.CATEGORY_COUNTS and (
            isinstance(self.top_n, bool) or not isinstance(self.top_n, int) or not 1 <= self.top_n <= 500
        ):
            raise ValueError("category top_n must be between 1 and 500")
        if self.title is not None and not isinstance(self.title, str):
            raise TypeError("title must be a string or None")

    @property
    def display_title(self) -> str:
        return self.title or _KIND_TITLES[self.kind]

    @property
    def spec_id(self) -> str:
        suffix = ",".join(self.column_ids)
        return f"{self.kind.value}:{suffix}" if suffix else self.kind.value


@dataclass(frozen=True, slots=True)
class PlotSeries:
    """One immutable plotted series."""

    name: str
    x_values: tuple[object, ...]
    y_values: tuple[object, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.name, str):
            raise TypeError("series name must be a string")
        object.__setattr__(self, "x_values", _freeze_tuple(self.x_values, allow_tuple=True))
        object.__setattr__(self, "y_values", _freeze_tuple(self.y_values, allow_tuple=True))
        if len(self.x_values) != len(self.y_values):
            raise ValueError("series x_values and y_values must have equal lengths")


@dataclass(frozen=True, slots=True)
class CorrelationCount:
    left_column_id: str
    right_column_id: str
    effective_count: int
    excluded_count: int
    correlation: float | None


@dataclass(frozen=True, slots=True)
class PlotPayload:
    """Pure, immutable render-ready values and accounting for one chart."""

    spec: PlotSpec
    source_kind: str
    source_id: str
    partition: str
    partition_count: int
    effective_count: int
    excluded_count: int
    sampled_count: int = 0
    title: str = ""
    x_label: str = ""
    y_label: str = ""
    x_values: tuple[object, ...] = ()
    y_values: tuple[object, ...] = ()
    labels: tuple[object, ...] = ()
    matrix: tuple[tuple[float | int, ...], ...] = ()
    category_counts: tuple[tuple[object, int], ...] = ()
    other_count: int = 0
    missing_count: int = 0
    series: tuple[PlotSeries, ...] = ()
    correlation_counts: tuple[CorrelationCount, ...] = ()
    note: str = ""

    def __post_init__(self) -> None:
        if self.partition_count < 0 or self.effective_count < 0 or self.excluded_count < 0 or self.sampled_count < 0:
            raise ValueError("payload counts must be non-negative")
        object.__setattr__(self, "x_values", _freeze_tuple(self.x_values, allow_tuple=True))
        object.__setattr__(self, "y_values", _freeze_tuple(self.y_values, allow_tuple=True))
        object.__setattr__(self, "labels", _freeze_tuple(self.labels, allow_tuple=True))
        object.__setattr__(self, "matrix", tuple(tuple(_freeze_scalar(item) for item in row) for row in self.matrix))
        frozen_categories = []
        for label, count in self.category_counts:
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise ValueError("category counts must be non-negative integers")
            frozen_categories.append((_freeze_scalar(label, allow_tuple=True), count))
        object.__setattr__(self, "category_counts", tuple(frozen_categories))
        object.__setattr__(self, "series", tuple(self.series))
        object.__setattr__(self, "correlation_counts", tuple(self.correlation_counts))
        if len(self.x_values) != len(self.y_values) and self.spec.kind in {
            PlotKind.HISTOGRAM,
            PlotKind.SCATTER,
            PlotKind.REGRESSION_ACTUAL_PREDICTED,
            PlotKind.REGRESSION_RESIDUAL,
            PlotKind.CLUSTER_SCATTER,
        }:
            if self.spec.kind != PlotKind.HISTOGRAM or len(self.x_values) != len(self.y_values) + 1:
                raise ValueError("chart x_values and y_values have incompatible lengths")


def _find_column(source: PlotSource, column_id: str) -> PlotColumn:
    for column in source.columns + source.outputs:
        if column.column_id == column_id:
            return column
    raise ValueError(f"PlotSource does not contain column_id {column_id!r}")


def _number(value: object) -> float | None:
    if _is_missing(value) or isinstance(value, bool) or not isinstance(value, (Real, Decimal, Fraction)):
        return None
    try:
        result = float(value)
    except (OverflowError, TypeError, ValueError):
        return None
    return result if math.isfinite(result) else None


def _numeric_column(column: PlotColumn) -> bool:
    return any(_number(value) is not None for value in column.values)


def _semantic_ids(source: PlotSource, prefix: str) -> dict[str, PlotColumn]:
    return {
        column.column_id: column
        for column in source.outputs
        if column.column_id.startswith(prefix)
    }


def available_plot_specs(source: PlotSource) -> tuple[PlotSpec, ...]:
    """Return only chart specs whose required source columns are present."""
    specs: list[PlotSpec] = []
    numeric_columns = tuple(column for column in source.columns if _numeric_column(column))
    for column in source.columns:
        if _numeric_column(column):
            specs.append(PlotSpec(PlotKind.HISTOGRAM, (column.column_id,)))
        specs.append(PlotSpec(PlotKind.CATEGORY_COUNTS, (column.column_id,)))
    for left_index, left in enumerate(numeric_columns):
        for right in numeric_columns[left_index + 1 :]:
            specs.append(PlotSpec(PlotKind.SCATTER, (left.column_id, right.column_id)))
    if len(numeric_columns) >= 2:
        specs.append(PlotSpec(PlotKind.CORRELATION, tuple(item.column_id for item in numeric_columns)))

    output_ids = {column.column_id for column in source.outputs}
    if {"classification.y_true", "classification.y_pred"}.issubset(output_ids):
        specs.append(PlotSpec(PlotKind.CLASSIFICATION_CONFUSION, ("classification.y_true", "classification.y_pred")))
    if {"regression.y_true", "regression.y_pred"}.issubset(output_ids):
        specs.extend((
            PlotSpec(PlotKind.REGRESSION_ACTUAL_PREDICTED, ("regression.y_true", "regression.y_pred")),
            PlotSpec(PlotKind.REGRESSION_RESIDUAL, ("regression.y_true", "regression.y_pred")),
        ))
    if "clustering.label" in output_ids:
        for left_index, left in enumerate(numeric_columns):
            for right in numeric_columns[left_index + 1 :]:
                specs.append(PlotSpec(
                    PlotKind.CLUSTER_SCATTER,
                    (left.column_id, right.column_id, "clustering.label"),
                ))
    components = tuple(sorted(
        (column for column in source.outputs if column.column_id.startswith("dimensionality_reduction.component.")),
        key=lambda item: _numeric_suffix(item.column_id),
    ))
    for left_index, left in enumerate(components):
        for right in components[left_index + 1 :]:
            if _numeric_column(left) and _numeric_column(right):
                specs.append(PlotSpec(PlotKind.SCATTER, (left.column_id, right.column_id), title="已有降维坐标散点图"))
    posterior_columns = tuple(sorted(
        (column for column in source.outputs if column.column_id.startswith("hmm.posterior.")),
        key=lambda item: _numeric_suffix(item.column_id),
    ))
    if "hmm.hidden_state" in output_ids and posterior_columns:
        specs.append(PlotSpec(
            PlotKind.HMM_STATE_POSTERIOR,
            ("hmm.hidden_state",) + tuple(column.column_id for column in posterior_columns),
        ))
    return tuple(specs)


def _numeric_suffix(column_id: str) -> int:
    try:
        return int(column_id.rsplit(".", 1)[1])
    except (ValueError, IndexError):
        return 0


def _validate_spec(source: PlotSource, spec: PlotSpec) -> None:
    if not isinstance(source, PlotSource) or not isinstance(spec, PlotSpec):
        raise TypeError("build_plot_payload requires a PlotSource and a PlotSpec")
    if (spec.kind, spec.column_ids) not in {
        (available.kind, available.column_ids) for available in available_plot_specs(source)
    }:
        raise ValueError(f"plot spec {spec.spec_id!r} is not available for this source")


def _payload(
    source: PlotSource,
    spec: PlotSpec,
    *,
    effective: int,
    excluded: int,
    title: str | None = None,
    x_label: str = "",
    y_label: str = "",
    **fields: Any,
) -> PlotPayload:
    return PlotPayload(
        spec=spec,
        source_kind=source.source_kind,
        source_id=source.source_id,
        partition=source.partition,
        partition_count=source.partition_count,
        effective_count=effective,
        excluded_count=excluded,
        sampled_count=0,
        title=title or spec.display_title,
        x_label=x_label,
        y_label=y_label,
        **fields,
    )


def _histogram(source: PlotSource, spec: PlotSpec) -> PlotPayload:
    column = _find_column(source, spec.column_ids[0])
    values = [number for value in column.values if (number := _number(value)) is not None]
    counts, edges = np.histogram(np.asarray(values, dtype=np.float64), bins=spec.bins)
    return _payload(
        source,
        spec,
        effective=len(values),
        excluded=source.partition_count - len(values),
        title=f"{spec.display_title}：{column.label}",
        x_label=str(column.label),
        y_label="计数",
        x_values=tuple(float(item) for item in edges),
        y_values=tuple(int(item) for item in counts),
    )


def _category_counts(source: PlotSource, spec: PlotSpec) -> PlotPayload:
    column = _find_column(source, spec.column_ids[0])
    counts: dict[tuple[type, object], int] = {}
    labels: dict[tuple[type, object], object] = {}
    order: list[tuple[type, object]] = []
    missing_count = 0
    for value in column.values:
        if _is_missing(value):
            missing_count += 1
            continue
        key = _typed_key(value)
        if key not in counts:
            counts[key] = 0
            labels[key] = value
            order.append(key)
        counts[key] += 1
    first_seen = {key: index for index, key in enumerate(order)}
    ordered = sorted(order, key=lambda key: (-counts[key], first_seen[key]))
    shown_keys = ordered[: spec.top_n]
    other_count = sum(counts[key] for key in ordered[spec.top_n :])
    categories = tuple((labels[key], counts[key]) for key in shown_keys)
    effective = sum(count for _label, count in categories) + other_count
    return _payload(
        source,
        spec,
        effective=effective,
        excluded=0,
        title=f"{spec.display_title}：{column.label}",
        x_label=str(column.label),
        y_label="计数",
        category_counts=categories,
        other_count=other_count,
        missing_count=missing_count,
    )


def _pair_values(left: PlotColumn, right: PlotColumn) -> tuple[tuple[float, ...], tuple[float, ...]]:
    x_values: list[float] = []
    y_values: list[float] = []
    for left_value, right_value in zip(left.values, right.values):
        x = _number(left_value)
        y = _number(right_value)
        if x is not None and y is not None:
            x_values.append(x)
            y_values.append(y)
    return tuple(x_values), tuple(y_values)


def _scatter(source: PlotSource, spec: PlotSpec) -> PlotPayload:
    left, right = (_find_column(source, item) for item in spec.column_ids)
    x_values, y_values = _pair_values(left, right)
    return _payload(
        source,
        spec,
        effective=len(x_values),
        excluded=source.partition_count - len(x_values),
        title=f"{spec.display_title}：{left.label} × {right.label}",
        x_label=str(left.label),
        y_label=str(right.label),
        x_values=x_values,
        y_values=y_values,
    )


def _correlation(source: PlotSource, spec: PlotSpec) -> PlotPayload:
    columns = tuple(_find_column(source, item) for item in spec.column_ids)
    matrix: list[tuple[float, ...]] = []
    pair_counts: list[CorrelationCount] = []
    for left in columns:
        row: list[float] = []
        for right in columns:
            x_values, y_values = _pair_values(left, right)
            n = len(x_values)
            correlation: float | None = None
            if n >= 2:
                mean_x = math.fsum(x_values) / n
                mean_y = math.fsum(y_values) / n
                centered_x = tuple(value - mean_x for value in x_values)
                centered_y = tuple(value - mean_y for value in y_values)
                sum_x = math.fsum(value * value for value in centered_x)
                sum_y = math.fsum(value * value for value in centered_y)
                if sum_x > 0.0 and sum_y > 0.0:
                    correlation = math.fsum(a * b for a, b in zip(centered_x, centered_y)) / math.sqrt(sum_x * sum_y)
                    correlation = min(1.0, max(-1.0, correlation))
            row.append(float("nan") if correlation is None else correlation)
            if left.column_id != right.column_id or left is right:
                pair_counts.append(CorrelationCount(
                    left_column_id=left.column_id,
                    right_column_id=right.column_id,
                    effective_count=n,
                    excluded_count=source.partition_count - n,
                    correlation=correlation,
                ))
        matrix.append(tuple(row))
    complete_rows = sum(
        all(_number(column.values[index]) is not None for column in columns)
        for index in range(source.partition_count)
    )
    return _payload(
        source,
        spec,
        effective=complete_rows,
        excluded=source.partition_count - complete_rows,
        x_label="列",
        y_label="列",
        labels=tuple(column.label for column in columns),
        matrix=tuple(matrix),
        correlation_counts=tuple(pair_counts),
        note="每个列对均使用各自的有限配对行数；常数列或少于两对时相关系数无定义。",
    )


def _classification_confusion(source: PlotSource, spec: PlotSpec) -> PlotPayload:
    actual, predicted = (_find_column(source, item) for item in spec.column_ids)
    labels: list[object] = []
    seen: set[tuple[type, object]] = set()
    pairs: list[tuple[object, object]] = []
    for true_value, predicted_value in zip(actual.values, predicted.values):
        if _is_missing(true_value) or _is_missing(predicted_value):
            continue
        pairs.append((true_value, predicted_value))
        for value in (true_value, predicted_value):
            key = _typed_key(value)
            if key not in seen:
                seen.add(key)
                labels.append(value)
    positions = {_typed_key(value): index for index, value in enumerate(labels)}
    matrix = [[0 for _ in labels] for _ in labels]
    for true_value, predicted_value in pairs:
        matrix[positions[_typed_key(true_value)]][positions[_typed_key(predicted_value)]] += 1
    return _payload(
        source,
        spec,
        effective=len(pairs),
        excluded=source.partition_count - len(pairs),
        x_label="预测类别",
        y_label="真实类别",
        labels=tuple(labels),
        matrix=tuple(tuple(row) for row in matrix),
    )


def _regression(source: PlotSource, spec: PlotSpec) -> PlotPayload:
    actual, predicted = (_find_column(source, item) for item in spec.column_ids)
    x_values: list[float] = []
    y_values: list[float] = []
    for actual_value, predicted_value in zip(actual.values, predicted.values):
        true = _number(actual_value)
        pred = _number(predicted_value)
        if true is None or pred is None:
            continue
        if spec.kind == PlotKind.REGRESSION_ACTUAL_PREDICTED:
            x_values.append(true)
            y_values.append(pred)
        else:
            x_values.append(true)
            y_values.append(true - pred)
    return _payload(
        source,
        spec,
        effective=len(x_values),
        excluded=source.partition_count - len(x_values),
        x_label="真实值" if spec.kind == PlotKind.REGRESSION_RESIDUAL else "真实值",
        y_label="残差（真实值 − 预测值）" if spec.kind == PlotKind.REGRESSION_RESIDUAL else "预测值",
        x_values=tuple(x_values),
        y_values=tuple(y_values),
        note="残差固定为真实值减预测值（y_true - y_pred）。" if spec.kind == PlotKind.REGRESSION_RESIDUAL else "",
    )


def _cluster_scatter(source: PlotSource, spec: PlotSpec) -> PlotPayload:
    x_column, y_column, cluster_column = (_find_column(source, item) for item in spec.column_ids)
    grouped: dict[tuple[type, object], tuple[object, list[float], list[float]]] = {}
    order: list[tuple[type, object]] = []
    excluded = 0
    for x_value, y_value, cluster in zip(x_column.values, y_column.values, cluster_column.values):
        x = _number(x_value)
        y = _number(y_value)
        if x is None or y is None or _is_missing(cluster):
            excluded += 1
            continue
        key = _typed_key(cluster)
        if key not in grouped:
            grouped[key] = (cluster, [], [])
            order.append(key)
        grouped[key][1].append(x)
        grouped[key][2].append(y)
    series = tuple(PlotSeries(
        name=_display_value(grouped[key][0]),
        x_values=tuple(grouped[key][1]),
        y_values=tuple(grouped[key][2]),
    ) for key in order)
    effective = source.partition_count - excluded
    return _payload(
        source,
        spec,
        effective=effective,
        excluded=excluded,
        x_label=str(x_column.label),
        y_label=str(y_column.label),
        series=series,
    )


def _hmm_payload(source: PlotSource, spec: PlotSpec) -> PlotPayload:
    state_column = _find_column(source, "hmm.hidden_state")
    posterior_columns = tuple(_find_column(source, item) for item in spec.column_ids[1:])
    sequence_by_position = {item.target_row_position: item for item in source.sequence_map}
    positions = list(range(source.partition_count))
    positions.sort(key=lambda index: _sequence_sort_key(sequence_by_position.get(source.row_positions[index]), index))
    sequences = tuple(sequence_by_position.get(source.row_positions[index]) for index in positions)
    utc_order_axis = bool(sequences) and all(
        sequence is not None
        and sequence.time_value is None
        and sequence.order_value_ns is not None
        for sequence in sequences
    )
    x_values: list[object] = []
    states: list[object] = []
    groups: list[object] = []
    excluded = 0
    for index in positions:
        position = source.row_positions[index]
        sequence = sequence_by_position.get(position)
        time_value = sequence.time_value if sequence is not None else None
        order_value = sequence.order_value_ns if sequence is not None else None
        x_values.append(time_value if time_value is not None else (order_value if order_value is not None else index))
        groups.append(sequence.group_value if sequence is not None else None)
        states.append(state_column.values[index])
        row_valid = not _is_missing(state_column.values[index])
        for column in posterior_columns:
            value = _number(column.values[index])
            if value is None:
                row_valid = False
                break
        if not row_valid:
            excluded += 1
    # Invalid posterior rows are marked as NaN so every posterior remains aligned to the sequence axis.
    series = [PlotSeries("隐藏状态", tuple(x_values), tuple(states))]
    for column in posterior_columns:
        values: list[float] = []
        for index in positions:
            number = _number(column.values[index])
            values.append(float("nan") if number is None else number)
        series.append(PlotSeries(column.column_id.rsplit(".", 1)[-1], tuple(x_values), tuple(values)))
    return _payload(
        source,
        spec,
        effective=source.partition_count - excluded,
        excluded=excluded,
        y_label="隐藏状态与后验概率",
        x_values=tuple(x_values),
        y_values=tuple(states),
        labels=tuple(groups),
        series=tuple(series),
        x_label="时间（UTC）" if utc_order_axis else (
            "时间值" if any(sequence is not None and sequence.time_value is not None for sequence in sequences)
            else "行 / 序列顺序"
        ),
        note=(
            "隐藏状态是模型潜在状态，不是真实类别。时间坐标由缓存的 UTC 纳秒顺序值规范化显示。"
            if utc_order_axis
            else "隐藏状态是模型潜在状态，不是真实类别。"
        ),
    )


def _sequence_sort_key(sequence: PlotSequenceMap | None, fallback: int) -> tuple[str, int, int]:
    if sequence is None:
        return ("", fallback, fallback)
    group_key = repr(_typed_key(sequence.group_value)) if sequence.group_value is not None else ""
    if sequence.order_value_ns is not None:
        order = sequence.order_value_ns
    elif isinstance(sequence.time_value, datetime):
        order = int(sequence.time_value.timestamp() * 1_000_000_000)
    elif isinstance(sequence.time_value, (int, float)) and not isinstance(sequence.time_value, bool):
        order = int(sequence.time_value)
    else:
        order = fallback
    return (group_key, order, fallback)


def _display_value(value: object) -> str:
    return f"{value!r} [{type(value).__name__}]"


def build_plot_payload(source: PlotSource, spec: PlotSpec) -> PlotPayload:
    """Build a render-ready chart payload without file, GUI, model, or Matplotlib access."""
    _validate_spec(source, spec)
    builders = {
        PlotKind.HISTOGRAM: _histogram,
        PlotKind.CATEGORY_COUNTS: _category_counts,
        PlotKind.SCATTER: _scatter,
        PlotKind.CORRELATION: _correlation,
        PlotKind.CLASSIFICATION_CONFUSION: _classification_confusion,
        PlotKind.REGRESSION_ACTUAL_PREDICTED: _regression,
        PlotKind.REGRESSION_RESIDUAL: _regression,
        PlotKind.CLUSTER_SCATTER: _cluster_scatter,
        PlotKind.HMM_STATE_POSTERIOR: _hmm_payload,
    }
    return builders[spec.kind](source, spec)


def _count_caption(payload: PlotPayload) -> str:
    caption = (
        f"来源 {payload.source_id} · 分区 {payload.partition} · 分区 {payload.partition_count} 行 · "
        f"有效 {payload.effective_count} · 排除 {payload.excluded_count} · 抽样 {payload.sampled_count}"
    )
    if payload.missing_count:
        caption += f" · 缺失类别 {payload.missing_count}"
    if payload.other_count:
        caption += f" · 其他类别 {payload.other_count}"
    return caption


def _label_text(value: object) -> str:
    return _display_value(value)


def render_plot_payload(payload: PlotPayload, figure: Any | None = None) -> Any:
    """Render one payload into the explicitly supplied Figure, without pyplot state."""
    from matplotlib.figure import Figure
    from matplotlib import font_manager
    from matplotlib.dates import AutoDateLocator, ConciseDateFormatter
    from matplotlib.text import Text

    if not isinstance(payload, PlotPayload):
        raise TypeError("payload must be a PlotPayload")
    if figure is None:
        figure = Figure(figsize=(7.2, 4.6))
    if not isinstance(figure, Figure):
        raise TypeError("figure must be a matplotlib.figure.Figure")
    figure.clear()
    kind = payload.spec.kind
    if kind == PlotKind.HMM_STATE_POSTERIOR:
        utc_axis = payload.x_label == "时间（UTC）"
        if utc_axis:
            x_values = tuple(_utc_datetime_from_ns(value) for value in payload.x_values)
        else:
            x_values = payload.x_values
        state_axis, posterior_axis = figure.subplots(2, 1, sharex=True)
        state_series = payload.series[0] if payload.series else None
        if state_series is not None:
            state_axis.step(x_values, state_series.y_values, where="mid", label="隐藏状态")
        state_axis.set_ylabel("隐藏状态")
        state_axis.legend(loc="best")
        for series in payload.series[1:]:
            posterior_axis.plot(x_values, series.y_values, label=f"后验 {series.name}")
        posterior_axis.set_ylabel("后验概率")
        posterior_axis.set_xlabel(payload.x_label)
        posterior_axis.set_ylim(-0.02, 1.02)
        if utc_axis:
            locator = AutoDateLocator(tz=timezone.utc)
            posterior_axis.xaxis.set_major_locator(locator)
            posterior_axis.xaxis.set_major_formatter(ConciseDateFormatter(locator, tz=timezone.utc))
        if len(payload.series) > 1:
            posterior_axis.legend(loc="best", ncol=2)
        figure.suptitle(payload.title or payload.spec.display_title)
    else:
        axis = figure.subplots()
        if kind == PlotKind.HISTOGRAM:
            axis.stairs(payload.y_values, payload.x_values, fill=True, alpha=0.7)
        elif kind == PlotKind.CATEGORY_COUNTS:
            labels = [_label_text(label) for label, _count in payload.category_counts]
            counts = [count for _label, count in payload.category_counts]
            if payload.other_count:
                labels.append("其他")
                counts.append(payload.other_count)
            if payload.missing_count:
                labels.append("缺失")
                counts.append(payload.missing_count)
            axis.bar(range(len(labels)), counts)
            axis.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
        elif kind == PlotKind.SCATTER:
            axis.scatter(payload.x_values, payload.y_values, alpha=0.75)
        elif kind == PlotKind.CORRELATION:
            data = np.asarray(payload.matrix, dtype=np.float64)
            masked = np.ma.masked_invalid(data)
            image = axis.imshow(masked, vmin=-1.0, vmax=1.0, cmap="coolwarm")
            labels = [_label_text(label) for label in payload.labels]
            axis.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
            axis.set_yticks(range(len(labels)), labels)
            figure.colorbar(image, ax=axis, label="Pearson r")
            for row_index, row in enumerate(payload.matrix):
                for column_index, value in enumerate(row):
                    text = "n/a" if not math.isfinite(float(value)) else f"{value:.2f}"
                    axis.text(column_index, row_index, text, ha="center", va="center")
        elif kind == PlotKind.CLASSIFICATION_CONFUSION:
            data = np.asarray(payload.matrix, dtype=np.int64)
            image = axis.imshow(data, cmap="Blues")
            labels = [_label_text(label) for label in payload.labels]
            axis.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
            axis.set_yticks(range(len(labels)), labels)
            figure.colorbar(image, ax=axis, label="计数")
            for row_index, row in enumerate(payload.matrix):
                for column_index, value in enumerate(row):
                    axis.text(column_index, row_index, str(value), ha="center", va="center")
        elif kind in {PlotKind.REGRESSION_ACTUAL_PREDICTED, PlotKind.REGRESSION_RESIDUAL}:
            axis.scatter(payload.x_values, payload.y_values, alpha=0.75)
            if kind == PlotKind.REGRESSION_ACTUAL_PREDICTED and payload.x_values:
                lower = min(min(payload.x_values), min(payload.y_values))
                upper = max(max(payload.x_values), max(payload.y_values))
                axis.plot((lower, upper), (lower, upper), linestyle="--", color="black", linewidth=1)
            elif kind == PlotKind.REGRESSION_RESIDUAL:
                axis.axhline(0.0, color="black", linestyle="--", linewidth=1)
        elif kind == PlotKind.CLUSTER_SCATTER:
            for series in payload.series:
                axis.scatter(series.x_values, series.y_values, alpha=0.8, label=series.name)
            if payload.series:
                axis.legend(title="已有簇标签")
        axis.set_title(payload.title)
        axis.set_xlabel(payload.x_label)
        axis.set_ylabel(payload.y_label)
    figure.text(0.01, 0.035, _count_caption(payload), fontsize=8, va="center")
    if payload.note:
        figure.text(0.01, 0.008, payload.note, fontsize=8, va="center")
    for family in ("Microsoft YaHei", "SimHei", "Noto Sans CJK SC"):
        try:
            font_manager.findfont(font_manager.FontProperties(family=family), fallback_to_default=False)
        except ValueError:
            continue
        for text_artist in figure.findobj(match=lambda artist: isinstance(artist, Text)):
            text_artist.set_fontfamily(family)
        break
    figure.tight_layout(rect=(0, 0.09, 1, 0.95))
    return figure


def _utc_datetime_from_ns(value: object) -> datetime:
    """Convert a normalized UTC epoch-nanosecond value for display only."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError("UTC sequence coordinates must be integer nanoseconds")
    epoch = datetime(1970, 1, 1, tzinfo=timezone.utc)
    return epoch + timedelta(microseconds=value // 1_000)


__all__ = [
    "CorrelationCount",
    "PlotColumn",
    "PlotKind",
    "PlotPayload",
    "PlotProvenance",
    "PlotSequenceMap",
    "PlotSeries",
    "PlotSource",
    "PlotSpec",
    "available_plot_specs",
    "build_plot_payload",
    "render_plot_payload",
]
