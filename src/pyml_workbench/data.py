"""Safe table loading, preview, and explicit feature/target selection."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .config import DatasetConfig


class DatasetError(ValueError):
    """Raised when a tabular file cannot be read or selected safely."""


class MissingTargetError(DatasetError):
    """Raised when a supervised target contains one or more missing values."""


@dataclass(frozen=True)
class DatasetPreview:
    source_path: str
    selected_sheet: str | None
    sheet_names: tuple[str, ...]
    columns: tuple[str, ...]
    dtypes: dict[str, str]
    missing_counts: dict[str, int]
    sample: pd.DataFrame


@dataclass(frozen=True)
class LoadedDataset:
    source_path: str
    selected_sheet: str | None
    sheet_names: tuple[str, ...]
    frame: pd.DataFrame

    @property
    def columns(self) -> tuple[str, ...]:
        return tuple(str(column) for column in self.frame.columns)

    def preview(self, rows: int = 20) -> pd.DataFrame:
        if rows < 0:
            raise ValueError("rows must be non-negative")
        return self.frame.head(rows).copy(deep=True)

    def describe(self, rows: int = 20) -> DatasetPreview:
        sample = self.preview(rows)
        return DatasetPreview(
            source_path=self.source_path,
            selected_sheet=self.selected_sheet,
            sheet_names=self.sheet_names,
            columns=self.columns,
            dtypes={str(name): str(dtype) for name, dtype in self.frame.dtypes.items()},
            missing_counts={str(name): int(value) for name, value in self.frame.isna().sum().items()},
            sample=sample,
        )


def _resolve_source(source: str | Path | DatasetConfig, sheet_name: str | int | None):
    if isinstance(source, DatasetConfig):
        if sheet_name is not None and source.sheet_name not in (None, sheet_name):
            raise DatasetError("sheet_name conflicts with the supplied DatasetConfig")
        return Path(source.source_path), source.sheet_name if sheet_name is None else sheet_name
    return Path(source), sheet_name


def _read_table(
    source: str | Path | DatasetConfig,
    *,
    sheet_name: str | int | None = None,
    nrows: int | None = None,
    encoding: str = "utf-8-sig",
) -> LoadedDataset:
    path, requested_sheet = _resolve_source(source, sheet_name)
    if not path.exists() or not path.is_file():
        raise DatasetError(f"Dataset file does not exist or is not a file: {path}")
    suffix = path.suffix.lower()
    if suffix not in {".csv", ".xlsx", ".xls"}:
        raise DatasetError(f"Unsupported file type {suffix!r}; use CSV, XLSX, or XLS")
    try:
        if suffix == ".csv":
            if requested_sheet is not None:
                raise DatasetError("sheet_name is only valid for Excel workbooks")
            frame = pd.read_csv(path, encoding=encoding, nrows=nrows)
            sheets: tuple[str, ...] = ()
            selected = None
        else:
            engine = "openpyxl" if suffix == ".xlsx" else "xlrd"
            with pd.ExcelFile(path, engine=engine) as workbook:
                sheets = tuple(str(item) for item in workbook.sheet_names)
                if not sheets:
                    raise DatasetError(f"Excel workbook has no worksheets: {path}")
                if isinstance(requested_sheet, int):
                    if requested_sheet < 0 or requested_sheet >= len(sheets):
                        raise DatasetError(
                            f"Worksheet index {requested_sheet} is out of range; workbook has {len(sheets)} sheets"
                        )
                    selected = sheets[requested_sheet]
                elif requested_sheet is None:
                    selected = sheets[0]
                else:
                    selected = str(requested_sheet)
                    if selected not in sheets:
                        raise DatasetError(
                            f"Worksheet {selected!r} was not found; available sheets: {', '.join(sheets)}"
                        )
                frame = workbook.parse(sheet_name=selected, nrows=nrows)
    except DatasetError:
        raise
    except Exception as exc:
        raise DatasetError(f"Could not read {path.name}: {exc}") from exc
    if frame.empty or len(frame.columns) == 0:
        raise DatasetError(f"Dataset has no data rows or columns: {path}")
    if frame.columns.has_duplicates:
        duplicates = frame.columns[frame.columns.duplicated()].tolist()
        raise DatasetError(f"Dataset has duplicate column names: {duplicates}")
    frame = frame.copy(deep=True)
    return LoadedDataset(
        source_path=str(path.resolve()),
        selected_sheet=selected,
        sheet_names=sheets,
        frame=frame,
    )


def load_dataset(
    source: str | Path | DatasetConfig,
    *,
    sheet_name: str | int | None = None,
    encoding: str = "utf-8-sig",
) -> LoadedDataset:
    """Read a CSV/XLSX/XLS table without modifying the original file."""
    return _read_table(source, sheet_name=sheet_name, encoding=encoding)


def preview_dataset(
    source: str | Path | DatasetConfig,
    *,
    sheet_name: str | int | None = None,
    rows: int = 20,
    encoding: str = "utf-8-sig",
) -> DatasetPreview:
    """Read only a small sample and report workbook sheets and column metadata."""
    if rows < 1:
        raise ValueError("rows must be at least 1")
    loaded = _read_table(source, sheet_name=sheet_name, nrows=rows, encoding=encoding)
    return loaded.describe(rows)


def select_features_target(
    dataset: LoadedDataset,
    *,
    task: str,
    target_column: str | None,
    feature_columns: tuple[str, ...] | list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series | None]:
    """Return copies of selected columns and reject incomplete supervised targets."""
    supervised = task in {"classification", "regression"}
    frame = dataset.frame
    if supervised and not target_column:
        raise DatasetError(f"A target column is required for supervised task {task!r}")
    if target_column is not None and target_column not in frame.columns:
        raise DatasetError(f"Target column {target_column!r} is not present in the dataset")
    if feature_columns is None:
        selected = [column for column in frame.columns if column != target_column]
    else:
        selected = list(feature_columns)
    if not selected:
        raise DatasetError("At least one feature column must be selected")
    if len(set(selected)) != len(selected):
        raise DatasetError("Feature columns must be unique")
    missing = [column for column in selected if column not in frame.columns]
    if missing:
        raise DatasetError(f"Feature columns are not present in the dataset: {missing}")
    if target_column is not None and target_column in selected:
        raise DatasetError("The target column cannot also be a feature")
    features = frame.loc[:, selected].copy(deep=True)
    all_missing = [str(column) for column in features.columns if features[column].isna().all()]
    if all_missing:
        raise DatasetError(f"Selected feature columns are entirely missing: {all_missing}")
    if not supervised:
        return features, None
    target = frame.loc[:, target_column].copy(deep=True)
    missing_count = int(target.isna().sum())
    if missing_count:
        raise MissingTargetError(
            f"Target column {target_column!r} contains {missing_count} missing value(s); "
            "fill or correct the source data before training. The source file was not changed."
        )
    if task == "classification" and target.nunique(dropna=False) < 2:
        raise DatasetError("Classification requires at least two target classes")
    return features, target
