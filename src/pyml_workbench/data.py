"""Safe table loading, preview, and explicit feature/target selection."""
from __future__ import annotations

from dataclasses import dataclass
import math
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


def _preserve_identity_csv_token(value: str) -> Any:
    """Keep identity tokens textual while retaining blank cells as missing."""
    return pd.NA if value == "" else value


def _restore_csv_integer_negative_zero(
    frame: pd.DataFrame,
    path: Path,
    *,
    encoding: str,
    nrows: int | None,
) -> pd.DataFrame:
    """Restore negative-zero tokens lost by pandas integer inference.

    Only integer columns that already contain zero are scanned. A bounded,
    pandas-parsed read of those columns preserves CSV quoting and row alignment
    without materializing a second string copy of the whole table.
    """
    if frame.empty:
        return frame

    frame_dtypes = frame.dtypes
    candidate_columns = [
        index
        for index in range(frame.shape[1])
        if pd.api.types.is_integer_dtype(frame_dtypes.iloc[index])
        and bool(frame.iloc[:, index].eq(0).fillna(False).any())
    ]
    if not candidate_columns:
        return frame

    positions: list[tuple[int, int]] = []
    rows_read = 0
    with pd.read_csv(
        path,
        encoding=encoding,
        usecols=candidate_columns,
        dtype=str,
        keep_default_na=False,
        na_filter=False,
        nrows=nrows,
        chunksize=8192,
    ) as chunks:
        for chunk in chunks:
            if chunk.shape[1] != len(candidate_columns):
                raise DatasetError("CSV negative-zero scan selected an unexpected column count")
            for token_column, frame_column in enumerate(candidate_columns):
                for row_offset, raw_token in enumerate(chunk.iloc[:, token_column]):
                    token = raw_token.strip()
                    if len(token) > 1 and token[0] == "-" and all(
                        character == "0" for character in token[1:]
                    ):
                        row_index = rows_read + row_offset
                        if row_index >= len(frame) or frame.iat[row_index, frame_column] != 0:
                            raise DatasetError("CSV negative-zero scan lost row alignment")
                        positions.append((row_index, frame_column))
            rows_read += len(chunk)

    if rows_read != len(frame):
        raise DatasetError("CSV negative-zero scan row count differs from the parsed table")
    if not positions:
        return frame

    affected_columns = {column_index for _, column_index in positions}
    for column_index in affected_columns:
        column = frame.iloc[:, column_index]
        safe_integers = all(abs(int(value)) <= 2**53 for value in column.dropna())
        if safe_integers:
            replacement = column.astype("Float64" if column.isna().any() else "float64")
        else:
            replacement = column.astype(object)
        frame.isetitem(column_index, replacement)

    for row_index, column_index in positions:
        frame.iat[row_index, column_index] = -0.0
    return frame


def _restore_xlsx_negative_zero(frame: pd.DataFrame, worksheet: Any) -> pd.DataFrame:
    """Restore numeric ``-0.0`` values normalized to integer zero by pandas.

    The openpyxl-backed pandas reader converts integral numeric cells to ints,
    which removes the sign bit from negative zero. The workbook still exposes
    the original float values, so restore those cells after parsing.
    """
    if frame.empty:
        return frame

    positions: list[tuple[int, int]] = []
    for row_index, row in enumerate(
        worksheet.iter_rows(min_row=2, max_row=frame.shape[0] + 1)
    ):
        if row_index >= frame.shape[0]:
            break
        for column_index, cell in enumerate(row[: len(frame.columns)]):
            value = cell.value
            if isinstance(value, float) and value == 0.0 and math.copysign(1.0, value) < 0:
                positions.append((row_index, column_index))

    if not positions:
        return frame

    affected_columns = {column_index for _, column_index in positions}
    for column_index in affected_columns:
        column = frame.iloc[:, column_index]
        if pd.api.types.is_integer_dtype(column.dtype):
            safe_integers = all(abs(int(value)) <= 2**53 for value in column.dropna())
            replacement = column.astype("float64") if safe_integers else column.astype(object)
            frame.isetitem(column_index, replacement)
    for row_index, column_index in positions:
        frame.iat[row_index, column_index] = -0.0
    return frame


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
            columns = pd.read_csv(path, encoding=encoding, nrows=0).columns
            identity_columns = [
                column
                for column in columns
                if column == "sample_id" or str(column).startswith("meta__")
            ]
            frame = pd.read_csv(
                path,
                encoding=encoding,
                nrows=nrows,
                converters={
                    column: _preserve_identity_csv_token
                    for column in identity_columns
                },
                float_precision="round_trip",
            )
            frame = _restore_csv_integer_negative_zero(
                frame,
                path,
                encoding=encoding,
                nrows=nrows,
            )
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
                if suffix == ".xlsx":
                    frame = _restore_xlsx_negative_zero(frame, workbook.book[selected])
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
