import hashlib
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from pyml_workbench.data import (
    DatasetError,
    MissingTargetError,
    load_dataset,
    preview_dataset,
    select_features_target,
)
from pyml_workbench.preprocessing import build_preprocessor


class DatasetLoadingTests(unittest.TestCase):
    def test_csv_preview_selection_and_missing_target_guard_preserve_source(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "data.csv"
            original = pd.DataFrame({"value": [1.0, None, 3.0], "kind": ["a", "b", "a"], "target": [0, 1, 0]})
            original.to_csv(path, index=False)
            before = hashlib.sha256(path.read_bytes()).hexdigest()

            preview = preview_dataset(path, rows=2)
            loaded = load_dataset(path)
            features, target = select_features_target(
                loaded,
                task="classification",
                target_column="target",
                feature_columns=["value", "kind"],
            )
            self.assertEqual(len(preview.sample), 2)
            self.assertEqual(list(features.columns), ["value", "kind"])
            self.assertEqual(target.tolist(), [0, 1, 0])
            self.assertEqual(before, hashlib.sha256(path.read_bytes()).hexdigest())

            original.loc[1, "target"] = None
            original.to_csv(path, index=False)
            after_missing_target_write = hashlib.sha256(path.read_bytes()).hexdigest()
            with self.assertRaises(MissingTargetError):
                select_features_target(
                    load_dataset(path),
                    task="classification",
                    target_column="target",
                    feature_columns=["value", "kind"],
                )
            self.assertEqual(after_missing_target_write, hashlib.sha256(path.read_bytes()).hexdigest())

    def test_xlsx_sheet_selection_and_preview(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "book.xlsx"
            with pd.ExcelWriter(path, engine="openpyxl") as writer:
                pd.DataFrame({"value": [1, 2]}).to_excel(writer, sheet_name="first", index=False)
                pd.DataFrame({"other": [3, 4]}).to_excel(writer, sheet_name="second", index=False)
            preview = preview_dataset(path, sheet_name="second", rows=1)
            self.assertEqual(preview.sheet_names, ("first", "second"))
            self.assertEqual(preview.selected_sheet, "second")
            self.assertEqual(preview.columns, ("other",))
            self.assertEqual(len(preview.sample), 1)
            with self.assertRaisesRegex(DatasetError, "not found"):
                load_dataset(path, sheet_name="missing")

    def test_preprocessor_fits_numeric_and_categorical_transformers(self):
        frame = pd.DataFrame({"numeric": [1.0, None, 3.0], "category": ["a", "b", None]})
        transformer, details = build_preprocessor(frame, "C01")
        transformed = transformer.fit_transform(frame)
        self.assertEqual(transformed.shape, (3, 3))
        self.assertEqual(details["numeric_columns"], ["numeric"])
        self.assertEqual(details["categorical_columns"], ["category"])


if __name__ == "__main__":
    unittest.main()
