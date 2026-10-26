"""Regression checks for CSV integer inference losing the sign of zero."""
from __future__ import annotations

import math
from pathlib import Path
import struct
import tempfile
import unittest

from pyml_workbench.data import load_dataset, preview_dataset


def _bits(value: float) -> int:
    return struct.unpack(">Q", struct.pack(">d", float(value)))[0]


class CsvNegativeZeroReaderTest(unittest.TestCase):
    def test_single_integer_token_negative_zero_keeps_its_sign(self) -> None:
        with tempfile.TemporaryDirectory(prefix="pyml-csv-negative-zero-") as temporary:
            csv_path = Path(temporary) / "single.csv"
            csv_path.write_text("feature__zero\n-0\n", encoding="utf-8")

            frame = load_dataset(csv_path).frame

        self.assertEqual(frame.shape, (1, 1))
        self.assertEqual(frame["feature__zero"].dtype.kind, "f")
        self.assertEqual(_bits(frame.loc[0, "feature__zero"]), _bits(-0.0))

    def test_chunked_lexical_scan_keeps_alignment_and_exact_integers(self) -> None:
        csv_text = (
            "sample_id,label,feature__negative_zero,feature__ordinary_integer,"
            "feature__large_integer,feature__missing_float,feature__precise\n"
            '001,"control\nparticipant",-0,7,-0,-0.0,0.10000000000000001\n'
            "002,case,2,8,9007199254740993,,1.2345678901234567\n"
            "\n"
            "003,case,-0,9,9007199254740994,2.5,3.141592653589793\n"
        )
        with tempfile.TemporaryDirectory(prefix="pyml-csv-negative-zero-") as temporary:
            csv_path = Path(temporary) / "quoted-and-large.csv"
            csv_path.write_bytes(csv_text.encode("utf-8"))

            frame = load_dataset(csv_path).frame
            preview = preview_dataset(csv_path, rows=1).sample

        self.assertEqual(frame.shape, (3, 7))
        self.assertEqual(frame["sample_id"].tolist(), ["001", "002", "003"])
        self.assertEqual(frame.loc[0, "label"], "control\nparticipant")
        self.assertEqual(_bits(frame.loc[0, "feature__negative_zero"]), _bits(-0.0))
        self.assertEqual(_bits(frame.loc[1, "feature__negative_zero"]), _bits(2.0))
        self.assertEqual(_bits(frame.loc[2, "feature__negative_zero"]), _bits(-0.0))

        self.assertIn(frame["feature__ordinary_integer"].dtype.kind, ("i", "u"))
        self.assertEqual(frame["feature__ordinary_integer"].tolist(), [7, 8, 9])
        self.assertEqual(frame["feature__large_integer"].dtype.kind, "O")
        self.assertEqual(_bits(frame.loc[0, "feature__large_integer"]), _bits(-0.0))
        self.assertEqual(frame.loc[1, "feature__large_integer"], 9007199254740993)
        self.assertEqual(frame.loc[2, "feature__large_integer"], 9007199254740994)

        self.assertEqual(_bits(frame.loc[0, "feature__missing_float"]), _bits(-0.0))
        self.assertTrue(math.isnan(float(frame.loc[1, "feature__missing_float"])))
        self.assertEqual(_bits(frame.loc[2, "feature__missing_float"]), _bits(2.5))
        self.assertEqual(_bits(frame.loc[0, "feature__precise"]), _bits(0.1))
        self.assertEqual(_bits(frame.loc[1, "feature__precise"]), _bits(1.2345678901234567))
        self.assertEqual(preview.shape, (1, 7))
        self.assertEqual(_bits(preview.loc[0, "feature__negative_zero"]), _bits(-0.0))

    def test_negative_zero_after_chunk_boundary_keeps_row_alignment(self) -> None:
        rows = [
            "sample_id,label,feature__negative_zero,feature__ordinary_integer\n"
        ]
        for row_index in range(8194):
            label = '"quoted\nlabel"' if row_index == 8191 else f"label-{row_index}"
            negative_zero = "-0" if row_index in (8191, 8192) else str(row_index)
            rows.append(
                f"{row_index:05d},{label},{negative_zero},{row_index}\n"
            )
            if row_index == 8191:
                rows.append("\n")

        with tempfile.TemporaryDirectory(prefix="pyml-csv-negative-zero-chunks-") as temporary:
            csv_path = Path(temporary) / "chunk-boundary.csv"
            csv_path.write_bytes("".join(rows).encode("utf-8"))
            frame = load_dataset(csv_path).frame

        self.assertEqual(frame.shape, (8194, 4))
        self.assertEqual(frame.loc[8191, "label"], "quoted\nlabel")
        self.assertEqual(_bits(frame.loc[8191, "feature__negative_zero"]), _bits(-0.0))
        self.assertEqual(_bits(frame.loc[8192, "feature__negative_zero"]), _bits(-0.0))
        self.assertEqual(frame.loc[8192, "sample_id"], "08192")
        self.assertEqual(frame.loc[8193, "feature__ordinary_integer"], 8193)
        self.assertIn(frame["feature__ordinary_integer"].dtype.kind, ("i", "u"))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
