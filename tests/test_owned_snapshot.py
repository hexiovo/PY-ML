"""Integrity and source-independence coverage for owned dataset snapshots."""
from __future__ import annotations

import copy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from pyml_workbench.config import DatasetConfig, ExperimentConfig
from pyml_workbench.experiment import DatasetSnapshot, _json_digest, build_snapshot
from pyml_workbench.owned_snapshot import load_owned_snapshot, save_owned_snapshot


class OwnedSnapshotTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / "synthetic.csv"
        frame = pd.DataFrame(
            {
                "numeric": np.arange(60, dtype=float),
                "category": ["甲", "乙"] * 30,
                "target": np.arange(60) % 2,
            }
        )
        frame.to_csv(self.source, index=False)
        self.config = ExperimentConfig(
            dataset=DatasetConfig(
                source_path=str(self.source),
                target_column="target",
                feature_columns=("numeric", "category"),
            ),
            task="classification",
            model_id="C01",
        )
        self.snapshot = build_snapshot(self.config)
        self.path = self.root / "artifacts" / "job-1" / "snapshot.joblib"

    def save(self):
        return save_owned_snapshot(self.snapshot, self.path)

    def test_round_trip_preserves_snapshot_without_changing_source(self):
        source_before = self.source.read_bytes()

        receipt = self.save()
        restored = load_owned_snapshot(
            self.path,
            receipt,
            expected_manifest=self.snapshot.to_dict(),
            config=self.config,
        )

        self.assertIsInstance(restored, DatasetSnapshot)
        self.assertEqual(restored.to_dict(), self.snapshot.to_dict())
        pd.testing.assert_frame_equal(restored.features, self.snapshot.features)
        pd.testing.assert_series_equal(restored.target, self.snapshot.target)
        self.assertEqual(self.source.read_bytes(), source_before)
        self.assertEqual(receipt["relative_path"], "snapshot.joblib")
        self.assertEqual(receipt["manifest"], self.snapshot.to_dict())

    def test_saved_snapshot_survives_source_modification_and_deletion(self):
        receipt = self.save()
        expected_manifest = self.snapshot.to_dict()
        expected_features = self.snapshot.features.copy(deep=True)
        self.source.write_text("replacement data", encoding="utf-8")

        modified_source_load = load_owned_snapshot(
            self.path, receipt, expected_manifest=expected_manifest, config=self.config
        )
        pd.testing.assert_frame_equal(modified_source_load.features, expected_features)

        self.source.unlink()
        deleted_source_load = load_owned_snapshot(
            self.path, receipt, expected_manifest=expected_manifest, config=self.config
        )
        self.assertEqual(deleted_source_load.to_dict(), expected_manifest)
        pd.testing.assert_frame_equal(deleted_source_load.features, expected_features)

    def test_rejects_binary_tampering_before_deserialization(self):
        receipt = self.save()
        with self.path.open("ab") as stream:
            stream.write(b"tampered")

        with patch("pyml_workbench.owned_snapshot.joblib.load") as deserialize:
            with self.assertRaisesRegex(ValueError, "file checksum"):
                load_owned_snapshot(self.path, receipt)
        deserialize.assert_not_called()

    def test_rejects_receipt_and_expected_metadata_mismatch(self):
        receipt = self.save()
        changed_receipt = copy.deepcopy(receipt)
        changed_receipt["manifest"]["row_count"] += 1
        with self.assertRaisesRegex(ValueError, "manifest checksum"):
            load_owned_snapshot(self.path, changed_receipt)

        changed_receipt["manifest_sha256"] = _json_digest(changed_receipt["manifest"])
        with self.assertRaisesRegex(ValueError, "metadata does not match its receipt"):
            load_owned_snapshot(self.path, changed_receipt)

        expected_manifest = self.snapshot.to_dict()
        expected_manifest["row_count"] += 1
        with self.assertRaisesRegex(ValueError, "expected manifest"):
            load_owned_snapshot(self.path, receipt, expected_manifest=expected_manifest)

    def test_rejects_wrong_configuration(self):
        receipt = self.save()
        wrong_config = ExperimentConfig(
            dataset=DatasetConfig(
                source_path=str(self.source),
                target_column="target",
                feature_columns=("numeric",),
            ),
            task="classification",
            model_id="C01",
        )

        with self.assertRaisesRegex(RuntimeError, "Configuration does not belong"):
            load_owned_snapshot(self.path, receipt, config=wrong_config)


if __name__ == "__main__":
    unittest.main()
