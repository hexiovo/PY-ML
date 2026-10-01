from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from pyml_workbench.sequence import (
    SequenceConfig,
    SequenceError,
    _readonly_array,
    _verify_partition_invariants,
    build_sequence_plan,
    load_owned_sequence_plan,
    save_owned_sequence_plan,
)
from pyml_workbench.sequence_reporting import sequence_plan_summary, sequence_plan_summary_from_manifest


_SOURCE_SHA = "a" * 64


def _grouped_frame(*, category_test: str = "test-only", invalid_test_numeric: bool = False) -> pd.DataFrame:
    rows = []
    for group in range(15):
        for step in range(12):
            is_test = group >= 12
            rows.append(
                {
                    "group": f"g{group:02d}",
                    "time": pd.Timestamp("2025-01-01", tz="UTC") + pd.Timedelta(days=step),
                    "value": "bad-test-value" if invalid_test_numeric and is_test else group * 12 + step,
                    "kind": category_test if is_test else ("a" if step % 2 == 0 else "b"),
                    "target": "bad-test-target" if invalid_test_numeric and is_test else group * 0.5 + step,
                }
            )
    return pd.DataFrame(rows)


def _plan(frame: pd.DataFrame, model_id: str, *, window: int = 3, horizon: int = 2):
    if model_id in {"H01", "H02"}:
        settings = SequenceConfig(group_column="group", time_column="time", order_mode="time", observation_columns=("value",))
        task, target, features = "sequence_modeling", None, ["value"]
    elif model_id == "H03":
        settings = SequenceConfig(group_column="group", time_column="time", order_mode="time", observation_columns=("kind",))
        task, target, features = "sequence_modeling", None, ["kind"]
    else:
        settings = SequenceConfig(group_column="group", time_column="time", order_mode="time", window=window, horizon=horizon)
        task, target, features = "regression", "target", ["value"]
    return build_sequence_plan(
        frame,
        settings,
        task=task,
        model_id=model_id,
        target_column=target,
        feature_columns=features,
        split_seed=7,
        source_path="sequence-fixture.csv",
        source_sha256=_SOURCE_SHA,
    )


class SequencePlanTests(unittest.TestCase):
    def test_summary_reports_real_split_rows_groups_windows_and_proportions(self):
        plan = _plan(_grouped_frame(), "N06", window=4, horizon=3)
        summary = sequence_plan_summary(plan)
        self.assertEqual(summary["plan_sha256"], plan.plan_sha256)
        self.assertEqual(summary["totals"]["source_rows"], len(_grouped_frame()))
        self.assertEqual(summary["totals"]["groups"], 15)
        self.assertEqual(
            summary["totals"]["windows"],
            sum(len(plan.partitions[name].window_target_positions) for name in ("train", "validation", "test")),
        )
        for metric in ("source_rows", "groups", "windows"):
            self.assertAlmostEqual(
                sum(summary["splits"][name][f"{metric}_proportion"] for name in ("train", "validation", "test")),
                1.0,
            )
        self.assertEqual(sequence_plan_summary_from_manifest(plan.to_dict()), summary)

    def test_grouped_time_split_keeps_whole_groups_and_reports_actual_sizes(self):
        plan = _plan(_grouped_frame(), "N04")

        split_names = {}
        for name, partition in plan.partitions.items():
            for group in partition.group_ids:
                self.assertNotIn(group, split_names)
                split_names[group] = name
        self.assertEqual(len(split_names), 15)
        self.assertEqual([len(plan.split_positions[name]) for name in ("train", "validation", "test")], [108, 36, 36])
        self.assertEqual(plan.manifest["split_counts"], {"train": 108, "validation": 36, "test": 36})
        self.assertEqual([len(plan.partitions[name].window_targets) for name in ("train", "validation", "test")], [72, 24, 24])

    def test_windows_follow_each_split_segment_and_exact_horizon_offsets(self):
        frame = _grouped_frame()
        plan = _plan(frame, "N06", window=4, horizon=3)
        features = plan.features

        for partition in plan.partitions.values():
            for values, sources, target in zip(
                partition.window_inputs, partition.window_source_positions, partition.window_target_positions
            ):
                self.assertEqual(sources.tolist(), list(range(int(sources[0]), int(sources[0]) + 4)))
                self.assertEqual(int(target), int(sources[-1]) + 3)
                np.testing.assert_array_equal(values, features.iloc[sources].to_numpy(dtype=object))

    def test_window_verifier_rejects_a_gapped_or_shifted_mapping(self):
        plan = _plan(_grouped_frame(), "N04")
        partition = plan.train
        damaged_sources = partition.window_source_positions.copy()
        damaged_sources[0, 1] = damaged_sources[0, 1] + 1
        object.__setattr__(partition, "window_source_positions", _readonly_array(damaged_sources, dtype=np.int64))
        object.__setattr__(plan, "_manifest_json", json.dumps(plan._make_manifest(), sort_keys=True, ensure_ascii=False, separators=(",", ":")))

        with self.assertRaisesRegex(SequenceError, "window mapping violates"):
            _verify_partition_invariants(plan)

    def test_train_only_hmm_encodings_leave_test_categories_unchecked(self):
        plan_a = _plan(_grouped_frame(category_test="new-category"), "H03")
        plan_b = _plan(_grouped_frame(category_test="another-new-category"), "H03")

        self.assertEqual(plan_a.categorical_alphabet, ("a", "b"))
        self.assertEqual(plan_a.categorical_n_features, 2)
        self.assertEqual(plan_a.train.observation_encoding, "categorical_train_encoded")
        self.assertEqual(plan_a.validation.observation_encoding, "categorical_train_encoded")
        self.assertEqual(plan_a.test.observation_encoding, "categorical_raw_unchecked")
        np.testing.assert_array_equal(plan_a.train.observations, plan_b.train.observations)
        np.testing.assert_array_equal(plan_a.validation.observations, plan_b.validation.observations)

    def test_unknown_validation_category_is_rejected_using_train_alphabet(self):
        frame = _grouped_frame()
        frame.loc[frame["group"] == "g09", "kind"] = "validation-only"

        with self.assertRaisesRegex(SequenceError, "validation observation.*unknown to the train-only alphabet"):
            _plan(frame, "H03")

    def test_test_only_numeric_and_target_values_do_not_change_fit_partitions(self):
        plan_a = _plan(_grouped_frame(), "H01")
        plan_b = _plan(_grouped_frame(invalid_test_numeric=True), "H01")

        self.assertEqual(plan_a.train.observations.dtype, np.float32)
        self.assertEqual(plan_b.train.observations.dtype, np.float32)
        np.testing.assert_array_equal(plan_a.train.observations, plan_b.train.observations)
        np.testing.assert_array_equal(plan_a.validation.observations, plan_b.validation.observations)
        self.assertEqual(plan_b.test.observation_encoding, "continuous_raw_unchecked")

        window_a = _plan(_grouped_frame(), "N04")
        window_b = _plan(_grouped_frame(invalid_test_numeric=True), "N04")
        np.testing.assert_array_equal(window_a.train.window_inputs, window_b.train.window_inputs)
        np.testing.assert_array_equal(window_a.validation.window_inputs, window_b.validation.window_inputs)
        np.testing.assert_array_equal(window_a.train.window_targets, window_b.train.window_targets)
        np.testing.assert_array_equal(window_a.validation.window_targets, window_b.validation.window_targets)
        self.assertIsInstance(window_b.test.window_targets[0, 0], str)

    def test_continuous_hmm_and_window_model_families_have_shared_protocols(self):
        frame = _grouped_frame()
        hmm_continuous_a = _plan(frame, "H01")
        hmm_continuous_b = _plan(frame, "H02")
        hmm_categorical = _plan(frame, "H03")
        window_lstm = _plan(frame, "N04")
        window_gru = _plan(frame, "N06")

        self.assertEqual(hmm_continuous_a.model_family, "hmm_continuous")
        self.assertEqual(hmm_continuous_a.model_family, hmm_continuous_b.model_family)
        self.assertEqual(hmm_continuous_a.comparison_protocol_sha256, hmm_continuous_b.comparison_protocol_sha256)
        self.assertEqual(hmm_categorical.model_family, "hmm_categorical")
        self.assertEqual(window_lstm.comparison_protocol_sha256, window_gru.comparison_protocol_sha256)
        self.assertNotEqual(hmm_continuous_a.comparison_protocol_sha256, hmm_categorical.comparison_protocol_sha256)

    def test_owned_plan_round_trip_verifies_receipt_and_restores_readonly_arrays(self):
        plan = _plan(_grouped_frame(), "N04")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plan.joblib"
            receipt = save_owned_sequence_plan(plan, path)
            loaded = load_owned_sequence_plan(path, receipt, expected_manifest=plan.to_dict())

            self.assertEqual(loaded.plan_sha256, plan.plan_sha256)
            self.assertFalse(loaded.train.row_positions.flags.writeable)
            self.assertFalse(loaded.train.window_inputs.flags.writeable)
            self.assertFalse(loaded.train.window_source_positions.flags.writeable)

            broken = dict(receipt)
            broken["file_sha256"] = "0" * 64
            with self.assertRaisesRegex(SequenceError, "checksum"):
                load_owned_sequence_plan(path, broken)

    def test_plan_config_rejects_structural_columns_as_features(self):
        with self.assertRaisesRegex(SequenceError, "structural columns cannot be model features"):
            SequenceConfig(group_column="group", time_column="time", order_mode="time").validate(
                task="regression",
                model_id="N04",
                feature_columns=["group", "value"],
                target_column="target",
                available_columns=["group", "time", "value", "target"],
            )

    def test_ungrouped_row_order_uses_stable_contiguous_sixty_twenty_twenty_split(self):
        frame = pd.DataFrame({"value": np.arange(30, dtype=np.float32), "target": np.arange(30, dtype=np.float32)})
        plan = build_sequence_plan(
            frame,
            SequenceConfig(order_mode="row", window=3, horizon=1),
            task="regression",
            model_id="N04",
            target_column="target",
            feature_columns=["value"],
            split_seed=5,
            source_path="ungrouped-fixture.csv",
            source_sha256=_SOURCE_SHA,
        )

        np.testing.assert_array_equal(plan.train.row_positions, np.arange(18))
        np.testing.assert_array_equal(plan.validation.row_positions, np.arange(18, 24))
        np.testing.assert_array_equal(plan.test.row_positions, np.arange(24, 30))

    def test_missing_or_invalid_group_time_is_rejected(self):
        for bad_time in (None, "not-a-time"):
            frame = _grouped_frame()
            frame["time"] = frame["time"].astype(object)
            frame.loc[4, "time"] = bad_time

            with self.assertRaisesRegex(SequenceError, "missing or invalid"):
                _plan(frame, "N04")

    def test_duplicate_time_within_group_is_rejected(self):
        frame = _grouped_frame()
        frame.loc[1, "time"] = frame.loc[0, "time"]

        with self.assertRaisesRegex(SequenceError, "Duplicate time value within a group"):
            _plan(frame, "N04")

    def test_short_group_partition_fails_instead_of_dropping_rows_or_windows(self):
        frame = _grouped_frame()
        frame = frame.loc[~((frame["group"] == "g00") & (frame.index >= 4) & (frame.index <= 11))].copy()

        with self.assertRaisesRegex(SequenceError, "window=3, horizon=2 requires at least 5"):
            _plan(frame, "N04")

    def test_owned_plan_rejects_manifest_config_and_file_tampering(self):
        plan = _plan(_grouped_frame(), "N04")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plan.joblib"
            receipt = save_owned_sequence_plan(plan, path)

            with self.assertRaisesRegex(SequenceError, "manifest checksum"):
                load_owned_sequence_plan(path, {**receipt, "manifest": {**receipt["manifest"], "row_count": 999}})
            with self.assertRaisesRegex(SequenceError, "task does not match"):
                load_owned_sequence_plan(path, receipt, config={"task": "classification", "model_id": "N04"})
            with self.assertRaisesRegex(SequenceError, "queued plan"):
                load_owned_sequence_plan(path, receipt, expected_manifest={**plan.to_dict(), "row_count": 999})

            with path.open("ab") as stream:
                stream.write(b"tamper")
            with self.assertRaisesRegex(SequenceError, "file checksum"):
                load_owned_sequence_plan(path, receipt)


if __name__ == "__main__":
    unittest.main()
