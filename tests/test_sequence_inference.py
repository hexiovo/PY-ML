from __future__ import annotations

import hashlib
import json
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from pyml_workbench.extended_experiment import ExtendedExperimentError, ExtendedFittedModel
from pyml_workbench.sequence import SequenceConfig
from pyml_workbench.sequence_inference import sequence_inference_frame


class _RecordingHMM:
    def __init__(self):
        self.calls: list[tuple[str, np.ndarray, list[int]]] = []

    def predict(self, observations, *, lengths):
        values = np.asarray(observations).copy()
        self.calls.append(("predict", values, list(lengths)))
        return np.arange(len(values), dtype=np.int64)

    def predict_proba(self, observations, *, lengths):
        values = np.asarray(observations).copy()
        self.calls.append(("posterior", values, list(lengths)))
        return np.tile(np.asarray([[0.25, 0.75]], dtype=np.float64), (len(values), 1))


class _RecordingPreprocessor:
    def __init__(self):
        self.transformed_frames: list[pd.DataFrame] = []

    def transform(self, frame):
        self.transformed_frames.append(frame.copy(deep=True))
        category_code = frame["category"].map({"x": 1.0, "y": 2.0}).to_numpy(dtype=np.float32)
        value = pd.to_numeric(frame["value"], errors="raise").to_numpy(dtype=np.float32)
        return np.column_stack([value / 10.0, category_code])


def _bundle(
    model_id: str,
    *,
    feature_columns: tuple[str, ...],
    sequence_config: SequenceConfig,
    estimator,
    preprocessor=None,
    categorical_alphabet=None,
) -> ExtendedFittedModel:
    task = "sequence_modeling" if model_id.startswith("H") else "regression"
    target = None if model_id.startswith("H") else "target"
    config = {
        "dataset": {
            "source_path": "synthetic.csv",
            "target_column": target,
            "feature_columns": list(feature_columns),
        },
        "task": task,
        "model_id": model_id,
        "parameters": {},
        "split": {
            "seed": 42,
            "train_fraction": 0.6,
            "validation_fraction": 0.2,
            "test_fraction": 0.2,
        },
        "sequence": sequence_config.to_dict(),
    }
    frozen_sha = hashlib.sha256(
        json.dumps(config, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    ).hexdigest()
    return ExtendedFittedModel(
        estimator=estimator,
        preprocessor=preprocessor,
        model_id=model_id,
        task=task,
        feature_columns=list(feature_columns),
        preprocessing={},
        config=config,
        frozen_config_sha256=frozen_sha,
        parameters={},
        categorical_alphabet=categorical_alphabet,
        sequence_config=sequence_config.to_dict(),
        snapshot_manifest_sha256="a" * 64,
        sequence_plan_manifest_sha256="b" * 64,
    )


def _hmm_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "group": ["B", "A", "B", "A", "B", "A"],
            "time": ["2024-01-03", "2024-01-02", "2024-01-01", "2024-01-01", "2024-01-02", "2024-01-03"],
            "value": [30.0, 12.0, 10.0, 11.0, 20.0, 13.0],
            "kind": ["b", "a", "a", "b", "b", "a"],
        },
        index=["b3", "a2", "b1", "a1", "b2", "a3"],
    )


def _window_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "group": ["B", "A", "B", "A", "B", "A", "B", "A", "B", "A"],
            "time": [
                "2024-01-03", "2024-01-02", "2024-01-01", "2024-01-01", "2024-01-02",
                "2024-01-03", "2024-01-04", "2024-01-04", "2024-01-05", "2024-01-05",
            ],
            "value": [30.0, 12.0, 10.0, 11.0, 20.0, 13.0, 40.0, 14.0, 50.0, 15.0],
            "category": ["x", "y", "x", "y", "x", "y", "x", "y", "x", "y"],
            "target": list(range(100, 110)),
        },
        index=[f"row-{position}" for position in range(10)],
    )


class SequenceInferenceFrameTests(unittest.TestCase):
    def test_h03_uses_frozen_alphabet_and_stable_group_time_sorting(self):
        config = SequenceConfig(
            group_column="group",
            time_column="time",
            order_mode="time",
            observation_columns=("kind",),
        )
        estimator = _RecordingHMM()
        bundle = _bundle(
            "H03",
            feature_columns=("kind",),
            sequence_config=config,
            estimator=estimator,
            categorical_alphabet=["a", "b"],
        )
        frame = _hmm_frame()
        original = frame.copy(deep=True)

        result = sequence_inference_frame(bundle, frame)

        pd.testing.assert_frame_equal(frame, original)
        self.assertEqual(result["source_row_position"].tolist(), [2, 4, 0, 3, 1, 5])
        self.assertEqual(result["source_index"].tolist(), ["b1", "b2", "b3", "a1", "a2", "a3"])
        self.assertEqual(result["group_id"].tolist(), ["B", "B", "B", "A", "A", "A"])
        self.assertEqual(result["hidden_state"].tolist(), list(range(6)))
        self.assertEqual(result["posterior_0"].tolist(), [0.25] * 6)
        self.assertEqual(len(estimator.calls), 2)
        self.assertEqual(estimator.calls[0][0], "predict")
        self.assertEqual(estimator.calls[0][2], [3, 3])
        self.assertEqual(estimator.calls[0][1].reshape(-1).tolist(), [0, 1, 1, 1, 0, 0])

    def test_h03_rejects_categories_outside_the_frozen_alphabet(self):
        config = SequenceConfig(
            group_column="group",
            time_column="time",
            order_mode="time",
            observation_columns=("kind",),
        )
        estimator = _RecordingHMM()
        bundle = _bundle(
            "H03",
            feature_columns=("kind",),
            sequence_config=config,
            estimator=estimator,
            categorical_alphabet=["a", "b"],
        )
        frame = _hmm_frame()
        frame.loc["b3", "kind"] = "unseen"

        with self.assertRaisesRegex(ExtendedExperimentError, "unknown to the frozen train-only alphabet"):
            sequence_inference_frame(bundle, frame)
        self.assertEqual(estimator.calls, [])

    def test_continuous_hmm_passes_sorted_observations_and_group_lengths(self):
        config = SequenceConfig(
            group_column="group",
            time_column="time",
            order_mode="time",
            observation_columns=("value",),
        )
        estimator = _RecordingHMM()
        bundle = _bundle(
            "H01",
            feature_columns=("value",),
            sequence_config=config,
            estimator=estimator,
        )

        result = sequence_inference_frame(bundle, _hmm_frame())

        expected = np.asarray([[10.0], [20.0], [30.0], [11.0], [12.0], [13.0]], dtype=np.float32)
        np.testing.assert_array_equal(estimator.calls[0][1], expected)
        self.assertEqual(estimator.calls[0][2], [3, 3])
        self.assertEqual(result["source_row_position"].tolist(), [2, 4, 0, 3, 1, 5])

    def test_n04_n06_transform_fitted_windows_and_preserve_horizon_mapping(self):
        expected_source_positions = [[2, 4], [4, 0], [3, 1], [1, 5]]
        expected_target_positions = [6, 8, 7, 9]
        expected_flat_values = [10.0, 20.0, 20.0, 30.0, 11.0, 12.0, 12.0, 13.0]
        for model_id in ("N04", "N06"):
            with self.subTest(model_id=model_id):
                config = SequenceConfig(
                    group_column="group",
                    time_column="time",
                    order_mode="time",
                    window=2,
                    horizon=2,
                )
                preprocessor = _RecordingPreprocessor()
                estimator = object()
                bundle = _bundle(
                    model_id,
                    feature_columns=("value", "category"),
                    sequence_config=config,
                    estimator=estimator,
                    preprocessor=preprocessor,
                )
                captured: dict[str, object] = {}

                def fake_predict(fitted_estimator, called_model_id, values):
                    captured["estimator"] = fitted_estimator
                    captured["model_id"] = called_model_id
                    captured["values"] = np.asarray(values).copy()
                    return np.asarray([0.1, 0.2, 0.3, 0.4], dtype=np.float32)

                with patch("pyml_workbench.sequence_models.predict_deep", side_effect=fake_predict):
                    result = sequence_inference_frame(bundle, _window_frame())

                self.assertEqual(len(preprocessor.transformed_frames), 1)
                transformed_input = preprocessor.transformed_frames[0]
                self.assertEqual(list(transformed_input.columns), ["value", "category"])
                self.assertEqual(pd.to_numeric(transformed_input["value"]).tolist(), expected_flat_values)
                self.assertIs(captured["estimator"], estimator)
                self.assertEqual(captured["model_id"], model_id)
                transformed_windows = np.asarray(captured["values"])
                self.assertEqual(transformed_windows.shape, (4, 2, 2))
                np.testing.assert_allclose(transformed_windows[0], [[1.0, 1.0], [2.0, 1.0]])
                self.assertEqual(result["source_row_position"].tolist(), expected_target_positions)
                self.assertEqual(result["source_index"].tolist(), [f"row-{item}" for item in expected_target_positions])
                self.assertEqual(result["group_id"].tolist(), ["B", "B", "A", "A"])
                np.testing.assert_allclose(result["prediction"].to_numpy(), [0.1, 0.2, 0.3, 0.4])
                self.assertEqual(result["window_source_row_positions"].tolist(), expected_source_positions)
                self.assertEqual(
                    result["window_source_indexes"].tolist(),
                    [[f"row-{item}" for item in positions] for positions in expected_source_positions],
                )


if __name__ == "__main__":
    unittest.main()
