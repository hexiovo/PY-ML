import json
import unittest

from pyml_workbench import (
    ConfigError,
    DatasetConfig,
    EstimatorAdapter,
    ExperimentConfig,
    SplitConfig,
    UnsupportedOperationError,
)


class ConfigContractTests(unittest.TestCase):
    def test_experiment_config_round_trips_through_json(self):
        expected = ExperimentConfig(
            dataset=DatasetConfig(
                source_path="synthetic.csv",
                target_column="label",
                feature_columns=("numeric", "category"),
            ),
            task="classification",
            model_id="C01",
            parameters={"max_iter": 50},
            split=SplitConfig(seed=42),
            output_dir="results",
        )

        encoded = json.dumps(expected.to_dict())
        actual = ExperimentConfig.from_dict(json.loads(encoded))

        self.assertEqual(actual, expected)

    def test_supervised_config_requires_target_column(self):
        config = ExperimentConfig(
            dataset=DatasetConfig(source_path="synthetic.csv"),
            task="classification",
            model_id="C01",
        )

        with self.assertRaisesRegex(ConfigError, "target_column is required"):
            config.validate()


class AdapterContractTests(unittest.TestCase):
    def test_inference_requires_fit_and_unsupported_capability_is_guarded(self):
        adapter = EstimatorAdapter("C02")
        features = [[0.0], [1.0], [2.0], [3.0]]
        labels = [0, 0, 1, 1]

        with self.assertRaisesRegex(RuntimeError, "must be fitted"):
            adapter.predict(features)

        adapter.fit(features, labels)
        with self.assertRaises(UnsupportedOperationError):
            adapter.predict_proba([[1.5]])


if __name__ == "__main__":
    unittest.main()
