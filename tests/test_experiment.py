import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from pyml_workbench import (
    DatasetConfig,
    ExperimentConfig,
    MissingTargetError,
    load_model,
    predict,
    run_experiment,
)


def write_classification_csv(path: Path, *, missing_target: bool = False) -> pd.DataFrame:
    rows = np.arange(30)
    frame = pd.DataFrame(
        {
            "numeric": rows.astype(float),
            "category": ["seen"] * 18 + ["new_validation"] * 6 + ["new_test"] * 6,
            "target": rows % 2,
        },
        index=np.arange(100, 130),
    )
    frame.loc[frame.index[18:24], "numeric"] = 1_000_000.0
    frame.loc[frame.index[24:30], "numeric"] = -1_000_000.0
    if missing_target:
        frame.loc[frame.index[20], "target"] = np.nan
    frame.to_csv(path, index=False)
    return frame


def controlled_splits():
    return {
        "train": np.arange(0, 18, dtype=np.int64),
        "validation": np.arange(18, 24, dtype=np.int64),
        "test": np.arange(24, 30, dtype=np.int64),
    }


def classification_config(source: Path, *, output_dir: Path | None = None, parameters=None):
    return ExperimentConfig(
        dataset=DatasetConfig(
            source_path=str(source),
            target_column="target",
            feature_columns=("numeric", "category"),
        ),
        task="classification",
        model_id="C01",
        parameters={} if parameters is None else parameters,
        output_dir=None if output_dir is None else str(output_dir),
    )


class ExperimentRunnerTests(unittest.TestCase):
    def test_train_only_preprocessing_unknown_categories_split_and_freeze_order(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "synthetic.csv"
            write_classification_csv(source)
            with patch("pyml_workbench.experiment._split_rows", return_value=controlled_splits()):
                result = run_experiment(classification_config(source))

            numeric = result.model.preprocessor.named_transformers_["numeric"]
            self.assertEqual(numeric.named_steps["imputer"].statistics_[0], 8.5)
            self.assertEqual(numeric.named_steps["scaler"].mean_[0], 8.5)
            onehot = result.model.preprocessor.named_transformers_["categorical"].named_steps["onehot"]
            self.assertEqual(onehot.categories_[0].tolist(), ["seen"])
            transformed_unknown = result.model.preprocessor.transform(
                pd.DataFrame({"numeric": [1_000_000.0], "category": ["new_validation"]})
            )
            self.assertTrue(np.array_equal(transformed_unknown[:, 1:], np.zeros((1, 1))))

            positions = result.audit["split_positions"]
            self.assertEqual(set(positions), {"train", "validation", "test"})
            self.assertEqual(len(set(positions["train"]) | set(positions["validation"]) | set(positions["test"])), 30)
            self.assertEqual(
                set(positions["train"]) | set(positions["validation"]) | set(positions["test"]),
                set(range(30)),
            )
            self.assertTrue(result.audit["test_metrics_computed_after_freeze"])
            self.assertEqual(result.audit["test_evaluation_count"], 1)
            events = result.audit["events"]
            self.assertLess(events.index("selection_frozen"), events.index("test_metrics_computed_once"))

    def test_missing_target_is_rejected_before_fitting(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "missing_target.csv"
            write_classification_csv(source, missing_target=True)
            with self.assertRaises(MissingTargetError):
                run_experiment(classification_config(source))

    def test_external_parameter_mutation_cannot_relabel_the_trained_model(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "mutation.csv"
            write_classification_csv(source)
            parameters = {"C": 0.5, "max_iter": 100}
            config = classification_config(source, parameters=parameters)
            sklearn_fit = LogisticRegression.fit

            def mutate_after_training_starts(estimator, values, target):
                # Simulate a caller changing a shared nested config while fit runs.
                parameters["C"] = 50.0
                return sklearn_fit(estimator, values, target)

            with (
                patch("pyml_workbench.experiment._split_rows", return_value=controlled_splits()),
                patch.object(LogisticRegression, "fit", new=mutate_after_training_starts),
            ):
                result = run_experiment(config)

            self.assertEqual(parameters["C"], 50.0)
            self.assertEqual(result.config["parameters"]["C"], 0.5)
            self.assertEqual(result.model.estimator.C, 0.5)
            self.assertEqual(result.model.config["parameters"]["C"], 0.5)
            encoded = json.dumps(result.model.config, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            self.assertEqual(
                result.model.frozen_config_sha256,
                hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
            )

            exposed_snapshot = result.model.config
            exposed_snapshot["parameters"]["C"] = 999.0
            self.assertEqual(result.model.config["parameters"]["C"], 0.5)

    def test_saved_bundle_reloads_with_prediction_parity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "artifact_source.csv"
            output = root / "artifacts"
            frame = write_classification_csv(source)
            with patch("pyml_workbench.experiment._split_rows", return_value=controlled_splits()):
                result = run_experiment(classification_config(source, output_dir=output))

            rows = frame.iloc[24:30].loc[:, ["numeric", "category"]]
            expected = result.model.predict(rows)
            loaded = load_model(output)
            observed = loaded.predict(rows)
            public_api_observed = predict(result.artifact_paths["model"], rows)
            np.testing.assert_array_equal(expected, observed)
            np.testing.assert_array_equal(expected, public_api_observed)
            for name in ("config.json", "metrics.json", "metrics.csv", "metrics.xlsx", "results.csv", "results.xlsx", "model.joblib", "model_manifest.json"):
                self.assertTrue((output / name).is_file(), name)


if __name__ == "__main__":
    unittest.main()
