"""Reference comparisons and lifecycle tests for the metrics/model expansion."""
import json
from pathlib import Path
import tempfile
import unittest
from dataclasses import replace
from unittest.mock import patch

import numpy as np
import pandas as pd
from sklearn import metrics as skm
from threadpoolctl import threadpool_limits

from pyml_workbench import (
    DatasetConfig, ExperimentConfig, SplitConfig, EvaluationConfig, ObjectiveSpec,
    prepare_experiment, freeze_experiment, evaluate_test, load_model, list_models,
    SearchSpec, SearchSpace, run_batch, freeze_search_winner, finalize_frozen_search,
)
from pyml_workbench.experiment import build_snapshot
from pyml_workbench.metrics import classification_metrics, regression_metrics, probability_cache, anomaly_metrics
from pyml_workbench.search import search


class MetricsTests(unittest.TestCase):
    def test_seeded_binary_and_multiclass_against_sklearn(self):
        rng = np.random.default_rng(20261027)
        for count in (2, 3, 7):
            for _ in range(10):
                actual = np.r_[np.arange(count), rng.integers(count, size=100)]
                probabilities = rng.dirichlet(np.ones(count), size=len(actual))
                predicted = probabilities.argmax(axis=1)
                result = classification_metrics(actual, predicted, probability_cache(probabilities, list(range(count))))
                truth = np.eye(count)[actual]
                columns = [1] if count == 2 else range(count)
                expected_auc = np.mean([skm.roc_auc_score(truth[:, i], probabilities[:, i]) for i in columns])
                expected_ap = np.mean([skm.average_precision_score(truth[:, i], probabilities[:, i]) for i in columns])
                self.assertAlmostEqual(result["roc_auc"], expected_auc)
                self.assertAlmostEqual(result["average_precision"], expected_ap)
                self.assertAlmostEqual(result["log_loss"], skm.log_loss(actual, probabilities))
                self.assertAlmostEqual(result["mcc"], skm.matthews_corrcoef(actual, predicted))
                self.assertAlmostEqual(result["f1_weighted"], skm.f1_score(actual, predicted, average="weighted", zero_division=0))
                json.dumps(result, allow_nan=False)

    def test_score_order_margins_missing_class_and_ap_is_not_trapezoid(self):
        actual = np.array(["b", "a", "b", "a"])
        cache = dict(available=True, class_order=["b", "a"], score_type="decision_score", values=[[.1], [.2], [.3], [.4]])
        result = classification_metrics(actual, actual, cache)
        self.assertAlmostEqual(result["roc_auc"], .75)
        self.assertIsNone(result["log_loss"])
        self.assertNotEqual(result["average_precision"], result["pr_auc"])
        missing = classification_metrics([0, 0], [0, 0], probability_cache([[.7, .3], [.8, .2]], [0, 1]))
        self.assertIsNone(missing["roc_auc"])
        json.dumps(missing, allow_nan=False)

    def test_regression_domains_mase_and_horizons(self):
        result = regression_metrics([1, 2, 3], [1, 3, 5], training_series=[np.array([1, 2, 4])], horizons=[1, 2, 2])
        self.assertAlmostEqual(result["mase"], 1 / 1.5)
        self.assertAlmostEqual(result["msle"], skm.mean_squared_log_error([1, 2, 3], [1, 3, 5]))
        self.assertEqual(len(result["by_horizon"]), 2)
        invalid = regression_metrics([0, -1], [0, 2])
        self.assertIsNone(invalid["mape"])
        self.assertIsNone(invalid["msle"])
        self.assertIsNone(invalid["mase"])
        json.dumps(invalid, allow_nan=False)

    def test_anomaly_score_sign_and_objective_directions(self):
        result = anomaly_metrics([-1, 1, -1, 1], [-1, 1, 1, 1], [-10, 3, -4, 6])
        self.assertEqual(result["roc_auc"], 1.)
        self.assertEqual(ObjectiveSpec(metric="log_loss").resolve("classification").direction, "min")
        self.assertEqual(ObjectiveSpec(metric="r2").resolve("regression").direction, "max")
        with self.assertRaises(ValueError):
            ObjectiveSpec(metric="log_loss", direction="max").resolve("classification")

    def test_deep_numeric_labels_keep_original_names(self):
        from pyml_workbench.extended_experiment import _deep_classification_metrics
        result = _deep_classification_metrics([0, 1, 0, 1], [0, 1, 0, 1],
            np.array([[3., 0.], [0., 3.], [2., 0.], [0., 2.]]),
            [{"label": 10, "encoded": 0}, {"label": 20, "encoded": 1}])
        self.assertEqual(result["accuracy"], 1)
        self.assertEqual([row["class_label"] for row in result["per_class"]], [10, 20])
        self.assertEqual(result["score_class_order"], [10, 20])


class IntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="pyml-expansion-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        rng = np.random.default_rng(20261027)
        values = rng.normal(size=(180, 5))
        self.frame = pd.DataFrame(values, columns=list("abcde"))
        self.frame["label"] = np.where(values[:, 0] + values[:, 1] > 0, "yes", "no")
        self.frame["target"] = 10 + values[:, 0] * 2 + rng.normal(scale=.2, size=len(values))
        self.path = self.root / "data.csv"
        self.frame.to_csv(self.path, index=False)
        self.limit = threadpool_limits(limits=1)
        self.addCleanup(self.limit.restore_original_limits)

    def config(self, model, task="regression", **kwargs):
        target = "label" if task == "classification" else "target" if task == "regression" else None
        return ExperimentConfig(DatasetConfig(str(self.path), target_column=target, feature_columns=tuple("abcde")), task, model, **kwargs)

    def test_new_traditional_models_train_freeze_test_export_reload(self):
        cases = {"classification": ["C28", "C29", "C30", "C31"],
                 "regression": ["R27", "R28", "R29", "R30", "R31", "R32", "R33", "R34", "R35", "R36"],
                 "clustering": ["K11", "K12", "K13"]}
        for task, models in cases.items():
            for model in models:
                with self.subTest(model=model):
                    config = self.config(model, task, output_dir=str(self.root / model))
                    session = prepare_experiment(config)
                    self.assertNotIn("test", session.metrics)
                    self.assertEqual(session.test_evaluation_count, 0)
                    if model == "K13":
                        self.assertEqual(session.metrics["validation"]["status"], "not_supported")
                    freeze_experiment(session)
                    result = evaluate_test(session)
                    json.dumps(result.metrics, allow_nan=False)
                    restored = load_model(result.artifact_paths["model"])
                    if restored.capabilities["predict"]:
                        np.testing.assert_array_equal(restored.predict(self.frame.iloc[:4]), session.fitted_model.predict(self.frame.iloc[:4]))
                    previous_count = session.test_evaluation_count
                    self.assertIs(evaluate_test(session), result)
                    self.assertEqual(session.test_evaluation_count, previous_count)

    def test_probability_metrics_on_real_margin_and_probability_models(self):
        for model in ("C01", "C06"):
            session = prepare_experiment(self.config(model, "classification"))
            self.assertIsNotNone(session.metrics["validation"]["roc_auc"])
            self.assertEqual(session.metrics["validation"]["log_loss"] is None, model == "C06")

    def test_cv_search_budget_fold_isolation_and_persistent_finalization(self):
        config = self.config("C01", "classification", evaluation=EvaluationConfig(folds=3))
        spec = SearchSpec(space=SearchSpace(fields={"C": {"type": "real", "low": .1, "high": 1., "values": [.1, 1.]}}), objective=ObjectiveSpec(metric="log_loss"), max_fits=8)
        result = search(config, spec)
        self.assertEqual(result.actual_fit_count, 8)
        self.assertEqual(len(result.trials), 2)
        self.assertEqual([record.fit_count for record in result.trials], [4, 4])
        self.assertEqual(result.winner.direction, "min")
        test = set(result.snapshot.splits["test"])
        for fold in result.winner.metrics["cross_validation"]["folds"]:
            self.assertFalse(test.intersection(fold["train_positions"]))
            self.assertFalse(test.intersection(fold["validation_positions"]))
            self.assertFalse(set(fold["train_positions"]).intersection(fold["validation_positions"]))
        history = self.root / "batch.sqlite"
        outcomes = run_batch(history, self.root / "batch", [(config, spec)])
        self.assertEqual(outcomes["results"][0]["status"], "completed")
        job = outcomes["jobs"][0]["job_id"]
        frozen = freeze_search_winner(history, job)
        self.assertEqual(frozen.metadata["validation_metrics"]["cross_validation"]["fold_count"], 3)
        final = finalize_frozen_search(history, job)
        self.assertEqual(final.session.test_evaluation_count, 1)
        self.assertEqual(finalize_frozen_search(history, job).session.test_evaluation_count, 1)

    def test_group_and_rolling_splits(self):
        frame = self.frame.copy()
        frame["group"] = np.repeat(np.arange(18), 10)
        frame["time"] = np.arange(len(frame))[::-1]
        frame.to_csv(self.path, index=False)
        for strategy, column in (("group", "group"), ("time", "time")):
            with self.subTest(strategy=strategy):
                split = SplitConfig(strategy=strategy, **{strategy + "_column": column})
                config = self.config("R02", split=split, evaluation=EvaluationConfig(folds=3, gap=1 if strategy == "time" else 0))
                session = prepare_experiment(config)
                if strategy == "group":
                    groups = [set(frame.iloc[part]["group"]) for part in session.splits.values()]
                    self.assertFalse(groups[0] & groups[1] | groups[1] & groups[2] | groups[0] & groups[2])
                else:
                    values = [frame.iloc[session.splits[name]]["time"] for name in ("train", "validation", "test")]
                    self.assertLess(values[0].max(), values[1].min())
                    self.assertLess(values[1].max(), values[2].min())
                self.assertEqual(session.metrics["cross_validation"]["scheme"], strategy)

    def test_fold_preprocessing_matches_reference_pipeline(self):
        from sklearn.pipeline import Pipeline
        from sklearn.metrics import mean_squared_error
        from pyml_workbench.catalog import build_estimator
        from pyml_workbench.preprocessing import build_preprocessor
        config = self.config("R02", evaluation=EvaluationConfig(folds=3))
        session = prepare_experiment(config)
        for fold in session.metrics["cross_validation"]["folds"]:
            train = fold["train_positions"]
            val = fold["validation_positions"]
            transformer, _ = build_preprocessor(self.frame.loc[train, list("abcde")], "R02")
            reference = Pipeline([("features", transformer), ("model", build_estimator("R02"))])
            reference.fit(self.frame.loc[train, list("abcde")], self.frame.loc[train, "target"])
            predicted = reference.predict(self.frame.loc[val, list("abcde")])
            self.assertAlmostEqual(fold["metrics"]["mse"], mean_squared_error(self.frame.loc[val, "target"], predicted))

    def test_old_canonical_config_and_new_preset_roundtrip(self):
        from pyml_workbench.presets import save_preset, load_preset
        old = self.config("R02").to_dict()
        self.assertNotIn("reference_labels_column", old["dataset"])
        self.assertNotIn("strategy", old["split"])
        self.assertNotIn("evaluation", old)
        config = self.config("R02", split=SplitConfig(strategy="time"), evaluation=EvaluationConfig(folds=3, gap=2))
        path = self.root / "settings.json"
        save_preset(path, config)
        self.assertEqual(load_preset(path).to_dict(), config.to_dict())

    def test_interrupted_fold_candidate_recovers_without_new_proposal(self):
        from pyml_workbench import create_search_job, run_search_job, HistoryStore
        history = self.root / "recover.sqlite"
        config = self.config("R02", evaluation=EvaluationConfig(folds=3))
        spec = SearchSpec(max_fits=4)
        job = create_search_job(history, self.root / "recover", config, spec)
        def interrupt(event):
            if event.get("name") == "fit_started" and event.get("data", {}).get("fold") == 2:
                raise KeyboardInterrupt("simulated worker interruption")
        with self.assertRaises(KeyboardInterrupt):
            run_search_job(history, job["job_id"], on_event=interrupt)
        store = HistoryStore(history)
        self.assertEqual(store.get_job(job["job_id"])["actual_fit_count"], 3)
        result = run_search_job(history, job["job_id"])
        self.assertEqual(result.status, "complete")
        self.assertEqual(result.actual_fit_count, 4)
        self.assertEqual(store.get_job(job["job_id"])["proposal_count"], 1)

    def test_cv_timer_counts_fold_fits_and_keeps_pending_folds(self):
        from pyml_workbench.run_timing import SearchTimingEstimator
        now = [0.]
        with patch("pyml_workbench.run_timing.time.monotonic", side_effect=lambda: now[0]):
            timer = SearchTimingEstimator(default_fit_seconds=10)
            timer.update_event(dict(job_id="cv", name="search_started", data=dict(
                spec=SearchSpec(max_fits=4).to_dict(), fits_per_candidate=4)))
            self.assertEqual(timer.jobs["cv"].limit, 4)
            timer.update_event(dict(job_id="cv", name="fit_started", data=dict(
                actual_fit_count=1, proposal_count=1, record=dict(trial_id="t", fit_count=1))))
            self.assertEqual(timer.remaining_seconds(), 40)
            now[0] = 40
            timer.update_event(dict(job_id="cv", name="fit_completed", data=dict(
                actual_fit_count=4, proposal_count=1,
                record=dict(trial_id="t", status="complete", fit_count=4, duration_seconds=40))))
            self.assertEqual(timer.jobs["cv"].fit_duration_count, 4)
            self.assertEqual(timer._unit(timer.jobs["cv"]), 10)

    def test_new_adapter_search_and_neural_refit(self):
        from pyml_workbench.sequence import SequenceConfig
        requests = []
        for model, task, parameters in (("C30", "classification", {}), ("R27", "regression", {}),
                                        ("D12", "dimensionality reduction", {"max_epochs": 2}),
                                        ("N07", "regression", {"max_epochs": 2, "patience": 1}),
                                        ("N08", "regression", {"max_epochs": 2, "patience": 1}),
                                        ("N09", "regression", {"max_epochs": 2, "patience": 1})):
            sequence = SequenceConfig(window=5) if model.startswith("N") else None
            requests.append((self.config(model, task, parameters=parameters, sequence=sequence), SearchSpec(max_fits=1)))
        history = self.root / "new-search.sqlite"
        outcomes = run_batch(history, self.root / "new-search", requests)
        self.assertTrue(all(row["status"] == "completed" for row in outcomes["results"]), outcomes["results"])
        for job in outcomes["jobs"]:
            freeze_search_winner(history, job["job_id"])
            final = finalize_frozen_search(history, job["job_id"])
            self.assertEqual(final.session.test_evaluation_count, 1)

    def test_external_cluster_and_anomaly_metrics(self):
        frame = self.frame.copy()
        frame["outlier"] = np.where(np.arange(len(frame)) % 5 == 0, -1, 1)
        frame.to_csv(self.path, index=False)
        for model, task, label in (("K11", "clustering", "label"), ("A01", "anomaly detection", "outlier")):
            config = self.config(model, task)
            config = replace(config, dataset=replace(config.dataset, reference_labels_column=label))
            session = prepare_experiment(config)
            self.assertIn("ari" if model == "K11" else "average_precision", session.metrics["validation"])
            freeze_experiment(session)
            result = evaluate_test(session, export_artifacts=False)
            self.assertIn("ari" if model == "K11" else "average_precision", result.metrics["test"])

    def test_fixed_origin_forecasters(self):
        frame = pd.DataFrame({"target": 10 + np.arange(120) * .1 + np.sin(np.arange(120) * 2 * np.pi / 12)})
        frame.to_csv(self.path, index=False)
        for model in ("T01", "T02", "T03", "T04", "T05"):
            with self.subTest(model=model):
                config = ExperimentConfig(DatasetConfig(str(self.path), target_column="target"), "regression", model, split=SplitConfig(strategy="time"), output_dir=str(self.root / model))
                session = prepare_experiment(config)
                self.assertIsNotNone(session.metrics["validation"]["mase"])
                self.assertEqual(session.metrics["validation"]["mase_seasonal_period"], 12 if model in {"T02", "T05"} else 1)
                self.assertEqual(len(session.metrics["validation"]["by_horizon"]), 24)
                np.testing.assert_allclose(session.fitted_model.predict(pd.DataFrame(index=range(4))), session.fitted_model.estimator.forecast(4))
                freeze_experiment(session)
                result = evaluate_test(session)
                rows = result.results.loc[result.results["split"] == "test", "y_pred"]
                np.testing.assert_allclose(rows, session.fitted_model.estimator.forecast(24, offset=24))
                restored = load_model(result.artifact_paths["model"])
                np.testing.assert_allclose(restored.predict(pd.DataFrame(index=range(4))), session.fitted_model.estimator.forecast(4))
                if model == "T01":
                    from argparse import Namespace
                    from pyml_workbench._worker import _inference
                    future = self.root / "future.csv"
                    pd.DataFrame({"future_step": range(4)}).to_csv(future, index=False)
                    output = self.root / "forecast.csv"
                    with patch("pyml_workbench._worker._emit"):
                        _inference(Namespace(model=Path(result.artifact_paths["model"]), operation="predict", data=future, sheet=None, output=output), {})
                    self.assertEqual(pd.read_csv(output)["forecast_horizon"].tolist(), [1, 2, 3, 4])

    def test_optional_deep_and_manifold_models(self):
        from pyml_workbench.sequence import SequenceConfig
        for model, task, parameters in (("D11", "dimensionality reduction", {"n_epochs": 10}),
                                        ("D12", "dimensionality reduction", {"max_epochs": 2}),
                                        ("N01", "classification", {"max_epochs": 2, "patience": 1}),
                                        ("N07", "regression", {"max_epochs": 2, "patience": 1}),
                                        ("N08", "regression", {"max_epochs": 2, "patience": 1}),
                                        ("N09", "regression", {"max_epochs": 2, "patience": 1})):
            with self.subTest(model=model):
                sequence = SequenceConfig(window=5) if model in {"N07", "N08", "N09"} else None
                config = self.config(model, task, parameters=parameters, sequence=sequence, output_dir=str(self.root / model))
                session = prepare_experiment(config)
                self.assertNotIn("test", session.metrics)
                if model == "N01":
                    self.assertIsNotNone(session.metrics["validation"]["log_loss"])
                if sequence:
                    self.assertIsNotNone(session.metrics["validation"]["mase"])
                freeze_experiment(session)
                result = evaluate_test(session)
                json.dumps(result.metrics, allow_nan=False)
                model_bundle = load_model(result.artifact_paths["model"])
                if sequence:
                    inputs = np.asarray(self.frame.loc[:4, list("abcde")], dtype=np.float32)[None, :, :]
                    np.testing.assert_allclose(model_bundle.predict(inputs), session.fitted_model.predict(inputs))
                elif task == "dimensionality reduction":
                    np.testing.assert_allclose(model_bundle.transform(self.frame.iloc[:4]), session.fitted_model.transform(self.frame.iloc[:4]))


if __name__ == "__main__":
    unittest.main()
