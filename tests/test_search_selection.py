"""Bounded synthetic regressions for real search adapters and final isolation."""
import copy
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch
import warnings

import joblib
import numpy as np
import pandas as pd

from pyml_workbench.config import ExperimentConfig
from pyml_workbench.experiment import ExperimentError, build_snapshot, prepare_experiment, load_model
from pyml_workbench.objectives import cluster_objective
from pyml_workbench.search import SearchSpec, search, load_search_result
from pyml_workbench.search_space import SearchSpace
from pyml_workbench.selection import freeze_selection, refit_selected, finalize_selected, export_selected
from pyml_workbench import load_search_result as load_batch_search_result, run_batch


class SearchSelectionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "synthetic.csv"
        rng = np.random.default_rng(42)
        self.frame = pd.DataFrame({"x": rng.normal(size=120), "category": ["甲", "乙"] * 60, "target": np.arange(120) % 2})
        self.frame.to_csv(self.source, index=False)
        self.config = {"dataset": {"source_path": str(self.source), "target_column": "target", "feature_columns": ["x", "category"]}, "task": "classification", "model_id": "C01", "parameters": {"max_iter": 100}}
        self.space = {"fields": {"C": {"type": "real", "low": .1, "high": 2., "values": [.1, 1., 2.]}}}
        self.warning_context = warnings.catch_warnings()
        self.warning_context.__enter__()
        warnings.simplefilter("ignore", FutureWarning)
        self.addCleanup(self.warning_context.__exit__, None, None, None)

    def run_search(self, method="grid", **kwargs):
        spec = {"method": method, "space": self.space, "max_fits": 3}
        spec.update(kwargs.pop("spec", {}))
        return search(self.config, spec, **kwargs)

    def test_five_real_backends_are_bounded_and_do_not_test(self):
        for method in ("grid", "random", "annealing", "tpe", "genetic"):
            with self.subTest(method=method):
                events = []
                result = self.run_search(method, on_event=events.append, artifact_dir=self.root / method)
                self.assertEqual(result.actual_fit_count, 3)
                self.assertIsNotNone(result.winner)
                self.assertEqual(result.test_evaluation_count, 0)
                self.assertNotIn("test", result.winner_session.metrics)
                self.assertEqual(result.winner_session.test_evaluation_count, 0)
                self.assertEqual(len(result.winner_session.splits["train"]), 72)
                for event in events:
                    json.dumps(event, allow_nan=False)
                receipts = [event for event in events if event["name"] == "fit_started"]
                self.assertEqual([item["data"]["actual_fit_count"] for item in receipts], [1, 2, 3])
                self.assertTrue(all(item["data"]["record"]["config_sha256"] for item in receipts))
                self.assertTrue(all(record.test_evaluation_count == 0 for record in result.trials))
                if method != "grid":
                    self.assertEqual(result.stop_reason, "fit_limit")

    def test_conditional_grid_deduplicates_inactive_parameters(self):
        config = dict(self.config, model_id="C07", parameters={})
        space = {"fields": {"kernel": {"type": "choice", "values": ["linear", "rbf"]}, "gamma": {"type": "real", "low": .1, "high": 1., "values": [.1, 1.], "when": {"kernel": ["rbf"]}}}}
        result = search(config, {"space": space, "max_fits": 10})
        self.assertEqual(result.actual_fit_count, 3)
        self.assertEqual(result.proposal_count, 4)
        self.assertEqual(sum(record.cache_hit for record in result.trials), 1)
        for record in result.trials:
            self.assertEqual("gamma" in record.parameters, record.parameters["kernel"] == "rbf")

    def test_mixed_ga_and_conditional_tpe_preserve_integer_choice_types(self):
        config = dict(self.config, model_id="C07", parameters={})
        space = {"fields": {"C": {"type": "real", "low": .1, "high": 2.}, "kernel": {"type": "choice", "values": ["linear", "poly"]}, "degree": {"type": "integer", "low": 2, "high": 3, "when": {"kernel": ["poly"]}}}}
        for method in ("genetic", "tpe", "random"):
            result = search(config, {"method": method, "space": space, "max_fits": 4})
            self.assertEqual(result.actual_fit_count, 4)
            for record in result.trials:
                self.assertTrue(.1 <= record.parameters["C"] <= 2.)
                if record.parameters["kernel"] == "poly":
                    self.assertIsInstance(record.parameters["degree"], int)
                    self.assertIn(record.parameters["degree"], [2, 3])
                else:
                    self.assertNotIn("degree", record.parameters)

    def test_annealing_keeps_discrete_parameters_fixed(self):
        config = dict(self.config, model_id="C07", parameters={"kernel": "linear"})
        space = {"fields": dict(self.space["fields"], kernel={"type": "choice", "values": ["linear", "rbf"]})}
        result = search(config, {"method": "annealing", "space": space, "max_fits": 3})
        self.assertTrue(all(record.parameters["kernel"] == "linear" for record in result.trials))
        config["parameters"] = {}
        with self.assertRaises(ValueError):
            search(config, {"method": "annealing", "space": space, "max_fits": 2})

    def test_space_validation_rejects_cycles_bounds_threads_unknown_fields(self):
        cases = [
            {"fields": {"C": {"type": "real", "low": 2., "high": 1.}}},
            {"fields": {"C": {"type": "real", "low": .1, "high": 1., "when": {"C": [1.]}}}},
            {"fields": {"n_jobs": {"type": "integer", "low": 1, "high": 2}}},
        ]
        for value in cases:
            with self.assertRaises(ValueError):
                SearchSpace.from_dict(value)
        with self.assertRaises(ValueError):
            search(self.config, {"space": {"fields": {"invented": {"type": "choice", "values": [1, 2]}}}})
        config = dict(self.config, parameters={"n_jobs": -1})
        with self.assertRaises(ValueError):
            search(config, {"space": self.space})

    def test_fixed_only_grid_fits_manual_c01_parameters_once_without_test(self):
        space = {"fields": {}, "fixed": {"max_iter": 80}}
        batch = run_batch(
            self.root / "history.sqlite3",
            self.root / "artifacts",
            [(self.config, {"method": "grid", "space": space, "max_fits": 5})],
        )

        outcome = batch["results"][0]
        job_id = batch["jobs"][0]["job_id"]
        self.assertEqual(outcome["status"], "completed")
        self.assertEqual(outcome["result"]["actual_fit_count"], 1)
        self.assertEqual(outcome["result"]["proposal_count"], 1)
        self.assertEqual(outcome["result"]["test_evaluation_count"], 0)
        restored = load_batch_search_result(batch["history_path"], job_id)
        self.assertIsNotNone(restored)
        self.assertEqual(restored.winner.parameters["max_iter"], 80)

        with self.assertRaisesRegex(ValueError, "Fixed-only.*grid"):
            SearchSpec(method="random", space=SearchSpace.from_dict(space))
        with self.assertRaisesRegex(ValueError, "Fixed-only.*grid"):
            SearchSpec(method="tpe", space=SearchSpace.from_dict(space))

        invalid_type = search(
            self.config,
            {"method": "grid", "space": {"fields": {}, "fixed": {"max_iter": "many"}}},
        )
        self.assertEqual(invalid_type.actual_fit_count, 0)
        self.assertEqual(invalid_type.trials[0].status, "invalid")
        with self.assertRaisesRegex(ValueError, "Unknown estimator"):
            search(
                self.config,
                {"method": "grid", "space": {"fields": {}, "fixed": {"invented": 1}}},
            )

    def test_failed_fits_consume_budget_and_persistence_failure_blocks_fit(self):
        with patch("pyml_workbench.search.prepare_experiment", side_effect=RuntimeError("synthetic fit failure")) as fitted:
            result = self.run_search(spec={"max_fits": 2})
        self.assertEqual(fitted.call_count, 2)
        self.assertEqual(result.actual_fit_count, 2)
        self.assertEqual(result.status, "no_valid_trial")
        self.assertTrue(all(record.status == "failed" for record in result.trials))
        def callback(event):
            if event["name"] == "fit_started":
                raise RuntimeError("synthetic storage failure")
        with patch("pyml_workbench.search.prepare_experiment") as fitted:
            with self.assertRaisesRegex(RuntimeError, "storage"):
                self.run_search(on_event=callback)
        fitted.assert_not_called()

    def test_soft_time_limit_completes_active_fit_and_pause_time_is_excluded(self):
        def callback(event):
            if event["name"] == "fit_started":
                time.sleep(.03)
        result = self.run_search(spec={"timeout_seconds": .02}, on_event=callback)
        self.assertEqual(result.actual_fit_count, 1)
        self.assertEqual(result.stop_reason, "time_limit")
        self.assertGreaterEqual(result.elapsed_seconds, .03)
        clock = [100.]
        def wait():
            clock[0] += 1000.
            return True
        with patch("pyml_workbench.search.time.monotonic", side_effect=lambda: clock[0]):
            paused = self.run_search(spec={"max_fits": 1, "timeout_seconds": .01}, wait_for_dispatch=wait)
        self.assertEqual(paused.actual_fit_count, 1)
        self.assertEqual(paused.elapsed_seconds, 0.)

    def test_proposal_limit_and_cancel_are_bounded(self):
        spec = {"space": {"fields": {"C": {"type": "choice", "values": [1., 1., 1.]}}}, "max_fits": 5, "max_proposals": 2}
        result = self.run_search(spec=spec)
        self.assertEqual(result.actual_fit_count, 1)
        self.assertEqual(result.proposal_count, 2)
        self.assertEqual(result.stop_reason, "proposal_limit")
        cancelled = self.run_search(should_cancel=lambda: True)
        self.assertEqual(cancelled.actual_fit_count, 0)
        self.assertEqual(cancelled.status, "cancelled")

    def test_snapshot_file_change_and_mutation_guards(self):
        snapshot = build_snapshot(self.config)
        original_hash = snapshot.manifest["source_sha256"]
        altered = self.frame.copy()
        altered["category"] = "constant"
        altered.to_csv(self.source, index=False)
        result = self.run_search(snapshot=snapshot)
        self.assertEqual(result.snapshot.manifest["source_sha256"], original_hash)
        self.assertEqual(result.winner.objective_value, 1.)
        snapshot.features.iloc[0, 0] = 12345.
        with self.assertRaisesRegex(ExperimentError, "modified"):
            self.run_search(snapshot=snapshot)

    def test_resume_loads_winner_without_refit_and_rejects_another_group(self):
        result = self.run_search(spec={"max_fits": 1}, artifact_dir=self.root / "search")
        summary = result.to_dict()
        self.source.unlink()  # Restore from persisted data, not the now-missing source.
        with patch("pyml_workbench.search.prepare_experiment", side_effect=AssertionError("must not refit")):
            loaded = load_search_result(summary)
            resumed = self.run_search(spec={"max_fits": 1}, resume=summary, artifact_dir=self.root / "search")
        np.testing.assert_array_equal(loaded.winner_session.fitted_model.predict(self.frame), result.winner_session.fitted_model.predict(self.frame))
        np.testing.assert_array_equal(loaded.winner_session.fitted_model.estimator.coef_, result.winner_session.fitted_model.estimator.coef_)
        self.assertEqual(resumed.actual_fit_count, 1)
        self.assertEqual(resumed.stop_reason, "fit_limit")
        changed = copy.deepcopy(summary)
        changed["fingerprint"] = "another backend"
        with self.assertRaises(ValueError):
            load_search_result(changed)

    def test_cluster_coverage_and_cluster_count_boundaries(self):
        values = np.array([[0, 0], [0, 1], [1, 0], [1, 1], [10, 10], [10, 11], [11, 10], [11, 11], [100, 100], [200, 200]])
        score, metrics = cluster_objective(values, [0] * 4 + [1] * 4 + [-1] * 2)
        self.assertEqual(metrics["coverage"], .8)
        self.assertIsNotNone(score)
        self.assertIsNone(cluster_objective(values, [0] * 4 + [1] * 3 + [-1] * 3)[0])
        self.assertIsNone(cluster_objective(values, [0] * 8 + [-1] * 2)[0])
        self.assertIsNone(cluster_objective(values, [-1] * 10)[0])

    def test_unsupported_heldout_never_enters_validation_winner(self):
        config = dict(self.config, model_id="K09", task="clustering", parameters={})
        result = search(config, {"space": {"fields": {"eps": {"type": "real", "low": .3, "high": 1., "values": [.3, 1.]}}}, "max_fits": 2})
        self.assertIsNone(result.winner)
        self.assertIsNone(result.best_validation)
        self.assertTrue(all(record.status == "unscorable" for record in result.trials))
        self.assertEqual(result.test_evaluation_count, 0)

    def test_anomaly_objective_is_independent_and_absent_objective_is_blocked(self):
        config = dict(self.config, task="anomaly detection", model_id="A01", parameters={"n_estimators": 10})
        space = {"fields": {"contamination": {"type": "real", "low": .1, "high": .4, "values": [.1, .4]}}}
        with self.assertRaisesRegex(ValueError, "objective_labels"):
            search(config, {"space": space, "max_fits": 2})
        config["dataset"] = dict(config["dataset"], feature_columns=None)
        result = search(config, {"space": space, "max_fits": 2, "objective": {"objective_labels_column": "target", "label_mapping": {"0": 1, "1": -1}}})
        self.assertIsNotNone(result.winner)
        self.assertNotIn("target", result.winner_session.fitted_model.feature_columns)
        self.assertIn("balanced_accuracy", result.winner.metrics)
        self.assertEqual(result.test_evaluation_count, 0)

    def test_regression_minimizes_rmse(self):
        frame = self.frame.copy()
        frame["target"] = 2 * frame["x"] + np.random.default_rng(11).normal(scale=.2, size=len(frame))
        frame.to_csv(self.source, index=False)
        config = dict(self.config, model_id="R02", task="regression", parameters={})
        result = search(config, {"space": {"fields": {"alpha": {"type": "real", "low": .01, "high": 10., "values": [.01, 10.]}}}, "max_fits": 2})
        self.assertEqual(result.winner.direction, "min")
        self.assertEqual(result.winner.objective_value, min(record.objective_value for record in result.trials))

    def test_final_refit_80_percent_test_once_reload_and_export(self):
        result = self.run_search(artifact_dir=self.root / "search")
        frozen = freeze_selection(result)
        path = self.root / "final.joblib"
        events = []
        final = refit_selected(frozen, session_path=path, on_event=events.append)
        self.assertEqual(final.session.test_evaluation_count, 0)
        self.assertNotIn("test", final.session.metrics)
        self.assertNotIn("validation", final.session.metrics)
        self.assertEqual(len(final.session.splits["train"]), 96)
        self.assertEqual(final.session.fitted_model.preprocessing["fit_row_count"], 96)
        expected_mean = frozen.snapshot.features.iloc[final.session.splits["train"]]["x"].mean()
        scaler = final.session.fitted_model.preprocessor.named_transformers_["numeric"].named_steps["scaler"]
        self.assertAlmostEqual(scaler.mean_[0], expected_mean)
        self.assertEqual(final.to_dict()["selection_validation"], result.winner.metrics)
        observed_test = []
        def callback(event):
            if event["name"] == "test_started":
                self.assertEqual(joblib.load(path)["state"], "testing")
                observed_test.append(event["final_run_id"])
        tested = finalize_selected(frozen, session_path=path, on_event=callback)
        self.assertEqual(tested.session.test_evaluation_count, 1)
        with patch("pyml_workbench.selection.prepare_experiment", side_effect=AssertionError("no second refit")), patch("pyml_workbench.selection.evaluate_test", side_effect=AssertionError("no second test")):
            repeated = finalize_selected(frozen, session_path=path)
        self.assertTrue(repeated.cached)
        self.assertEqual(repeated.final_run_id, tested.final_run_id)
        self.assertEqual(observed_test, [tested.final_run_id])
        self.assertEqual(tested.session.result.audit["fit_scope"], "train_validation")
        self.assertEqual(tested.session.result.audit["snapshot_manifest"]["split_sha256"], result.snapshot.manifest["split_sha256"])
        np.testing.assert_array_equal(tested.session.splits["test"], result.snapshot.splits["test"])
        artifacts = export_selected(tested, self.root / "export")
        loaded = load_model(artifacts["model"])
        np.testing.assert_array_equal(loaded.predict(self.frame), tested.session.fitted_model.predict(self.frame))
        self.assertFalse(hasattr(loaded, "features"))
        self.assertTrue(Path(artifacts["results_xlsx"]).exists())
        self.assertEqual(loaded.manifest()["preprocessing"]["fit_scope"], "train_validation")
        self.assertEqual(loaded.manifest()["preprocessing"]["split_sha256"], result.snapshot.manifest["split_sha256"])
        self.assertEqual(finalize_selected(frozen, session_path=path).artifact_paths, artifacts)

    def test_wrong_winner_modified_selection_and_interrupted_test_rejected(self):
        result = self.run_search()
        with self.assertRaises(ExperimentError):
            freeze_selection(result, winner_trial_id="wrong-winner")
        frozen = freeze_selection(result)
        path = self.root / "final.joblib"
        refit_selected(frozen, session_path=path)
        with patch("pyml_workbench.selection.evaluate_test", side_effect=RuntimeError("synthetic final failure")):
            with self.assertRaises(RuntimeError):
                finalize_selected(frozen, session_path=path)
        self.assertEqual(joblib.load(path)["state"], "test_failed")
        with self.assertRaises(ExperimentError):
            finalize_selected(frozen, session_path=path)
        frozen.config.parameters["C"] = 123.
        with self.assertRaises(ExperimentError):
            refit_selected(frozen, session_path=self.root / "changed.joblib")

    def test_resume_underreported_budget_and_final_data_tamper_rejected(self):
        result = self.run_search(spec={"max_fits": 1}, artifact_dir=self.root / "search")
        summary = result.to_dict()
        summary["actual_fit_count"] = 0
        with self.assertRaises(ValueError):
            load_search_result(summary)
        frozen = freeze_selection(result)
        path = self.root / "final.joblib"
        refit_selected(frozen, session_path=path)
        saved = joblib.load(path)
        saved["session"].features.iloc[0, 0] = 9999.
        joblib.dump(saved, path)
        with self.assertRaises(ExperimentError):
            finalize_selected(frozen, session_path=path)

    def test_final_labeled_anomaly_objective_is_reported_once(self):
        config = dict(self.config, task="anomaly detection", model_id="A01", parameters={"n_estimators": 10})
        config["dataset"] = dict(config["dataset"], feature_columns=None)
        spec = {"space": {"fields": {"contamination": {"type": "real", "low": .1, "high": .4, "values": [.1, .4]}}}, "max_fits": 2, "objective": {"objective_labels_column": "target", "label_mapping": {"0": 1, "1": -1}}}
        result = search(config, spec)
        frozen = freeze_selection(result)
        final = finalize_selected(frozen, session_path=self.root / "anomaly-final.joblib")
        self.assertIn("balanced_accuracy", final.to_dict()["final_test"])
        self.assertNotIn("target", final.session.fitted_model.feature_columns)
        repeated = finalize_selected(frozen, session_path=self.root / "anomaly-final.joblib")
        self.assertEqual(repeated.to_dict()["final_test"], final.to_dict()["final_test"])
        self.assertEqual(repeated.session.test_evaluation_count, 1)


if __name__ == "__main__":
    unittest.main()
