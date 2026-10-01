"""Tests for exporting a completed training-exploratory batch winner."""
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from pyml_workbench import (
    BatchError,
    DatasetConfig,
    ExperimentConfig,
    ExperimentError,
    ObjectiveSpec,
    SearchSpace,
    SearchSpec,
    SplitConfig,
    create_search_job,
    export_training_exploration,
    freeze_search_winner,
    get_job_summary,
    load_model,
    load_search_result,
    run_search_job,
)
from pyml_workbench import _worker as worker
from pyml_workbench.history import HistoryStore


class TrainingExplorationExportTests(unittest.TestCase):
    def test_completed_batch_winner_exports_same_fit_without_validation_or_test(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "embedding.csv"
            rng = np.random.default_rng(20261001)
            centers = np.array([[-2.5, -2.5, 0.0], [-2.5, 2.5, 1.0], [2.5, -2.5, -1.0], [2.5, 2.5, 0.5]])
            values = np.vstack([rng.normal(center, 0.4, size=(30, 3)) for center in centers])
            pd.DataFrame(values, columns=["x1", "x2", "x3"]).to_csv(source, index=False)
            config = ExperimentConfig(
                dataset=DatasetConfig(source_path=str(source), feature_columns=("x1", "x2", "x3")),
                task="dimensionality reduction",
                model_id="D10",
                split=SplitConfig(seed=42),
            )
            spec = SearchSpec(
                method="grid",
                space=SearchSpace.from_dict({"fields": {}, "fixed": {
                    "n_components": 2, "perplexity": 15.0, "max_iter": 250,
                }}),
                objective=ObjectiveSpec(split="train_exploratory"),
                max_fits=1,
                max_proposals=1,
                timeout_seconds=120.0,
                seed=42,
            )
            history = root / "history.sqlite3"
            artifact_root = root / "artifacts"
            job = create_search_job(history, artifact_root, config, spec)
            outcome = run_search_job(history, job["job_id"])
            self.assertEqual(outcome.status, "complete")
            self.assertEqual(outcome.actual_fit_count, 1)
            self.assertEqual(outcome.test_evaluation_count, 0)
            self.assertEqual(outcome.winner.score_split, "train_exploratory")
            self.assertIsNone(outcome.best_validation)
            self.assertIn("trustworthiness", outcome.winner.metrics)
            with self.assertRaisesRegex(ExperimentError, "independent-validation winner"):
                freeze_search_winner(history, job["job_id"])

            result_before = load_search_result(history, job["job_id"])
            self.assertIsNotNone(result_before)
            export_dir = root / "exports" / "d10"
            stdout, stderr = io.StringIO(), io.StringIO()
            with patch.dict(os.environ, {"PYML_LOG_ROOT": str(root / "diagnostics")}):
                with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
                    worker_code = worker.main([
                        "batch-export-exploration", "--history", str(history),
                        "--job-id", job["job_id"], "--output", str(export_dir),
                    ])
            self.assertEqual(worker_code, 0, stderr.getvalue())
            worker_payload = json.loads(stdout.getvalue().splitlines()[-1])
            self.assertEqual(worker_payload["type"], "result")
            self.assertEqual(worker_payload["phase"], "training_exploration_exported")
            exported = worker_payload["result"]
            self.assertEqual(exported["status"], "exported")
            self.assertEqual(exported["actual_fit_count"], 1)
            self.assertEqual(exported["test_evaluation_count"], 0)
            self.assertTrue(Path(exported["artifact_paths"]["results_csv"]).is_file())
            self.assertTrue(Path(exported["artifact_paths"]["manifest"]).is_file())

            reloaded = load_model(exported["artifact_paths"]["model"])
            self.assertEqual(reloaded.model_id, "D10")
            self.assertEqual(reloaded.frozen_config_sha256, outcome.winner.config_sha256)
            metrics = json.loads(Path(exported["artifact_paths"]["metrics_json"]).read_text(encoding="utf-8"))
            self.assertNotIn("test", metrics)
            manifest = json.loads(Path(exported["artifact_paths"]["manifest"]).read_text(encoding="utf-8"))
            self.assertFalse(manifest["validation_ranked"])
            self.assertEqual(manifest["score_split"], "train_exploratory")
            self.assertEqual(manifest["actual_fit_count"], 1)
            self.assertEqual(manifest["test_evaluation_count"], 0)

            summary = get_job_summary(history, job["job_id"])
            self.assertEqual(summary["artifact_paths"], exported["artifact_paths"])
            cached = export_training_exploration(history, job["job_id"], output_dir=export_dir)
            self.assertEqual(cached["status"], "cached")
            self.assertEqual(cached["artifact_paths"], exported["artifact_paths"])
            self.assertEqual(get_job_summary(history, job["job_id"])["actual_fit_count"], 1)
            self.assertEqual(get_job_summary(history, job["job_id"])["test_permission"], None)
            result_after = load_search_result(history, job["job_id"])
            self.assertEqual(result_after.actual_fit_count, result_before.actual_fit_count)
            self.assertEqual(result_after.winner.trial_id, result_before.winner.trial_id)

    def test_validation_search_cannot_use_training_exploration_export(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            history = root / "history.sqlite3"
            store = HistoryStore(history)
            job = store.create_job(
                job_id="validation-job",
                dataset_id="a" * 64,
                task="dimensionality reduction",
                model_id="D10",
                snapshot={},
                split_summary={},
                search_spec={"method": "grid", "objective": {"split": "validation"}, "space": {"fields": {}, "fixed": {}}},
                budget={"max_actual_fits": 1, "max_proposals": 1},
                artifact_dir=root / "artifacts" / "validation-job",
                config={
                    "dataset": {"source_path": str(root / "embedding.csv"), "feature_columns": ["x1"]},
                    "task": "dimensionality reduction",
                    "model_id": "D10",
                },
            )
            store.set_job_status(job["job_id"], "completed")

            with self.assertRaisesRegex(BatchError, "objective split='train_exploratory'"):
                export_training_exploration(history, job["job_id"])


if __name__ == "__main__":
    unittest.main()
