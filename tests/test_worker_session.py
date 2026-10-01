"""Durable worker lifecycle contracts using only temporary synthetic tables."""
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd

from pyml_workbench import _worker as worker
from pyml_workbench import experiment as api


class WorkerSessionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        log_root = self.root / "diagnostics"
        log_root.mkdir()
        environment = patch.dict(os.environ, {"PYML_LOG_ROOT": str(log_root)})
        environment.start()
        self.addCleanup(environment.stop)
        self.source = self.root / "synthetic.csv"
        self.frame = pd.DataFrame({"x": np.arange(80), "category": ["甲", "乙"] * 40, "target": np.arange(80) % 2})
        self.frame.to_csv(self.source, index=False, encoding="utf-8-sig")
        self.config = self.root / "config.json"
        self.payload = {
            "dataset": {"source_path": str(self.source), "target_column": "target", "feature_columns": ["x", "category"]},
            "task": "classification", "model_id": "C01", "parameters": {"max_iter": 100},
            "output_dir": str(self.root / "not_automatic_export"),
        }
        self.config.write_text(json.dumps(self.payload, ensure_ascii=False), encoding="utf-8")
        self.session = self.root / "internal.joblib"
        self.session_id = None

    def call(self, action, *extra, success=True):
        argv = [action]
        if action in {"train", "freeze", "test", "export"}:
            argv += ["--config", str(self.config), "--session", str(self.session)]
            if action != "train":
                argv += ["--session-id", self.session_id]
        argv += list(extra)
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = worker.main(argv)
        events = [json.loads(line) for line in out.getvalue().splitlines()]
        self.assertEqual(code, 0 if success else 1, err.getvalue())
        self.assertTrue(events)
        for event in events:
            self.assertEqual(event["action"], action)
            self.assertIn("operation", event)
            self.assertIn("session_id", event)
            self.assertIn(event["type"], {"progress", "result", "error"})
        self.assertEqual(events[-1]["type"], "result" if success else "error")
        if success and action == "train":
            self.session_id = events[-1]["session_id"]
        if not success:
            self.assertIn("Traceback", err.getvalue())
        return events[-1], events

    def train(self):
        return self.call("train")[0]

    def test_complete_lifecycle_no_refit_cached_test_export_and_inference(self):
        trained = self.train()
        original = joblib.load(self.session)["session"]
        original_model_hash = joblib.hash(original.fitted_model)
        self.assertEqual(trained["state"], "trained")
        self.assertNotIn("test", trained["metrics"])
        self.assertEqual(trained["test_evaluation_count"], 0)
        self.assertIsNone(trained["audit"])
        self.assertFalse(Path(self.payload["output_dir"]).exists())
        self.assertNotIn("test", {row["split"] for row in original.rows})
        self.call("test", success=False)
        with patch.object(worker, "prepare_experiment", side_effect=AssertionError("must not retrain")):
            frozen, _ = self.call("freeze")
            self.assertEqual(frozen["state"], "frozen")
            tested, _ = self.call("test")
        self.assertFalse(tested["cached"])
        self.assertEqual(tested["test_evaluation_count"], 1)
        self.assertEqual(tested["audit"]["split_counts"], {"train": 48, "validation": 16, "test": 16})
        self.assertEqual(tested["audit"]["preprocessor_fit_positions"], tested["audit"]["split_positions"]["train"])
        self.assertLess(tested["events"].index("user_selection_frozen"), tested["events"].index("test_metrics_computed_once"))
        with patch.object(worker, "evaluate_test", side_effect=AssertionError("must use cache")):
            repeated, _ = self.call("test")
            frozen_again, _ = self.call("freeze")
            self.call("test")
        self.assertTrue(repeated["cached"])
        self.assertEqual(frozen_again["state"], "tested")
        self.assertEqual(repeated["metrics"], tested["metrics"])
        self.assertEqual(repeated["events"].count("test_metrics_computed_once"), 1)
        persisted = joblib.load(self.session)["session"]
        self.assertEqual(joblib.hash(persisted.fitted_model), original_model_hash)
        self.assertFalse(Path(self.payload["output_dir"]).exists())
        exported, _ = self.call("export", "--output", str(self.root / "export"))
        for name in ("model", "config", "metrics_csv", "metrics_xlsx", "results_csv", "results_xlsx"):
            self.assertTrue(Path(exported["artifact_paths"][name]).is_file())
        reloaded = api.load_model(exported["artifact_paths"]["model"])
        self.assertIsInstance(reloaded, api.FittedModel)
        self.assertFalse(hasattr(reloaded, "features"))
        self.assertFalse(hasattr(reloaded, "target"))
        self.assertFalse(hasattr(reloaded, "splits"))
        expected = persisted.fitted_model.predict(self.frame)
        np.testing.assert_array_equal(reloaded.predict(self.frame), expected)
        output = self.root / "预测.csv"
        result, _ = self.call("inference", "--operation", "predict", "--model", exported["artifact_paths"]["model"], "--data", str(self.source), "--output", str(output))
        self.assertEqual(result["rows"], 80)
        np.testing.assert_array_equal(pd.read_csv(output)["prediction"].to_numpy(), expected)
        self.call("inference", "--operation", "transform", "--model", exported["artifact_paths"]["model"], "--data", str(self.source), "--output", str(self.root / "unsupported.csv"), success=False)
        self.assertFalse((self.root / "unsupported.csv").exists())

    def test_changed_config_wrong_id_wrong_file_and_snapshot_tamper_rejected(self):
        self.train()
        self.payload["parameters"]["C"] = 0.5
        self.config.write_text(json.dumps(self.payload), encoding="utf-8")
        self.call("freeze", success=False)
        self.payload["parameters"].pop("C")
        self.config.write_text(json.dumps(self.payload), encoding="utf-8")
        correct_id = self.session_id
        self.session_id = "another-session"
        self.call("freeze", success=False)
        self.session_id = correct_id
        saved = joblib.load(self.session)
        saved["session"].config.parameters["C"] = 0.2
        joblib.dump(saved, self.session)
        self.call("freeze", success=False)
        joblib.dump({"not": "a session"}, self.session)
        self.call("freeze", success=False)

    def test_moved_session_and_overwrite_rejected(self):
        self.train()
        self.call("train", success=False)
        moved = self.root / "moved.joblib"
        self.session.rename(moved)
        self.session = moved
        self.call("freeze", success=False)

    def test_failure_consumes_final_test_permission(self):
        self.train()
        self.call("freeze")
        with patch.object(worker, "evaluate_test", side_effect=RuntimeError("synthetic test failure")):
            self.call("test", success=False)
        self.assertEqual(joblib.load(self.session)["state"], "test_failed")
        self.call("test", success=False)
        self.call("freeze", success=False)

    def test_interrupted_test_marker_and_busy_lock_rejected(self):
        self.train()
        self.call("freeze")
        saved = joblib.load(self.session)
        saved["state"] = "testing"
        joblib.dump(saved, self.session)
        self.call("test", success=False)
        lock = self.session.with_name(self.session.name + ".lock")
        lock.write_text("synthetic busy worker", encoding="ascii")
        self.call("freeze", success=False)
        self.assertTrue(lock.exists())

    def test_validation_export_keeps_test_untouched(self):
        self.train()
        self.call("export", "--output", str(self.root / "export"), success=False)
        self.call("freeze")
        exported, _ = self.call("export", "--output", str(self.root / "export"))
        metrics = json.loads(Path(exported["artifact_paths"]["metrics_json"]).read_text(encoding="utf-8"))
        self.assertNotIn("test", metrics)
        self.assertEqual(joblib.load(self.session)["state"], "frozen")
        self.assertFalse(list(self.root.glob("internal.joblib.*.tmp")))
        self.assertFalse(self.session.with_name(self.session.name + ".lock").exists())

    def test_public_stage_api_snapshot_guard_and_run_compatibility(self):
        payload = dict(self.payload, output_dir=None)
        session = api.prepare_experiment(payload)
        self.assertNotIn("test", session.metrics)
        with self.assertRaises(api.ExperimentError):
            api.evaluate_test(session)
        api.freeze_experiment(session, payload)
        first = api.evaluate_test(session)
        self.assertIs(api.evaluate_test(session), first)
        api.freeze_experiment(session)
        self.assertEqual(session.test_evaluation_count, 1)
        changed = json.loads(json.dumps(payload))
        changed["parameters"]["C"] = 0.5
        with self.assertRaises(api.ExperimentError):
            api.freeze_experiment(session, changed)
        self.assertIn("test", api.run_experiment(payload).metrics)

    def test_invalid_cli_is_json_error(self):
        result, _ = self.call("invalid", success=False)
        self.assertEqual(result["error_type"], "ValueError")


if __name__ == "__main__":
    unittest.main()
