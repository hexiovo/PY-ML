"""Durable batch jobs train and resume only from their enqueue-owned snapshot."""
from __future__ import annotations

import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd

from pyml_workbench import (
    BatchError,
    DatasetConfig,
    ExperimentConfig,
    ExperimentError,
    SearchSpace,
    SearchSpec,
    SplitConfig,
    create_search_job,
    load_search_result,
    request_job_control,
    run_search_job,
    run_search_jobs,
)
from pyml_workbench.history import HistoryStore
from pyml_workbench.diagnostics import LogSession, read_records
from pyml_workbench.owned_snapshot import load_owned_snapshot


class BatchOwnedSnapshotTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.source = self.root / "data.csv"
        self._write_source(scale=1.0)
        self.config = ExperimentConfig(
            dataset=DatasetConfig(
                source_path=str(self.source),
                target_column="target",
                feature_columns=("x",),
            ),
            task="classification",
            model_id="C01",
            split=SplitConfig(seed=42),
        )
        self.history = self.root / "history.sqlite3"
        self.artifact_root = self.root / "artifacts"

    def _write_source(self, *, scale: float) -> None:
        x = np.linspace(-2, 2, 120) * scale
        pd.DataFrame({"x": x, "target": (np.sin(x) > 0).astype(int)}).to_csv(self.source, index=False)

    @staticmethod
    def _spec(*, candidates=(0.1,), max_fits=1) -> SearchSpec:
        return SearchSpec(
            method="grid",
            space=SearchSpace.from_dict({
                "fields": {"C": {"type": "real", "low": 0.1, "high": 2.0, "values": list(candidates)}},
            }),
            max_fits=max_fits,
            max_proposals=len(candidates),
            timeout_seconds=120,
            seed=42,
        )

    def test_source_change_after_enqueue_fits_and_restores_queued_snapshot(self):
        job = create_search_job(self.history, self.artifact_root, self.config, self._spec())
        queued_manifest = json.loads(job["snapshot_json"])
        receipt_before = json.loads(job["snapshot_receipt_json"])
        self._write_source(scale=25.0)

        result = run_search_job(self.history, job["job_id"])
        self.assertEqual(result.snapshot.to_dict(), queued_manifest)
        self.assertEqual(result.snapshot.manifest["data_sha256"], queued_manifest["data_sha256"])
        self.assertEqual(result.actual_fit_count, 1)
        restored = load_search_result(self.history, job["job_id"])
        self.assertEqual(restored.snapshot.to_dict(), queued_manifest)

        current = HistoryStore(self.history).get_job(job["job_id"])
        receipt_after = json.loads(current["snapshot_receipt_json"])
        self.assertEqual(receipt_after, receipt_before)
        self.assertEqual(
            load_owned_snapshot(
                Path(current["artifact_dir"]) / "snapshot.joblib",
                receipt_after,
                expected_manifest=queued_manifest,
                config=self.config,
            ).to_dict(),
            queued_manifest,
        )

    def test_failed_trial_keeps_error_id_traceback_in_logs_and_sqlite(self):
        job = create_search_job(self.history, self.artifact_root, self.config, self._spec())
        log_root = self.root / "diagnostics"
        logger = LogSession(log_root, session_id="search-errors", role="worker")
        self.addCleanup(logger.close)

        with patch(
            "pyml_workbench.search.prepare_experiment",
            side_effect=RuntimeError("synthetic fit failure"),
        ):
            outcome = run_search_jobs(
                self.history,
                [job["job_id"]],
                max_workers=1,
                error_handler=logger.record_error,
            )[0]

        self.assertEqual(outcome["status"], "completed")
        self.assertEqual(len(outcome["diagnostic_errors"]), 1)
        detail = outcome["diagnostic_errors"][0]
        self.assertTrue(detail["error_id"])
        self.assertIn("Traceback", detail["traceback"])
        self.assertIn("synthetic fit failure", detail["traceback"])

        stored = HistoryStore(self.history).list_trials(job["job_id"])[0]
        result = json.loads(stored["result_json"])
        self.assertEqual(result["error_id"], detail["error_id"])
        self.assertEqual(result["traceback"], detail["traceback"])
        records = read_records(
            log_root,
            session_id="search-errors",
            error_ids=[detail["error_id"]],
        )
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["job_id"], job["job_id"])
        self.assertEqual(records[0]["traceback"], detail["traceback"])

    def test_run_search_jobs_failure_returns_logged_error_details(self):
        job = create_search_job(self.history, self.artifact_root, self.config, self._spec())
        snapshot_path = Path(job["artifact_dir"]) / "snapshot.joblib"
        snapshot = joblib.load(snapshot_path)
        snapshot.features.iloc[0, 0] += 1000
        joblib.dump(snapshot, snapshot_path, compress=3)
        log_root = self.root / "diagnostics"
        logger = LogSession(log_root, session_id="batch-errors", role="worker")
        self.addCleanup(logger.close)

        outcome = run_search_jobs(
            self.history,
            [job["job_id"]],
            error_handler=logger.record_error,
        )[0]

        self.assertEqual(outcome["status"], "failed")
        self.assertTrue(outcome["error_id"])
        self.assertIn("Traceback", outcome["traceback"])
        self.assertIn("verification", outcome["error"])
        self.assertEqual(outcome["diagnostic_errors"][0]["error_id"], outcome["error_id"])
        records = read_records(
            log_root,
            session_id="batch-errors",
            error_ids=[outcome["error_id"]],
        )
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["job_id"], job["job_id"])
        self.assertEqual(records[0]["traceback"], outcome["traceback"])

    def test_pause_delete_source_and_resume_from_owned_snapshot(self):
        job = create_search_job(
            self.history,
            self.artifact_root,
            self.config,
            self._spec(candidates=(0.1, 0.5, 2.0), max_fits=3),
        )
        queued_manifest = json.loads(job["snapshot_json"])
        outcomes = []
        pause_sent = threading.Event()

        def pause_after_first(_job_id, event):
            if event.get("name") == "trial_completed" and not pause_sent.is_set():
                pause_sent.set()
                request_job_control(self.history, job["job_id"], "pause")

        worker = threading.Thread(
            target=lambda: outcomes.extend(run_search_jobs(
                self.history, [job["job_id"]], max_workers=1, on_event=pause_after_first,
            )),
            daemon=True,
        )
        worker.start()
        store = HistoryStore(self.history)
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            current = store.get_job(job["job_id"])
            if current["status"] == "paused":
                break
            if not worker.is_alive():
                self.fail(f"search ended before the pause checkpoint: {outcomes}")
            time.sleep(0.05)
        else:
            self.fail("search did not reach its persisted pause checkpoint")

        paused = store.get_job(job["job_id"])
        self.assertEqual(paused["actual_fit_count"], 1)
        self.source.unlink()
        request_job_control(self.history, job["job_id"], "resume")
        worker.join(timeout=30)

        self.assertFalse(worker.is_alive(), "resumed worker did not finish")
        self.assertEqual(len(outcomes), 1, outcomes)
        self.assertEqual(outcomes[0]["status"], "completed")
        self.assertEqual(outcomes[0]["result"]["actual_fit_count"], 3)
        self.assertEqual(outcomes[0]["result"]["snapshot"], queued_manifest)
        self.assertEqual(load_search_result(self.history, job["job_id"]).snapshot.to_dict(), queued_manifest)

    def test_tampered_owned_snapshot_and_legacy_missing_snapshot_fail_before_fit(self):
        job = create_search_job(self.history, self.artifact_root, self.config, self._spec())
        snapshot_path = Path(job["artifact_dir"]) / "snapshot.joblib"
        snapshot = joblib.load(snapshot_path)
        snapshot.features.iloc[0, 0] += 1000
        joblib.dump(snapshot, snapshot_path, compress=3)
        with patch("pyml_workbench.search.prepare_experiment") as fit:
            with self.assertRaisesRegex(BatchError, "snapshot failed verification|checksum"):
                run_search_job(self.history, job["job_id"])
        fit.assert_not_called()
        failed = HistoryStore(self.history).get_job(job["job_id"])
        self.assertEqual(failed["actual_fit_count"], 0)
        self.assertEqual(failed["status"], "failed")
        self.assertIn("Preflight failed before fit", HistoryStore(self.history).list_events(job["job_id"])[-1]["payload_json"])

        legacy = HistoryStore(self.history)
        # Rebuild a clean snapshot directly from source for a queued legacy receipt.
        from pyml_workbench.experiment import build_snapshot
        legacy_snapshot = build_snapshot(self.config)
        legacy_spec = self._spec()
        legacy_job = legacy.create_job(
            job_id="legacy-pending",
            dataset_id=legacy_snapshot.manifest["source_sha256"],
            task=self.config.task,
            model_id=self.config.model_id,
            config=self.config.to_dict(),
            snapshot=legacy_snapshot.to_dict(),
            split_summary={
                "seed": self.config.split.seed,
                "split_sha256": legacy_snapshot.manifest["split_sha256"],
                "positions": legacy_snapshot.manifest["split_positions"],
            },
            search_spec=legacy_spec.to_dict(),
            budget={"max_actual_fits": 1, "max_proposals": 1},
            artifact_dir=self.artifact_root / "legacy-pending",
        )
        self.source.write_text("changed after old job enqueue", encoding="utf-8")
        with patch("pyml_workbench.search.prepare_experiment") as fit:
            with self.assertRaisesRegex(BatchError, "Owned search snapshot is missing"):
                run_search_job(self.history, legacy_job["job_id"])
        fit.assert_not_called()
        self.assertEqual(legacy.get_job(legacy_job["job_id"])["actual_fit_count"], 0)
        self.assertEqual(legacy.get_job(legacy_job["job_id"])["status"], "failed")

    def test_experiment_error_during_snapshot_preflight_marks_job_failed_without_fit(self):
        job = create_search_job(self.history, self.artifact_root, self.config, self._spec())
        with patch(
            "pyml_workbench.batch._load_owned_job_snapshot",
            side_effect=ExperimentError("snapshot/config identity mismatch"),
        ), patch("pyml_workbench.search.prepare_experiment") as fit:
            with self.assertRaisesRegex(ExperimentError, "identity mismatch"):
                run_search_job(self.history, job["job_id"])
        fit.assert_not_called()
        failed = HistoryStore(self.history).get_job(job["job_id"])
        self.assertEqual(failed["status"], "failed")
        self.assertEqual(failed["actual_fit_count"], 0)
        status_events = [
            json.loads(event["payload_json"])
            for event in HistoryStore(self.history).list_events(job["job_id"])
            if event["event_type"] == "job_status"
        ]
        self.assertIn("Preflight failed before fit", status_events[-1]["reason"])


if __name__ == "__main__":
    unittest.main()
