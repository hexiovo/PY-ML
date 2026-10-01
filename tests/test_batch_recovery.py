"""Regression coverage for reconstructing a search after worker interruption."""
import tempfile
from pathlib import Path
import unittest

from pyml_workbench.batch import BatchError, _rebuild_resume
from pyml_workbench.history import HistoryStore


class BatchRecoveryTests(unittest.TestCase):
    def _new_job(self, root, *, max_fits=2):
        store = HistoryStore(root / "history.sqlite3")
        job = store.create_job(
            job_id="job-1",
            dataset_id="b" * 64,
            task="classification",
            model_id="C01",
            snapshot={},
            split_summary={},
            search_spec={"method": "grid", "space": {"fields": {}, "fixed": {}}},
            budget={"max_actual_fits": max_fits, "max_proposals": 4},
            artifact_dir=root / "artifacts" / "job-1",
            config={
                "dataset": {
                    "source_path": str(root / "synthetic.csv"),
                    "target_column": "target",
                    "feature_columns": ["x"],
                },
                "task": "classification",
                "model_id": "C01",
                "parameters": {"max_iter": 20},
            },
        )
        store.set_search_fingerprint(job["job_id"], "a" * 64)
        return store, job

    @staticmethod
    def _propose(store, job_id, *, trial_id, index, max_iter):
        digest = str(index) * 64
        store.record_proposal(
            job_id=job_id,
            trial_id=trial_id,
            parameters={"max_iter": max_iter},
            objective_direction="max",
            max_proposals=4,
            config_sha256=digest,
            result={
                "trial_id": trial_id,
                "proposal_index": index,
                "fit_index": None,
                "status": "proposed",
                "parameters": {"max_iter": max_iter},
                "config_sha256": digest,
                "direction": "max",
                "metric": "balanced_accuracy",
                "score_split": "validation",
            },
        )

    def test_running_receipt_restores_consumed_global_fit_index(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store, job = self._new_job(root)
            self._propose(store, job["job_id"], trial_id="trial-1", index=1, max_iter=20)
            store.mark_fit_started(job_id=job["job_id"], trial_id="trial-1", max_actual_fits=2)
            store.finish_trial(
                job_id=job["job_id"], trial_id="trial-1", status="succeeded",
                objective_value=0.5, metrics={"balanced_accuracy": 0.5},
                result={
                    "trial_id": "trial-1", "proposal_index": 1, "fit_index": 1,
                    "status": "complete", "parameters": {"max_iter": 20},
                    "config_sha256": "1" * 64, "direction": "max",
                    "metric": "balanced_accuracy", "score_split": "validation",
                    "objective_value": 0.5, "metrics": {"balanced_accuracy": 0.5},
                },
            )
            self._propose(store, job["job_id"], trial_id="trial-2", index=2, max_iter=30)
            store.mark_fit_started(job_id=job["job_id"], trial_id="trial-2", max_actual_fits=2)

            resume = _rebuild_resume(store, store.get_job(job["job_id"]))

            self.assertEqual(resume.actual_fit_count, 2)
            self.assertEqual(resume.proposal_count, 2)
            self.assertEqual([record.fit_index for record in resume.records], [1, 2])
            self.assertEqual([record.status for record in resume.records], ["complete", "failed"])
            self.assertTrue(all(record.test_evaluation_count == 0 for record in resume.records))
            interrupted = store.get_trial(job["job_id"], "trial-2")
            self.assertEqual(interrupted["actual_fit_count"], 1)
            self.assertEqual(interrupted["status"], "failed")

    def test_multiple_running_receipts_are_rejected_before_recovery_mutates_them(self):
        with tempfile.TemporaryDirectory() as directory:
            store, job = self._new_job(Path(directory))
            self._propose(store, job["job_id"], trial_id="trial-1", index=1, max_iter=20)
            store.mark_fit_started(job_id=job["job_id"], trial_id="trial-1", max_actual_fits=2)
            self._propose(store, job["job_id"], trial_id="trial-2", index=2, max_iter=30)
            store.mark_fit_started(job_id=job["job_id"], trial_id="trial-2", max_actual_fits=2)

            with self.assertRaisesRegex(BatchError, "multiple running trial receipts"):
                _rebuild_resume(store, store.get_job(job["job_id"]))

            self.assertEqual(
                [row["status"] for row in store.list_trials(job["job_id"])],
                ["running", "running"],
            )

    def test_aggregate_fit_counter_must_match_per_trial_receipts(self):
        with tempfile.TemporaryDirectory() as directory:
            store, job = self._new_job(Path(directory))
            self._propose(store, job["job_id"], trial_id="trial-1", index=1, max_iter=20)
            store.mark_fit_started(job_id=job["job_id"], trial_id="trial-1", max_actual_fits=2)
            store.persist_search_result(
                job_id=job["job_id"],
                result={"trials": [], "actual_fit_count": 2, "proposal_count": 1, "elapsed_seconds": 0},
            )

            with self.assertRaisesRegex(BatchError, "aggregate fit counter does not match"):
                _rebuild_resume(store, store.get_job(job["job_id"]))


if __name__ == "__main__":
    unittest.main()
