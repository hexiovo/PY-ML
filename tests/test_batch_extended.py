"""Durable batch integration for sequence plans and neural refit provenance."""
from __future__ import annotations

import json
import hashlib
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from pyml_workbench.batch import (
    create_search_job,
    finalize_frozen_search,
    freeze_search_winner,
    get_job_summary,
    load_search_result,
    request_job_control,
    run_search_jobs,
)
from pyml_workbench.config import DatasetConfig, ExperimentConfig
from pyml_workbench.history import HistoryStore
from pyml_workbench.search import SearchSpec
from pyml_workbench.search_space import SearchSpace
from pyml_workbench.sequence import SequenceConfig, load_owned_sequence_plan


def _frame() -> pd.DataFrame:
    rows = []
    for group in range(15):
        for step in range(12):
            value = float(group * 12 + step)
            rows.append({
                "group": f"g{group:02d}",
                "time": step,
                "value": value,
                "target": float(group * 0.5 + step * 0.25),
            })
    return pd.DataFrame(rows)


class ExtendedBatchTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.source = self.root / "sequence.csv"
        _frame().to_csv(self.source, index=False)
        self.history = self.root / "history.sqlite3"
        self.artifacts = self.root / "artifacts"

    def _hmm_request(self, values=(2,), *, max_fits=1):
        config = ExperimentConfig(
            dataset=DatasetConfig(str(self.source), feature_columns=("value",)),
            task="sequence_modeling",
            model_id="H01",
            parameters={"n_iter": 2},
            sequence=SequenceConfig(
                group_column="group",
                time_column="time",
                order_mode="time",
                observation_columns=("value",),
            ),
        )
        spec = SearchSpec(
            method="grid",
            space=SearchSpace.from_dict({
                "fields": {"n_components": {
                    "type": "choice", "values": list(values),
                }},
            }),
            max_fits=max_fits,
            max_proposals=max_fits,
            timeout_seconds=120,
            seed=42,
        )
        return config, spec

    def test_owned_hmm_plan_survives_pause_source_deletion_resume_and_final_cache(self):
        config, spec = self._hmm_request(values=(2, 3), max_fits=2)
        job = create_search_job(self.history, self.artifacts, config, spec)
        queued = HistoryStore(self.history).get_job(job["job_id"])
        receipt = json.loads(queued["sequence_plan_receipt_json"])
        plan_path = Path(queued["artifact_dir"]) / "sequence_plan.joblib"
        self.assertTrue(plan_path.is_file())
        queued_snapshot = json.loads(queued["snapshot_json"])
        plan_manifest = receipt["manifest"]
        self.assertEqual(plan_manifest["kind"], "pyml_workbench.sequence_plan")
        changed_source = _frame()
        changed_source["value"] += 10000.0
        changed_source.to_csv(self.source, index=False)
        self.assertNotEqual(
            hashlib.sha256(self.source.read_bytes()).hexdigest(),
            queued_snapshot["source_sha256"],
        )

        outcomes = []
        pause_sent = threading.Event()

        def pause_after_first(_job_id, event):
            if event.get("name") == "trial_completed" and not pause_sent.is_set():
                pause_sent.set()
                request_job_control(self.history, job["job_id"], "pause")

        worker = threading.Thread(
            target=lambda: outcomes.extend(run_search_jobs(
                self.history,
                [job["job_id"]],
                max_workers=1,
                on_event=pause_after_first,
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
                self.fail(f"HMM queue completed before reaching the pause checkpoint: {outcomes}")
            time.sleep(0.05)
        else:
            self.fail("HMM queue did not reach its persisted pause checkpoint")

        self.assertEqual(store.get_job(job["job_id"])["actual_fit_count"], 1)
        self.source.unlink()
        request_job_control(self.history, job["job_id"], "resume")
        worker.join(timeout=30)
        self.assertFalse(worker.is_alive(), "resumed HMM job did not finish")
        self.assertEqual(len(outcomes), 1)
        self.assertEqual(outcomes[0]["status"], "completed")
        self.assertEqual(outcomes[0]["result"]["actual_fit_count"], 2)

        restored = load_search_result(self.history, job["job_id"])
        self.assertEqual(restored.snapshot.to_dict(), queued_snapshot)
        self.assertIsNotNone(restored.sequence_plan)
        self.assertEqual(restored.sequence_plan.to_dict(), plan_manifest)
        self.assertEqual(restored.sequence_plan_path, str(plan_path.resolve()))
        self.assertEqual(restored.sequence_plan_receipt, receipt)
        loaded = load_owned_sequence_plan(
            plan_path,
            receipt,
            expected_manifest=plan_manifest,
            config=config,
            snapshot=restored.snapshot,
        )
        self.assertEqual(loaded.to_dict(), plan_manifest)

        selection = freeze_search_winner(self.history, job["job_id"])
        provenance = selection.metadata["extended_provenance"]
        self.assertEqual(provenance["schema"], "pyml_workbench.refit_provenance/1")
        self.assertEqual(provenance["model_id"], "H01")
        self.assertEqual(
            provenance["sequence_plan_manifest_sha256"],
            selection.sequence_plan.plan_sha256,
        )
        final = finalize_frozen_search(self.history, job["job_id"])
        self.assertEqual(final.session.test_evaluation_count, 1)
        cached = finalize_frozen_search(self.history, job["job_id"])
        self.assertTrue(cached.cached)
        self.assertEqual(cached.session.test_evaluation_count, 1)
        self.assertEqual(cached.session.metrics["test"], final.session.metrics["test"])
        self.assertEqual(get_job_summary(self.history, job["job_id"])["test_permission"]["state"], "completed")

    def test_missing_or_tampered_owned_hmm_plan_fails_before_any_fit(self):
        for failure in ("missing", "tampered"):
            with self.subTest(failure=failure):
                config, spec = self._hmm_request(values=(2,), max_fits=1)
                job = create_search_job(self.history, self.artifacts, config, spec)
                plan_path = Path(job["artifact_dir"]) / "sequence_plan.joblib"
                if failure == "missing":
                    plan_path.unlink()
                else:
                    with plan_path.open("ab") as stream:
                        stream.write(b"tampered owned HMM plan")

                with patch("pyml_workbench.search.prepare_experiment") as fit:
                    outcome = run_search_jobs(self.history, [job["job_id"]], max_workers=1)[0]

                fit.assert_not_called()
                self.assertEqual(outcome["status"], "failed")
                self.assertIn("Owned sequence plan failed verification", outcome["error"])
                stored = HistoryStore(self.history).get_job(job["job_id"])
                self.assertEqual(stored["status"], "failed")
                self.assertEqual(stored["actual_fit_count"], 0)

    def test_deep_batch_freezes_selected_epochs_into_the_refit_contract(self):
        config = ExperimentConfig(
            dataset=DatasetConfig(
                str(self.source),
                target_column="target",
                feature_columns=("value",),
            ),
            task="regression",
            model_id="N06",
            parameters={
                "hidden_size": 4,
                "batch_size": 16,
                "max_epochs": 2,
                "patience": 1,
                "random_state": 42,
            },
            sequence=SequenceConfig(
                group_column="group",
                time_column="time",
                order_mode="time",
                window=3,
                horizon=1,
            ),
        )
        spec = SearchSpec(
            method="grid",
            space=SearchSpace(),
            max_fits=1,
            max_proposals=1,
            timeout_seconds=120,
            seed=42,
        )
        job = create_search_job(self.history, self.artifacts, config, spec)
        self.source.unlink()
        outcomes = run_search_jobs(self.history, [job["job_id"]], max_workers=1)
        self.assertEqual(outcomes[0]["status"], "completed")
        restored = load_search_result(self.history, job["job_id"])
        self.assertIsNotNone(restored.sequence_plan)
        winner_curves = restored.winner_session.training_metadata["training_curves"]
        trial_result = json.loads(HistoryStore(self.history).get_trial(job["job_id"], restored.winner.trial_id)["result_json"])
        self.assertEqual(trial_result["training_curves"], winner_curves)
        summary = get_job_summary(self.history, job["job_id"])["sequence_summary"]
        self.assertEqual(summary["totals"]["source_rows"], restored.sequence_plan.row_count)
        self.assertEqual(summary["totals"]["groups"], 15)
        self.assertEqual(summary["totals"]["windows"], sum(len(restored.sequence_plan.partitions[name].window_target_positions) for name in ("train", "validation", "test")))
        self.assertAlmostEqual(sum(summary["splits"][name]["source_rows_proportion"] for name in ("train", "validation", "test")), 1.0)

        selection = freeze_search_winner(self.history, job["job_id"])
        provenance = selection.metadata["extended_provenance"]
        self.assertEqual(provenance["model_id"], "N06")
        self.assertEqual(provenance["selected_epochs"], 2)
        self.assertEqual(provenance["best_epoch"], 2)
        self.assertEqual(
            provenance["sequence_plan_manifest_sha256"],
            selection.sequence_plan.plan_sha256,
        )
        export_dir = self.root / "final-export"
        final = finalize_frozen_search(self.history, job["job_id"], output_dir=export_dir)
        self.assertEqual(final.session.test_evaluation_count, 1)
        self.assertEqual(final.session.extended_training_metadata["selected_epochs"], 2)
        refit_curves = final.session.training_metadata["training_curves"]
        self.assertEqual(refit_curves["fit_scope"], "train_validation")
        self.assertFalse(refit_curves["validation_available"])
        self.assertIsNone(refit_curves["validation_loss"])
        self.assertEqual(final.to_dict()["training_curves"], refit_curves)
        saved_curves = json.loads(Path(final.artifact_paths["training_curves"]).read_text(encoding="utf-8"))
        self.assertEqual(saved_curves, refit_curves)


if __name__ == "__main__":
    unittest.main()
