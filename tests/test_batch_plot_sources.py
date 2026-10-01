"""Owned-cache and live batch-result plotting entry contracts."""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import unittest
from contextlib import ExitStack, contextmanager
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import joblib
import numpy as np
import pandas as pd
from pyml_workbench.config import DatasetConfig, ExperimentConfig
from pyml_workbench.history import HistoryStore
from pyml_workbench.gui import PlotSourceChooserDialog
from pyml_workbench.plot_cache import (
    PlotCacheError,
    load_batch_plot_sources,
    session_plot_cache_manifest_path,
)
from pyml_workbench.batch import (
    BatchError,
    create_search_job,
    finalize_frozen_search,
    freeze_search_winner,
    load_search_result,
    run_search_job,
)
from pyml_workbench.search import SearchSpec
from pyml_workbench.search_space import SearchSpace
from pyml_workbench.gui import WorkbenchWindow
from pyml_workbench.batch_gui import BatchSearchDialog
from PySide6.QtWidgets import QApplication, QDialog


@contextmanager
def _forbid_model_work():
    """Fail if selecting, rendering, or saving a plot starts model work."""
    targets = (
        "pyml_workbench.experiment.prepare_experiment",
        "pyml_workbench.experiment.freeze_experiment",
        "pyml_workbench.experiment.evaluate_test",
        "pyml_workbench.experiment.finalize_experiment",
        "pyml_workbench.experiment.predict",
        "pyml_workbench.experiment.FittedModel._invoke",
        "pyml_workbench.search.search",
        "pyml_workbench.batch.create_search_job",
        "pyml_workbench.batch.run_search_job",
        "pyml_workbench.batch.run_search_jobs",
        "pyml_workbench.batch.freeze_search_winner",
        "pyml_workbench.batch.finalize_frozen_search",
        "pyml_workbench.selection.prepare_experiment",
        "pyml_workbench.selection.freeze_experiment",
        "pyml_workbench.selection.evaluate_test",
        "pyml_workbench.selection.finalize_selected",
        "pyml_workbench.extended_experiment.prepare_extended_experiment",
        "pyml_workbench.extended_experiment.evaluate_extended_test",
        "pyml_workbench.extended_experiment.predict_extended",
    )
    with ExitStack() as stack:
        spies = [
            stack.enter_context(patch(target, side_effect=AssertionError(f"plot called {target}")))
            for target in targets
        ]
        yield
        for spy in spies:
            spy.assert_not_called()


class BatchPlotSourceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.application = QApplication.instance() or QApplication([])

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.history = self.root / "history.sqlite3"
        self.artifacts = self.root / "artifacts"
        self.source = self.root / "data.csv"
        x = np.linspace(-3.0, 3.0, 100)
        pd.DataFrame({"x": x, "target": (np.sin(x) > 0).astype(int)}).to_csv(self.source, index=False)
        self.config = ExperimentConfig(
            dataset=DatasetConfig(str(self.source), target_column="target", feature_columns=("x",)),
            task="classification",
            model_id="C01",
            parameters={"max_iter": 100},
        )
        self.spec = SearchSpec(
            method="grid",
            space=SearchSpace.from_dict({
                "fields": {"C": {"type": "real", "low": 0.1, "high": 1.0, "values": [0.5]}},
            }),
            max_fits=1,
            max_proposals=1,
            timeout_seconds=120,
            seed=42,
        )
        self.job = create_search_job(self.history, self.artifacts, self.config, self.spec)
        run_search_job(self.history, self.job["job_id"])

    def test_only_owned_partition_results_appear_and_valid_test_gate_opens_real_button(self):
        job_id = self.job["job_id"]
        self.source.unlink()
        with self.assertRaisesRegex(PlotCacheError, "移除"):
            load_batch_plot_sources(self.history, f"{job_id}-not-owned")

        original_get_job = HistoryStore.get_job

        def reject_job_record(mutate, message):
            calls = 0

            def get_job(store, selected_job_id):
                nonlocal calls
                value = original_get_job(store, selected_job_id)
                if selected_job_id == job_id and calls == 0:
                    calls += 1
                    mutate(value)
                return value

            with patch.object(HistoryStore, "get_job", get_job):
                with self.assertRaisesRegex(PlotCacheError, message):
                    load_batch_plot_sources(self.history, job_id)
            self.assertEqual(calls, 1)

        reject_job_record(lambda value: value.update(status="running"), "已完成")
        reject_job_record(lambda value: value.update(snapshot_sha256="0" * 64), "snapshot_sha256")

        def tamper_receipt_hash(value):
            receipt = json.loads(value["snapshot_receipt_json"])
            receipt["manifest_sha256"] = "0" * 64
            value["snapshot_receipt_json"] = json.dumps(receipt)

        reject_job_record(tamper_receipt_hash, "manifest 哈希")

        winner_for_manifest = load_search_result(self.history, job_id)
        winner_session_path = Path(winner_for_manifest.winner.session_path)
        winner_manifest_path = session_plot_cache_manifest_path(winner_session_path)
        self.assertTrue(winner_manifest_path.is_file())
        trusted_winner_session = winner_session_path.read_bytes()
        trusted_winner_manifest = winner_manifest_path.read_bytes()
        try:
            winner_manifest_path.unlink()
            with self.assertRaisesRegex(PlotCacheError, "winner session 缺少绘图完整性 manifest"):
                load_batch_plot_sources(self.history, job_id)
        finally:
            winner_manifest_path.write_bytes(trusted_winner_manifest)

        try:
            tampered_winner = joblib.load(winner_session_path)
            tampered_winner.rows[0]["y_pred"] += 1
            joblib.dump(tampered_winner, winner_session_path)
            with self.assertRaisesRegex(PlotCacheError, "prediction/result rows"):
                load_batch_plot_sources(self.history, job_id)
        finally:
            winner_session_path.write_bytes(trusted_winner_session)
            winner_manifest_path.write_bytes(trusted_winner_manifest)

        snapshot_path = Path(self.job["artifact_dir"]) / "snapshot.joblib"
        snapshot_bytes = snapshot_path.read_bytes()
        try:
            snapshot_path.write_bytes(snapshot_bytes + b"x")
            with self.assertRaises(BatchError):
                load_batch_plot_sources(self.history, job_id)
        finally:
            snapshot_path.write_bytes(snapshot_bytes)

        sources = load_batch_plot_sources(self.history, job_id)
        self.assertEqual(set(sources.by_partition()), {"train", "validation"})
        self.assertIn("test", sources.unavailable_by_partition())
        winner = load_search_result(self.history, job_id).winner
        validation = sources.by_partition()["validation"]
        self.assertEqual(validation.owner_kind, "batch_job")
        self.assertEqual(validation.owner_id, job_id)
        self.assertEqual(validation.partition_count, len(validation.row_positions))
        receipt = json.loads(self.job["snapshot_receipt_json"])
        self.assertEqual(validation.provenance.source_sha256, self.job["dataset_id"])
        self.assertEqual(validation.provenance.data_sha256, receipt["manifest"]["data_sha256"])
        self.assertEqual(validation.provenance.split_sha256, receipt["manifest"]["split_sha256"])
        self.assertEqual(validation.provenance.frozen_config_sha256, winner.config_sha256)
        self.assertEqual(validation.provenance.batch_snapshot_sha256, self.job["snapshot_sha256"])
        self.assertEqual(validation.provenance.receipt_manifest_sha256, receipt["manifest_sha256"])
        self.assertEqual(validation.provenance.receipt_file_sha256, receipt["file_sha256"])

        freeze_search_winner(self.history, job_id)
        finalize_frozen_search(self.history, job_id)
        completed = load_batch_plot_sources(self.history, job_id)
        self.assertEqual(set(completed.by_partition()), {"train", "validation", "test"})
        test_source = completed.by_partition()["test"]
        self.assertEqual(test_source.row_positions, tuple(
            load_search_result(self.history, job_id).snapshot.splits["test"].tolist()
        ))
        self.assertEqual(
            {column.column_id for column in test_source.outputs},
            {"classification.y_true", "classification.y_pred"},
        )

        original_get_permission = HistoryStore.get_test_permission

        def corrupted_permission(store, selected_job_id):
            value = original_get_permission(store, selected_job_id)
            payload = json.loads(value["result_json"])
            payload["test_evaluation_count"] = 0
            value["result_json"] = json.dumps(payload)
            return value

        with patch.object(HistoryStore, "get_test_permission", corrupted_permission):
            rejected = load_batch_plot_sources(self.history, job_id)
        self.assertEqual(set(rejected.by_partition()), {"train", "validation"})
        self.assertIn("回执", rejected.unavailable_by_partition()["test"])

        def rejected_permission_variant(mutate, message=None):
            def corrupted_permission_variant(store, selected_job_id):
                value = original_get_permission(store, selected_job_id)
                mutate(value)
                return value

            with patch.object(HistoryStore, "get_test_permission", corrupted_permission_variant):
                result = load_batch_plot_sources(self.history, job_id)
            self.assertEqual(set(result.by_partition()), {"train", "validation"})
            unavailable = result.unavailable_by_partition()["test"]
            self.assertIn("test", result.unavailable_by_partition())
            if message is not None:
                self.assertIn(message, unavailable)

        invalid_counts = (True, 1.0, 0, 2, -1, "1")

        def mutate_permission_count(count):
            def mutate(value):
                payload = json.loads(value["result_json"])
                payload["test_evaluation_count"] = count
                value["result_json"] = json.dumps(payload)

            return mutate

        for invalid_count in invalid_counts:
            with self.subTest(permission_count=invalid_count):
                rejected_permission_variant(
                    mutate_permission_count(invalid_count), message="回执"
                )

        rejected_permission_variant(lambda value: value.update(state="consumed"))
        rejected_permission_variant(lambda value: value.update(final_session_id="foreign-final-session"))

        final_path = Path(self.job["artifact_dir"]) / "final-session.joblib"
        trusted_final_cache = final_path.read_bytes()
        final_envelope = joblib.load(final_path)
        self.assertIn("plot_cache_manifest", final_envelope)
        self.assertIn("result_rows", final_envelope["plot_cache_manifest"]["digests"])

        def reject_final_session_variant(mutate, message=None):
            final_path.write_bytes(trusted_final_cache)
            envelope = joblib.load(final_path)
            mutate(envelope)
            joblib.dump(envelope, final_path)
            try:
                result = load_batch_plot_sources(self.history, job_id)
            finally:
                final_path.write_bytes(trusted_final_cache)
            self.assertEqual(set(result.by_partition()), {"train", "validation"})
            unavailable = result.unavailable_by_partition()["test"]
            self.assertIn("test", result.unavailable_by_partition())
            if message is not None:
                self.assertIn(message, unavailable)

        def mutate_final_session_count(field, count):
            def mutate(value):
                session = value["session"]
                if field == "session":
                    session.test_evaluation_count = count
                else:
                    session.result.audit["test_evaluation_count"] = count

            return mutate

        for field in ("session", "audit"):
            for invalid_count in invalid_counts:
                with self.subTest(final_session_count_field=field, count=invalid_count):
                    reject_final_session_variant(
                        mutate_final_session_count(field, invalid_count),
                        message="最终测试结果",
                    )

        reject_final_session_variant(
            lambda value: value.update(session_id="foreign-final-session")
        )
        reject_final_session_variant(
            lambda value: value.update(session_path=str(self.root / "foreign-final-session.joblib"))
        )
        reject_final_session_variant(
            lambda value: value.update(frozen_config_sha256="0" * 64)
        )
        reject_final_session_variant(
            lambda value: value["session"].fitted_model.capabilities.update(predict=False)
        )
        reject_final_session_variant(
            lambda value: value.update(selection_sha256="0" * 64)
        )
        def mutate_result_rows(value):
            result = value["session"].result
            changed = result.results.copy(deep=True)
            changed.loc[0, "y_pred"] = changed.loc[0, "y_pred"] + 1
            result.results = changed

        reject_final_session_variant(mutate_result_rows)
        reject_final_session_variant(lambda value: value.pop("plot_cache_manifest"))

        dialog = BatchSearchDialog()
        self.addCleanup(dialog.close)
        dialog.root_edit.setText(str(self.root))
        dialog.refresh_history()
        self.assertTrue(dialog.plot_result_button.isEnabled())

        def choose_test_chart(chooser):
            test_index = chooser.partition_combo.findData("test")
            self.assertGreaterEqual(test_index, 0)
            chooser.partition_combo.setCurrentIndex(test_index)
            from pyml_workbench.plotting import PlotKind

            confusion_index = next(
                index for index in range(chooser.plot_combo.count())
                if chooser.plot_combo.itemData(index).kind == PlotKind.CLASSIFICATION_CONFUSION
            )
            chooser.plot_combo.setCurrentIndex(confusion_index)
            chooser._accept_selection()
            return QDialog.DialogCode.Accepted

        with patch.object(PlotSourceChooserDialog, "exec", choose_test_chart):
            with _forbid_model_work():
                with patch.object(dialog, "_start_worker") as start_worker:
                    dialog.plot_result_button.click()
                start_worker.assert_not_called()
                plot_window = dialog._plot_dialogs[0]
                for format_name in ("png", "svg"):
                    target = self.root / f"batch.{format_name}"
                    with patch(
                        "pyml_workbench.plot_dialog.QFileDialog.getSaveFileName",
                        return_value=(str(target), ""),
                    ):
                        self.assertTrue(plot_window.save_selected(format_name))
                    self.assertGreater(target.stat().st_size, 0)
        self.assertEqual(len(dialog._plot_dialogs), 1)
        plot_window = dialog._plot_dialogs[0]
        self.assertFalse(plot_window.isModal())
        self.assertEqual(plot_window.payloads[0].partition, "test")
        self.assertEqual(plot_window.payloads[0].spec.kind.value, "classification_confusion")


if __name__ == "__main__":
    unittest.main()
