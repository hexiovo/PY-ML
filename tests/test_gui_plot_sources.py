"""Real Qt-entry contracts for data overview and plot-source adapters."""
from __future__ import annotations

import os
from pathlib import Path
import tempfile
import json
from types import SimpleNamespace
import unittest
from contextlib import ExitStack, contextmanager
from io import StringIO
from unittest.mock import patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import joblib
import numpy as np
import pandas as pd
from pyml_workbench.config import DatasetConfig, ExperimentConfig
from pyml_workbench.data import LoadedDataset
from pyml_workbench.experiment import _canonical_config, evaluate_test, freeze_experiment, prepare_experiment
from pyml_workbench.gui import WorkbenchWindow
from pyml_workbench.plot_cache import (
    PlotCacheError,
    build_session_plot_cache_manifest,
    loaded_dataset_plot_source,
    load_single_session_plot_sources,
)
from pyml_workbench.plotting import PlotKind
from pyml_workbench.plot_dialog import PlotDialog
from pyml_workbench.gui import PlotSourceChooserDialog
from pyml_workbench.sequence import SequenceConfig
from PySide6.QtWidgets import QDialog
from PySide6.QtWidgets import QApplication


@contextmanager
def _forbid_model_work():
    """Fail if plot build/render/export crosses any model lifecycle entrypoint."""
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


class GuiDataOverviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.application = QApplication.instance() or QApplication([])

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        self.window = WorkbenchWindow(
            preferences_path=Path(self.temp_dir.name) / "preferences.json"
        )
        self.addCleanup(self.window.close)

    def test_real_action_shows_full_table_metadata_and_describe_without_mutation(self):
        source = Path(self.temp_dir.name) / "source.csv"
        frame = pd.DataFrame({
            "signal": [1.0, 2.0, None, 4.0, 100.0],
            "label": ["a", "b", "a", None, "c"],
        }, index=[8, 8, 2, 4, 0])
        original = frame.copy(deep=True)
        dataset = LoadedDataset(str(source), None, (), frame)
        self.assertTrue(self.window.load_data(source, prepared_dataset=dataset))
        self.assertTrue(self.window.data_overview_action.isEnabled())

        with patch.object(self.window, "_start_worker") as start_worker:
            self.window.data_overview_action.trigger()

        dialog = self.window._data_overview_dialog
        self.assertIsNotNone(dialog)
        self.assertTrue(dialog.isVisible())
        self.assertEqual(dialog.summary_label.text(), "完整数据：5 行 × 2 列")
        metadata = dialog.column_table.model()._frame
        self.assertEqual(metadata.to_dict("records"), [
            {"列": "signal", "类型": str(frame["signal"].dtype), "缺失数": 1},
            {"列": "label", "类型": str(frame["label"].dtype), "缺失数": 1},
        ])
        expected_description = frame.describe(include="all").transpose()
        expected_description.index.name = "列"
        expected_description = expected_description.reset_index()
        pd.testing.assert_frame_equal(dialog.describe_frame, expected_description)
        self.assertIs(self.window.dataset.frame, frame)
        pd.testing.assert_frame_equal(frame, original)
        start_worker.assert_not_called()

    def test_data_overview_action_is_disabled_until_a_dataset_is_loaded(self):
        self.assertFalse(self.window.data_overview_action.isEnabled())
        self.assertFalse(self.window.basic_plot_action.isEnabled())
        self.assertIsNone(self.window.show_data_overview())
        self.assertIn("请先加载数据", self.window.result_status_label.text())

    def test_loaded_eda_source_keeps_typed_labels_and_rows(self):
        frame = pd.DataFrame([[1, "1"], [2, "2"]], columns=[1, "1"], index=[9, 9])
        dataset = LoadedDataset("memory.csv", None, (), frame)
        source = loaded_dataset_plot_source(dataset)
        self.assertEqual(source.source_kind, "loaded_eda")
        self.assertEqual(source.owner_kind, "loaded_dataset")
        self.assertEqual(source.partition, "all")
        self.assertEqual(source.row_positions, (0, 1))
        self.assertEqual(source.partition_count, 2)
        self.assertIs(type(source.columns[0].label), int)
        self.assertIs(type(source.columns[1].label), str)
        self.assertEqual(source.columns[0].values, (1, 2))
        self.assertEqual(source.columns[1].values, ("1", "2"))
        self.assertEqual(len(source.provenance.data_sha256), 64)

    def test_real_basic_plot_action_opens_non_modal_plot_after_source_and_column_selection(self):
        frame = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [3.0, 2.0, 1.0]})
        source = Path(self.temp_dir.name) / "eda.csv"
        dataset = LoadedDataset(str(source), None, (), frame)
        self.assertTrue(self.window.load_data(source, prepared_dataset=dataset))

        def choose_histogram(chooser):
            self.assertEqual(chooser.source_combo.count(), 1)
            self.assertEqual(chooser.partition_combo.currentData(), "all")
            first_column = chooser.column_list.item(0)
            self.assertEqual(first_column.data(256)[1], "data.0")
            histogram_index = next(
                index for index in range(chooser.plot_combo.count())
                if chooser.plot_combo.itemData(index).kind == PlotKind.HISTOGRAM
            )
            chooser.plot_combo.setCurrentIndex(histogram_index)
            chooser._accept_selection()
            return QDialog.DialogCode.Accepted

        with patch.object(PlotSourceChooserDialog, "exec", choose_histogram):
            with _forbid_model_work():
                with patch.object(self.window, "_start_worker") as start_worker:
                    self.window.basic_plot_action.trigger()
                start_worker.assert_not_called()
                plot_window = self.window._plot_dialogs[-1]
                for format_name in ("png", "svg"):
                    target = Path(self.temp_dir.name) / f"eda.{format_name}"
                    with patch(
                        "pyml_workbench.plot_dialog.QFileDialog.getSaveFileName",
                        return_value=(str(target), ""),
                    ):
                        self.assertTrue(plot_window.save_selected(format_name))
                    self.assertGreater(target.stat().st_size, 0)

        self.assertEqual(len(self.window._plot_dialogs), 1)
        plot_window = self.window._plot_dialogs[0]
        self.assertIsInstance(plot_window, PlotDialog)
        self.assertFalse(plot_window.isModal())
        self.assertEqual(plot_window.payloads[0].source_kind, "loaded_eda")
        self.assertEqual(plot_window.payloads[0].partition, "all")
        self.assertEqual(plot_window.payloads[0].spec.kind, PlotKind.HISTOGRAM)

    def test_export_captures_owned_result_plot_cache_before_session_cleanup(self):
        source_path = Path(self.temp_dir.name) / "classification.csv"
        x = np.arange(80, dtype=float)
        pd.DataFrame({"x": x, "target": (x % 2).astype(int)}).to_csv(source_path, index=False)
        config = ExperimentConfig(
            dataset=DatasetConfig(str(source_path), target_column="target", feature_columns=("x",)),
            task="classification", model_id="C01", parameters={"max_iter": 100},
        )
        session = prepare_experiment(config)
        freeze_experiment(session, config)
        evaluate_test(session, export_artifacts=False)
        _frozen, digest = _canonical_config(config)
        session_id = "gui-owned-session"
        session_path = Path(self.temp_dir.name) / "gui-owned-session.joblib"
        joblib.dump({
            "kind": "pyml-workbench-internal-session", "schema_version": 1,
            "session_id": session_id, "session_path": str(session_path.resolve()),
            "frozen_config_sha256": digest, "state": "tested", "session": session,
            "plot_cache_manifest": build_session_plot_cache_manifest(session),
        }, session_path)
        self.window._prepared = True
        self.window._prepared_signature = digest
        self.window._run_signature = digest
        self.window._session_id = session_id
        self.window._session_file_path = session_path
        self.window._test_result_payload = {
            "frozen_config_sha256": digest,
            "metrics": session.metrics,
            "audit": session.result.audit,
        }

        self.window._apply_export_result({
            "frozen_config_sha256": digest,
            "capabilities": session.fitted_model.capabilities,
            "artifact_paths": {},
        })
        self.assertFalse(session_path.exists())
        self.assertEqual(set(self.window._session_plot_sources), {"train", "validation", "test"})
        self.assertEqual(self.window._session_plot_sources["test"].owner_id, session_id)
        self.assertTrue(self.window.result_plot_action.isEnabled())

        def choose_test_confusion(chooser):
            test_index = chooser.partition_combo.findData("test")
            self.assertGreaterEqual(test_index, 0)
            chooser.partition_combo.setCurrentIndex(test_index)
            confusion_index = next(
                index for index in range(chooser.plot_combo.count())
                if chooser.plot_combo.itemData(index).kind == PlotKind.CLASSIFICATION_CONFUSION
            )
            chooser.plot_combo.setCurrentIndex(confusion_index)
            chooser._accept_selection()
            return QDialog.DialogCode.Accepted

        source_path.unlink()
        with patch.object(PlotSourceChooserDialog, "exec", choose_test_confusion):
            with _forbid_model_work():
                with patch.object(self.window, "_start_worker") as start_worker:
                    self.window.result_plot_action.trigger()
                start_worker.assert_not_called()
                plot_window = self.window._plot_dialogs[-1]
                for format_name in ("png", "svg"):
                    target = Path(self.temp_dir.name) / f"result.{format_name}"
                    with patch(
                        "pyml_workbench.plot_dialog.QFileDialog.getSaveFileName",
                        return_value=(str(target), ""),
                    ):
                        self.assertTrue(plot_window.save_selected(format_name))
                    self.assertGreater(target.stat().st_size, 0)
        plot_window = self.window._plot_dialogs[-1]
        self.assertEqual(plot_window.payloads[0].partition, "test")
        self.assertEqual(plot_window.payloads[0].spec.kind, PlotKind.CLASSIFICATION_CONFUSION)

    def test_single_session_sources_bind_partitions_to_owned_cached_rows(self):
        source_path = Path(self.temp_dir.name) / "classification.csv"
        x = np.arange(80, dtype=float)
        frame = pd.DataFrame({"x": x, "target": (x % 2).astype(int)})
        frame.to_csv(source_path, index=False)
        config = ExperimentConfig(
            dataset=DatasetConfig(str(source_path), target_column="target", feature_columns=("x",)),
            task="classification",
            model_id="C01",
            parameters={"max_iter": 100},
        )
        session = prepare_experiment(config)
        freeze_experiment(session, config)
        evaluate_test(session, export_artifacts=False)
        _frozen, digest = _canonical_config(config)
        session_id = "owned-session"
        session_path = Path(self.temp_dir.name) / "owned-session.joblib"
        joblib.dump({
            "kind": "pyml-workbench-internal-session",
            "schema_version": 1,
            "session_id": session_id,
            "session_path": str(session_path.resolve()),
            "frozen_config_sha256": digest,
            "state": "tested",
            "session": session,
            "plot_cache_manifest": build_session_plot_cache_manifest(session),
        }, session_path)

        sources = load_single_session_plot_sources(
            session_path,
            session_id=session_id,
            frozen_config_sha256=digest,
        )
        by_partition = sources.by_partition()
        self.assertEqual(set(by_partition), {"train", "validation", "test"})
        test_source = by_partition["test"]
        self.assertEqual(test_source.owner_id, session_id)
        self.assertEqual(test_source.row_positions, tuple(session.splits["test"].tolist()))
        self.assertEqual(test_source.partition_count, len(session.splits["test"]))
        self.assertEqual(
            {column.column_id for column in test_source.outputs},
            {"classification.y_true", "classification.y_pred"},
        )
        self.assertEqual(test_source.provenance.frozen_config_sha256, digest)
        self.assertEqual(len(test_source.provenance.cache_payload_sha256), 64)

    def test_worker_saved_session_manifest_rejects_result_row_tampering(self):
        from pyml_workbench._worker import _load, _session_action

        source_path = Path(self.temp_dir.name) / "worker-owned.csv"
        x = np.arange(80, dtype=float)
        pd.DataFrame({"x": x, "target": (x % 2).astype(int)}).to_csv(source_path, index=False)
        config = ExperimentConfig(
            dataset=DatasetConfig(str(source_path), target_column="target", feature_columns=("x",)),
            task="classification", model_id="C01", parameters={"max_iter": 100},
        )
        config_path = Path(self.temp_dir.name) / "worker-config.json"
        frozen_config, digest = _canonical_config(config)
        config_path.write_text(
            json.dumps(frozen_config, ensure_ascii=False), encoding="utf-8"
        )
        session_id = "worker-owned-session"
        session_path = Path(self.temp_dir.name) / "worker-owned-session.joblib"
        args = SimpleNamespace(
            action="train",
            config=config_path,
            session=session_path,
            session_id=session_id,
        )
        context = {"session_id": session_id, "_stdout": StringIO()}
        for action in ("train", "freeze", "test"):
            args.action = action
            _session_action(args, context)

        saved = joblib.load(session_path)
        self.assertEqual(saved["state"], "tested")
        self.assertIn("result_rows", saved["plot_cache_manifest"]["digests"])
        sources = load_single_session_plot_sources(
            session_path, session_id=session_id, frozen_config_sha256=digest
        )
        self.assertEqual(set(sources.by_partition()), {"train", "validation", "test"})

        trusted_cache = session_path.read_bytes()
        legacy_cache = joblib.load(session_path)
        legacy_cache.pop("plot_cache_manifest")
        joblib.dump(legacy_cache, session_path)
        self.assertIsNotNone(_load(session_path, session_id, config)["session"])
        with self.assertRaisesRegex(PlotCacheError, "缺少绘图完整性 manifest"):
            load_single_session_plot_sources(
                session_path, session_id=session_id, frozen_config_sha256=digest
            )
        session_path.write_bytes(trusted_cache)

        saved["session"].result.results.loc[0, "y_pred"] += 1
        joblib.dump(saved, session_path)
        with self.assertRaisesRegex(PlotCacheError, "prediction/result rows"):
            load_single_session_plot_sources(
                session_path, session_id=session_id, frozen_config_sha256=digest
            )

    def test_n04_n06_owned_cache_preserves_target_window_group_and_utc_order_mapping(self):
        source_path = Path(self.temp_dir.name) / "windowed.csv"
        rows = []
        out_of_order_steps = (4, 0, 7, 3, 1, 8, 2, 5, 6, 9)
        for group in range(8):
            start = pd.Timestamp("2026-01-01", tz="UTC") + pd.Timedelta(days=group * 30)
            for step in out_of_order_steps:
                rows.append({
                    "group": f"g{group}",
                    "time": start + pd.Timedelta(hours=step * 7),
                    "value": float(group * 10 + step),
                    "target": float(group + step / 4),
                })
        frame = pd.DataFrame(rows)
        frame.to_csv(source_path, index=False)
        parsed_times = pd.to_datetime(frame["time"].map(str), utc=True)
        time_unit = parsed_times.dtype.unit
        expected_order_values = parsed_times.astype("int64").to_numpy(copy=True)

        for model_id in ("N04", "N06"):
            with self.subTest(model_id=model_id):
                config = ExperimentConfig(
                    dataset=DatasetConfig(
                        str(source_path), target_column="target", feature_columns=("value",)
                    ),
                    task="regression",
                    model_id=model_id,
                    parameters={
                        "hidden_size": 4,
                        "batch_size": 8,
                        "max_epochs": 1,
                        "patience": 1,
                        "random_state": 4,
                    },
                    sequence=SequenceConfig(
                        group_column="group",
                        time_column="time",
                        order_mode="time",
                        window=3,
                        horizon=2,
                    ),
                )
                session = prepare_experiment(config)
                freeze_experiment(session, config)
                evaluate_test(session, export_artifacts=False)
                _frozen, digest = _canonical_config(config)
                session_id = f"{model_id}-window-session"
                session_path = Path(self.temp_dir.name) / f"{session_id}.joblib"
                joblib.dump({
                    "kind": "pyml-workbench-internal-session",
                    "schema_version": 1,
                    "session_id": session_id,
                    "session_path": str(session_path.resolve()),
                    "frozen_config_sha256": digest,
                    "state": "tested",
                    "session": session,
                    "plot_cache_manifest": build_session_plot_cache_manifest(session),
                }, session_path)

                collection = load_single_session_plot_sources(
                    session_path, session_id=session_id, frozen_config_sha256=digest
                )
                by_partition = collection.by_partition()
                self.assertEqual(set(by_partition), {"train", "validation", "test"})
                plan = session.sequence_plan
                self.assertIsNotNone(plan)
                expected_outputs = {"regression.y_true", "regression.y_pred"}
                result_rows = session.result.results
                for partition, source in by_partition.items():
                    target_column = next(
                        column for column in source.columns if column.label == "target"
                    )
                    selected = result_rows.loc[result_rows["split"] == partition]
                    records = selected.to_dict("records")
                    self.assertEqual(source.row_positions, tuple(int(row["row_position"]) for row in records))
                    self.assertEqual(source.partition_count, len(records))
                    self.assertEqual({column.column_id for column in source.outputs}, expected_outputs)
                    self.assertEqual(
                        source.provenance.sequence_plan_sha256, plan.plan_sha256
                    )
                    self.assertEqual(len(source.provenance.cache_payload_sha256), 64)
                    source_by_target = {item.target_row_position: item for item in source.sequence_map}
                    self.assertEqual(set(source_by_target), set(source.row_positions))
                    for row in records:
                        target_position = int(row["sequence_target_row_position"])
                        mapping = source_by_target[target_position]
                        expected_sources = tuple(int(value) for value in row["sequence_source_row_positions"])
                        expected_group = frame.iloc[target_position]["group"]
                        expected_time_order = int(expected_order_values[target_position])
                        expected_time = parsed_times.iloc[target_position]
                        self.assertEqual(mapping.target_row_position, target_position)
                        self.assertEqual(mapping.source_row_positions, expected_sources)
                        self.assertEqual(mapping.group_value, expected_group)
                        self.assertEqual(mapping.order_value_ns, expected_time_order)
                        self.assertEqual(mapping.order_value_ns, int(plan._order_values_ns[target_position]))
                        self.assertEqual(
                            pd.Timestamp(mapping.order_value_ns, unit=time_unit, tz="UTC"),
                            expected_time,
                        )
                        self.assertIsNone(mapping.time_value)
                        self.assertEqual(
                            session.target.iloc[target_position],
                            target_column.values[source.row_positions.index(target_position)],
                        )
                        for source_position in expected_sources:
                            self.assertEqual(frame.iloc[source_position]["group"], expected_group)
                            self.assertLess(
                                int(expected_order_values[source_position]),
                                expected_time_order,
                            )

                tampered = joblib.load(session_path)
                plan_to_tamper = tampered["session"].sequence_plan
                order_values = plan_to_tamper._order_values_ns.copy()
                order_values[0] += 1
                object.__setattr__(plan_to_tamper, "_order_values_ns", order_values)
                joblib.dump(tampered, session_path)
                with self.assertRaisesRegex(PlotCacheError, "sequence mapping"):
                    load_single_session_plot_sources(
                        session_path,
                        session_id=session_id,
                        frozen_config_sha256=digest,
                    )

    def test_single_session_rejects_wrong_owner_and_locks_failed_test_partition(self):
        source_path = Path(self.temp_dir.name) / "classification.csv"
        x = np.arange(80, dtype=float)
        pd.DataFrame({"x": x, "target": (x % 2).astype(int)}).to_csv(source_path, index=False)
        config = ExperimentConfig(
            dataset=DatasetConfig(str(source_path), target_column="target", feature_columns=("x",)),
            task="classification", model_id="C01", parameters={"max_iter": 100},
        )
        session = prepare_experiment(config)
        freeze_experiment(session, config)
        evaluate_test(session, export_artifacts=False)
        _frozen, digest = _canonical_config(config)
        session_id = "owned-session"
        session_path = Path(self.temp_dir.name) / "owned-session.joblib"
        envelope = {
            "kind": "pyml-workbench-internal-session", "schema_version": 1,
            "session_id": session_id, "session_path": str(session_path.resolve()),
            "frozen_config_sha256": digest, "state": "tested", "session": session,
            "plot_cache_manifest": build_session_plot_cache_manifest(session),
        }
        joblib.dump(envelope, session_path)
        with self.assertRaisesRegex(PlotCacheError, "身份或路径"):
            load_single_session_plot_sources(
                session_path, session_id="different-session", frozen_config_sha256=digest
            )

        trusted_cache = session_path.read_bytes()

        legacy_cache = joblib.load(session_path)
        legacy_cache.pop("plot_cache_manifest")
        joblib.dump(legacy_cache, session_path)
        with self.assertRaisesRegex(PlotCacheError, "缺少绘图完整性 manifest"):
            load_single_session_plot_sources(
                session_path, session_id=session_id, frozen_config_sha256=digest
            )
        session_path.write_bytes(trusted_cache)

        def reject_mutated_session(mutate, message):
            session_path.write_bytes(trusted_cache)
            variant = joblib.load(session_path)
            mutate(variant)
            joblib.dump(variant, session_path)
            try:
                with self.assertRaisesRegex(PlotCacheError, message):
                    load_single_session_plot_sources(
                        session_path, session_id=session_id, frozen_config_sha256=digest
                    )
            finally:
                session_path.write_bytes(trusted_cache)

        def change_data(value):
            features = value["session"].features.copy(deep=True)
            features.iloc[:, 0] = features.iloc[:, 0] + 0.25
            value["session"].features = features

        reject_mutated_session(
            lambda value: value["session"].config.parameters.update(max_iter=101),
            "冻结配置",
        )
        reject_mutated_session(change_data, "数据哈希")

        def change_split(value):
            splits = dict(value["session"].splits)
            train = np.asarray(splits["train"], dtype=np.int64).copy()
            train[0] = int(splits["validation"][0])
            splits["train"] = train
            value["session"].splits = splits

        reject_mutated_session(change_split, "split 哈希")
        def change_prediction_rows(value):
            result = value["session"].result
            changed = result.results.copy(deep=True)
            changed.loc[0, "y_pred"] = changed.loc[0, "y_pred"] + 1
            result.results = changed

        reject_mutated_session(change_prediction_rows, "prediction/result rows")
        reject_mutated_session(
            lambda value: value.update(frozen_config_sha256="0" * 64),
            "配置哈希",
        )
        session_path.write_bytes(trusted_cache)

        invalid_counts = (True, 1.0, 0, 2, -1, "1")
        for field in ("session", "audit"):
            for invalid_count in invalid_counts:
                with self.subTest(count_field=field, count=invalid_count):
                    variant = joblib.load(session_path)
                    if field == "session":
                        variant["session"].test_evaluation_count = invalid_count
                    else:
                        variant["session"].result.audit["test_evaluation_count"] = invalid_count
                    joblib.dump(variant, session_path)
                    try:
                        rejected_count = load_single_session_plot_sources(
                            session_path,
                            session_id=session_id,
                            frozen_config_sha256=digest,
                        )
                    finally:
                        session_path.write_bytes(trusted_cache)
                    self.assertEqual(
                        set(rejected_count.by_partition()), {"train", "validation"}
                    )
                    self.assertIn(
                        "一次性计数",
                        rejected_count.unavailable_by_partition()["test"],
                    )

        def remove_predict_capability(value):
            value["session"].fitted_model.capabilities["predict"] = False

        session_path.write_bytes(trusted_cache)
        variant = joblib.load(session_path)
        remove_predict_capability(variant)
        joblib.dump(variant, session_path)
        try:
            capability_sources = load_single_session_plot_sources(
                session_path, session_id=session_id, frozen_config_sha256=digest
            )
        finally:
            session_path.write_bytes(trusted_cache)
        self.assertEqual(set(capability_sources.by_partition()), {"train", "validation"})
        self.assertIn("模型能力", capability_sources.unavailable_by_partition()["test"])

        sources = load_single_session_plot_sources(
            session_path, session_id=session_id, frozen_config_sha256=digest
        )
        self.assertEqual(set(sources.by_partition()), {"train", "validation", "test"})
        self.assertNotIn("test", sources.unavailable_by_partition())

    def test_dimensionality_reduction_without_transform_hides_test_plot_partition(self):
        source_path = Path(self.temp_dir.name) / "dimensionality-reduction.csv"
        values = np.linspace(-4.0, 4.0, 80)
        pd.DataFrame({"x": values, "y": np.sin(values)}).to_csv(source_path, index=False)
        config = ExperimentConfig(
            dataset=DatasetConfig(
                str(source_path), feature_columns=("x", "y"),
            ),
            task="dimensionality reduction",
            model_id="D01",
        )
        session = prepare_experiment(config)
        freeze_experiment(session, config)
        evaluate_test(session, export_artifacts=False)
        self.assertEqual(session.test_evaluation_count, 1)
        self.assertIn("test", session.result.results["split"].tolist())
        self.assertTrue(session.fitted_model.capabilities["transform"])

        # Simulate a cache whose result rows contain a test projection but whose
        # persisted estimator no longer advertises the required transform API.
        session.fitted_model.capabilities["transform"] = False
        _frozen, digest = _canonical_config(config)
        session_id = "dimensionality-reduction-session"
        session_path = Path(self.temp_dir.name) / "dimensionality-reduction-session.joblib"
        joblib.dump({
            "kind": "pyml-workbench-internal-session",
            "schema_version": 1,
            "session_id": session_id,
            "session_path": str(session_path.resolve()),
            "frozen_config_sha256": digest,
            "state": "tested",
            "session": session,
            "plot_cache_manifest": build_session_plot_cache_manifest(session),
        }, session_path)

        sources = load_single_session_plot_sources(
            session_path,
            session_id=session_id,
            frozen_config_sha256=digest,
        )
        self.assertEqual(set(sources.by_partition()), {"train", "validation"})
        self.assertIn("模型能力", sources.unavailable_by_partition()["test"])


if __name__ == "__main__":
    unittest.main()
