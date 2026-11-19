from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd
from openpyxl import load_workbook

from pyml_workbench import DatasetConfig, EvaluationConfig, ExperimentConfig, SplitConfig
from pyml_workbench import evaluate_test, freeze_experiment, prepare_experiment
from pyml_workbench.experiment import ArtifactError
from pyml_workbench.plugins import PYEDA_SOURCE, PluginError, PluginSettings, PluginSettingsStore
from pyml_workbench.validation import ordered_positions


ROOT = Path(__file__).resolve().parents[1]
PIPELINE = ROOT / "skills/pyml-study-config/assets/templates/pyeda_ml/pyeda_ml_pipeline.py"
spec = importlib.util.spec_from_file_location("pyeda_ml_pipeline", PIPELINE)
pipeline = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pipeline)


class ExperimentRegressionTests(unittest.TestCase):
    def test_ordering_preserves_integer_precision_for_numeric_and_csv_text(self):
        for values in ([1700000000000000001, 1700000000000000000],
                       ["1700000000000000001", "1700000000000000000"]):
            self.assertEqual(ordered_positions(values, 2).tolist(), [1, 0])

    def test_stacking_keeps_groups_inside_every_outer_fold(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "rows.csv"
            pd.DataFrame({"meta__group": np.repeat(np.arange(20), 5), "x": np.arange(100),
                          "y": 2 * np.arange(100)}).to_csv(source, index=False)
            cfg = ExperimentConfig(
                DatasetConfig(str(source), target_column="y", feature_columns=("x",)),
                "regression", "R29", parameters={"cv": 2, "member_parameters": {"R14": {"n_estimators": 3}}},
                split=SplitConfig(strategy="group", group_column="meta__group"),
                evaluation=EvaluationConfig(strategy="group", folds=3),
            )
            session = prepare_experiment(cfg)
            folds = session.metrics["cross_validation"]["folds"]
            self.assertEqual(len(folds), 3)
            for outer in folds:
                for inner in outer["inner_cv"]["folds"]:
                    self.assertTrue(set(inner["train_groups"]).isdisjoint(inner["validation_groups"]))

    def test_export_preflight_and_failed_write_preserve_one_test_and_literal_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source, output = root / "rows.csv", root / "output"
            pd.DataFrame({"x": range(60), "label": ["normal", "=1+1"] * 30}).to_csv(source, index=False)
            cfg = ExperimentConfig(DatasetConfig(str(source), target_column="label", feature_columns=("x",)),
                                   "classification", "C28", output_dir=str(output))
            session = freeze_experiment(prepare_experiment(cfg))
            output.mkdir()
            existing = output / "existing.txt"
            existing.touch()
            with self.assertRaises(ArtifactError):
                evaluate_test(session)
            self.assertEqual(session.test_evaluation_count, 0)
            existing.unlink()
            with patch("pyml_workbench.experiment.write_literal_xlsx", side_effect=OSError("disk unavailable")):
                with self.assertRaisesRegex(OSError, "disk unavailable"):
                    evaluate_test(session)
            self.assertEqual(list(output.iterdir()), [])
            cached = evaluate_test(session, export_artifacts=False)
            self.assertIs(evaluate_test(session), cached)
            self.assertEqual(session.test_evaluation_count, 1)
            workbook = load_workbook(output / "results.xlsx", read_only=True)
            try:
                labelled = [cell for row in workbook["results"] for cell in row if cell.value == "=1+1"]
                self.assertTrue(labelled)
                self.assertTrue(all(cell.data_type == "s" for cell in labelled))
            finally:
                workbook.close()

    def test_corrupt_plugin_registry_can_be_restored_with_original_bytes(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plugins.json"
            path.write_bytes(b"damaged registry")
            store = PluginSettingsStore(path)
            with self.assertRaises(PluginError):
                store.load_all()
            backup = store.recover()
            self.assertEqual(backup.read_bytes(), b"damaged registry")
            store.save(PluginSettings(directory), PYEDA_SOURCE)
            self.assertEqual(store.load(PYEDA_SOURCE).directory, directory)


class PyedaPipelineTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.run = Path(self.directory.name)
        self.conversion = self.run / "conversion"
        self.conversion.mkdir()
        self.table("dataset.csv", ["sample_id", "meta__subject_id", "feature__x", "target__score", "target__label"],
                   [("a", "001", "Inf", 1, "NA"), ("b", "1", "2", 2, "001"), ("c", "NA", "3", 3, "1")])
        self.table("valid_mask.csv", ["sample_id", "feature__x", "target__score", "target__label"],
                   [("a", 0, 1, 0), ("b", 1, 1, 1), ("c", 1, 1, 1)])
        (self.conversion / "feature_dictionary.json").write_text(
            json.dumps({"features": [{"output_column": "feature__x"}]}), encoding="utf-8")
        self.config = {"target_column": "target__score", "task": "regression", "group_column": "meta__subject_id",
                       "machine_learning": {"outcomes": ["target__score", "target__label"],
                                            "outcome_tasks": {"target__score": "regression", "target__label": "classification"},
                                            "all_missing_feature_policy": "constant_zero_with_indicator"}}

    def table(self, name, fields, rows):
        with (self.conversion / name).open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(fields)
            writer.writerows(rows)

    def test_preparation_preserves_identities_and_ordinary_classification(self):
        data = pipeline.prepare_ml_dataset(self.config, self.run, None)
        table = pipeline._read_text_csv(Path(data["outcome_paths"]["target__label"]))
        self.assertEqual(list(table["meta__subject_id"]), ["001", "1", "NA"])
        self.assertEqual(list(table["feature__x"]), ["", "2.0", "3.0"])
        self.assertEqual(data["prep"]["classification_label_map"]["target__label"],
                         {"class__NA": "NA", "class__001": "001", "class__1": "1"})

    def test_rejects_other_outcome_as_feature_and_misaligned_mask(self):
        ml = self.config["machine_learning"]
        ml["feature_columns"] = ["target__label"]
        with self.assertRaisesRegex(pipeline.PipelineError, "已登记"):
            pipeline.prepare_ml_dataset(self.config, self.run, None)
        del ml["feature_columns"]
        mask = self.conversion / "valid_mask.csv"
        mask.write_text(mask.read_text(encoding="utf-8").replace("a,0,1,0", "other,0,1,0"), encoding="utf-8")
        with self.assertRaisesRegex(pipeline.PipelineError, "样本顺序"):
            pipeline.prepare_ml_dataset(self.config, self.run, None)

    def test_cli_returns_failure_for_partial_delivery(self):
        config = self.run / "config.json"
        config.write_text("{}", encoding="utf-8")
        with patch.object(pipeline, "run_pipeline", return_value={"status": "completed_with_report_failure"}):
            self.assertEqual(pipeline.main(["--config", str(config)]), 1)

    def test_ioc_checks_hash_and_audits_unmatched_events(self):
        source = self.run / "events.csv"
        source.write_text("group_id,task,participant_label,event_id,ioc\nG1,A,001,e1,0\nG1,A,001,e2,1\nG2,A,NA,e1,1\n",
                          encoding="utf-8")
        digest = hashlib.sha256(source.read_bytes()).hexdigest()
        (self.run / "manifest.json").write_text(json.dumps({"artifacts": [{"path": source.name, "sha256": digest}]}), encoding="utf-8")
        table = pd.DataFrame({"meta__group_id": ["G1"], "meta__task": ["A"],
                              "meta__participant_label": ["001"], "meta__subject_id": ["NA"], "feature__x": [2.0]})
        config = {"input": str(self.run), "conversion": {"verify_manifest_hashes": True},
                  "machine_learning": {"ioc_classification": {"source": source.name, "target_column": "ioc",
                                      "join_keys": ["group_id", "task", "participant_label"]}}}
        rows, audit = pipeline._prepare_ioc(config, table, table, ["feature__x"])
        self.assertEqual(audit["unmatched_labeled_events"], 1)
        self.assertAlmostEqual(audit["match_rate"], 2 / 3)
        self.assertEqual(rows["sample_id"].nunique(), 2)
        source.write_text(source.read_text(encoding="utf-8").replace("e1,0", "e1,1"), encoding="utf-8")
        with self.assertRaisesRegex(pipeline.PipelineError, "清单哈希"):
            pipeline._prepare_ioc(config, table, table, ["feature__x"])

    def test_search_never_calls_test_and_requires_a_real_time_axis(self):
        outcome = "target__score"
        path = self.run / "outcome.csv"
        pd.DataFrame({"meta__subject_id": [f"s{n}" for n in range(20)], "feature__x": range(20),
                      outcome: range(20)}).to_csv(path, index=False)
        self.config.update(model_id="R01")
        self.config["machine_learning"].update(outcomes=[outcome], methods=["R01", "T01"], iterations=1,
                                              evaluation_mode="validation")
        data = {"outcome_paths": {outcome: str(path)}, "features": ["feature__x"],
                "prep": {"outcomes": [outcome], "all_missing_features": []}}
        with patch("pyml_workbench.evaluate_test", side_effect=AssertionError("unexpected test access")):
            result = pipeline.run_machine_learning(self.config, self.run, data, None, None)
        self.assertEqual(result["counts"], {"completed": 1, "unavailable": 1})
        self.assertNotIn("test", result["primary_result"]["metrics"])
        self.assertEqual(result["primary_result"]["test_evaluation_count"], 0)
