"""Release integration: both CV contracts and fold-local native reduction."""
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

from pyml_workbench import ConfigError, DatasetConfig, ExperimentConfig, EvaluationConfig, SplitConfig
from pyml_workbench.experiment import build_snapshot, prepare_experiment, evaluate_test
from pyml_workbench.cv_experiment import refit_cv_session, verify_cv_session
from pyml_workbench.presets import save_preset, load_preset
from pyml_workbench.search import SearchSpec, search
from pyml_workbench.objectives import ObjectiveSpec


class ReleaseIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        rng = np.random.default_rng(46)
        self.frame = pd.DataFrame(rng.normal(size=(150, 5)), columns=list("abcde"))
        self.frame["category"] = np.tile(["A", "B", "C"], 50)
        self.frame["target"] = (self.frame.a + self.frame.b > 0).astype(int)
        self.frame["group"] = np.repeat(np.arange(15), 10)
        self.path = Path(self.tmp.name) / "data.csv"
        self.frame.to_csv(self.path, index=False)
        self.config = ExperimentConfig(DatasetConfig(str(self.path), target_column="target", feature_columns=tuple("abcde") + ("category",)), "classification", "C01")
        self.limits = threadpool_limits(limits=1)
        self.addCleanup(self.limits.restore_original_limits)

    def test_old_identity_and_both_split_dialects(self):
        default = self.config.to_dict()
        self.assertNotIn("strategy", default["split"])
        self.assertNotIn("n_splits", default["split"])
        self.assertEqual(default, replace(self.config, split=SplitConfig(strategy="holdout")).to_dict())
        for strategy in ["random", "holdout", "group", "time", "kfold"]:
            split = SplitConfig(strategy=strategy, group_column="group" if strategy == "group" else None)
            cfg = replace(self.config, split=split)
            self.assertEqual(cfg.to_dict(), ExperimentConfig.from_dict(cfg.to_dict()).to_dict())

    def test_legacy_cv_reduction_search_refit_and_single_test(self):
        cfg = replace(self.config, split=SplitConfig(strategy="kfold", n_splits=3), reduction_method="pca", reduction_components=2)
        session = prepare_experiment(cfg)
        verify_cv_session(session)
        self.assertIsNone(session.fitted_model)
        self.assertEqual(len(session.cv_metadata["fold_preprocessing"]), 3)
        self.assertIsNotNone(session.metrics["validation"]["log_loss"])
        self.assertIsNotNone(session.metrics["oof"]["roc_auc"])
        final = refit_cv_session(session)
        self.assertEqual(final.fitted_model.preprocessing["learned_transform_metadata"]["fit_row_count"], 120)
        evaluate_test(final)
        self.assertEqual(final.test_evaluation_count, 1)
        result = search(cfg, SearchSpec(max_fits=1, max_proposals=1, objective=ObjectiveSpec(metric="log_loss")))
        self.assertEqual(result.trials[0].status, "complete", result.trials[0].error)
        self.assertEqual(result.trials[0].validation_source, "cv_fold_mean")
        from pyml_workbench.selection import freeze_selection, finalize_selected
        selected = freeze_selection(result)
        final_path = Path(self.tmp.name) / "final.joblib"
        final_selection = finalize_selected(selected, session_path=final_path)
        cached = finalize_selected(selected, session_path=final_path)
        first, second = final_selection.to_dict(), cached.to_dict()
        self.assertFalse(first.pop("cached"))
        self.assertTrue(second.pop("cached"))
        self.assertEqual(first, second)

    def test_group_evaluation_keeps_isolation_and_fits_reduction_per_fold(self):
        cfg = replace(self.config, split=SplitConfig(strategy="group", group_column="group"), evaluation=EvaluationConfig(folds=3), reduction_method="pca", reduction_components=2)
        snapshot = build_snapshot(cfg)
        session = prepare_experiment(cfg, snapshot=snapshot)
        report = session.metrics["cross_validation"]
        test = set(snapshot.splits["test"])
        for fold in report["folds"]:
            train, valid = fold["train_positions"], fold["validation_positions"]
            self.assertFalse(test.intersection(train + valid))
            self.assertFalse(set(self.frame.group.iloc[train]).intersection(self.frame.group.iloc[valid]))
            self.assertEqual(fold["reduction"]["fit_row_count"], len(train))

    def test_conflicting_protocols_fail_before_training(self):
        with self.assertRaisesRegex(ConfigError, "configured together"):
            replace(self.config, split=SplitConfig(strategy="kfold"), evaluation=EvaluationConfig()).validate()
        for model_id in ["N07", "N08", "N09", "T01"]:
            with self.assertRaises(ConfigError):
                replace(self.config, model_id=model_id, split=SplitConfig(strategy="kfold")).validate()
        with self.assertRaises(ConfigError):
            replace(self.config, model_id="C01", split=SplitConfig(strategy="group", group_column="group"), evaluation=EvaluationConfig(strategy="kfold")).validate()

    def test_native_reduction_methods_and_preset_roundtrip(self):
        for method in ["pca", "truncated_svd", "fast_ica", "gaussian_random_projection", "kernel_pca"]:
            with self.subTest(method=method):
                cfg = replace(self.config, reduction_method=method, reduction_components=2)
                session = prepare_experiment(cfg)
                self.assertEqual(session.fitted_model.preprocessing["reduction"]["fit_row_count"], 90)
        cfg = replace(self.config, split=SplitConfig(strategy="kfold", n_splits=3), reduction_method="pca", reduction_components=2)
        path = Path(self.tmp.name) / "preset.json"
        save_preset(path, cfg)
        self.assertEqual(load_preset(path).to_dict(), cfg.to_dict())
