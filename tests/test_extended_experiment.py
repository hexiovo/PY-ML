from __future__ import annotations

import copy
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd

from pyml_workbench.config import DatasetConfig, ExperimentConfig
from pyml_workbench.experiment import _build_snapshot_from_frame, build_extended_snapshot
from pyml_workbench.extended_experiment import (
    ExtendedExperimentError,
    evaluate_extended_test,
    export_extended_model,
    load_extended_model,
    predict_extended,
    prepare_extended_experiment,
    refit_extended_session,
    score_extended_session,
)
from pyml_workbench.sequence import SequenceConfig
from pyml_workbench.sequence_models import ExtendedModelError


_SOURCE_SHA = "a" * 64
_EXTENDED_IDS = ("H01", "H02", "H03", "N01", "N02", "N04", "N06")


def _fixture_frame() -> pd.DataFrame:
    rows = []
    origin = pd.Timestamp("2025-01-01", tz="UTC")
    for group in range(15):
        for step in range(12):
            rows.append(
                {
                    "group": f"g{group:02d}",
                    "time": origin + pd.Timedelta(days=step),
                    "value": float(group * 12 + step),
                    "kind": "a" if step % 2 == 0 else "b",
                    "target": float(group * 0.5 + step * 0.25),
                    "label": step % 2,
                }
            )
    return pd.DataFrame(rows)


def _config(model_id: str, source_path: str) -> ExperimentConfig:
    if model_id in {"H01", "H02", "H03"}:
        observation = ("kind",) if model_id == "H03" else ("value",)
        sequence = SequenceConfig(
            group_column="group",
            time_column="time",
            order_mode="time",
            observation_columns=observation,
        )
        task, target, features = "sequence_modeling", None, observation
        parameters = {"n_components": 2, "n_iter": 2}
        if model_id == "H02":
            parameters["n_mix"] = 1
    elif model_id == "N01":
        sequence = None
        task, target, features = "classification", "label", ("value",)
        parameters = {"hidden_size": 4, "batch_size": 8, "max_epochs": 2, "patience": 1, "random_state": 4}
    else:
        task, target, features = "regression", "target", ("value",)
        parameters = {"hidden_size": 4, "batch_size": 8, "max_epochs": 2, "patience": 1, "random_state": 4}
        sequence = (
            SequenceConfig(group_column="group", time_column="time", order_mode="time", window=3, horizon=1)
            if model_id in {"N04", "N06"}
            else None
        )
    return ExperimentConfig(
        dataset=DatasetConfig(source_path=source_path, target_column=target, feature_columns=features),
        task=task,
        model_id=model_id,
        parameters=parameters,
        sequence=sequence,
    )


class ExtendedExperimentTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.root = Path(self.tempdir.name)
        self.source = self.root / "sequence-fixture.csv"
        self.frame = _fixture_frame()
        self.frame.to_csv(self.source, index=False)

    def tearDown(self):
        self.tempdir.cleanup()

    def _build(self, model_id: str, frame: pd.DataFrame | None = None):
        if frame is not None:
            frame.to_csv(self.source, index=False)
        config = _config(model_id, str(self.source))
        snapshot, plan = build_extended_snapshot(config)
        return config, snapshot, plan

    def test_all_seven_models_fit_validation_and_round_trip_cpu_bundles_without_raw_data(self):
        for model_id in _EXTENDED_IDS:
            with self.subTest(model_id=model_id):
                config, snapshot, plan = self._build(model_id, _fixture_frame())
                source_windows = None
                if model_id in {"N04", "N06"}:
                    source_windows = {
                        name: (
                            plan.partitions[name].window_inputs.copy(),
                            plan.partitions[name].window_targets.copy(),
                        )
                        for name in ("train", "validation", "test")
                    }
                session = prepare_extended_experiment(config, snapshot=snapshot, sequence_plan=plan)
                self.assertEqual(session.fit_scope, "train")
                self.assertEqual(session.test_evaluation_count, 0)
                self.assertFalse(session.finalized)
                self.assertEqual(session.events, [])
                self.assertNotIn("test", session.metrics)
                objective = SimpleNamespace(
                    metric="log_likelihood_per_observation" if model_id in {"H01", "H02", "H03"} else "accuracy" if model_id == "N01" else "r2",
                    split="validation",
                )
                value, metrics = score_extended_session(session, objective)
                self.assertIsNotNone(value)
                self.assertIn(objective.metric, metrics)

                bundle_dir = self.root / model_id
                path = export_extended_model(session, bundle_dir)
                payload = joblib.load(path)
                self.assertTrue(
                    {"snapshot", "sequence_plan", "features", "target", "optimizer", "train_x", "train_y", "test_x", "test_y"}.isdisjoint(payload)
                )
                self.assertIn("state_dict" if model_id not in {"H01", "H02", "H03"} else "estimator", payload)
                loaded = load_extended_model(path)
                if model_id in {"H01", "H02"}:
                    values = plan.validation.observations[:2]
                    prediction = predict_extended(loaded, {"observations": values, "lengths": [len(values)]})
                elif model_id == "H03":
                    prediction = predict_extended(loaded, {"observations": ["a", "b"], "lengths": [2]})
                elif model_id in {"N04", "N06"}:
                    prediction = predict_extended(loaded, plan.validation.window_inputs[:2])
                else:
                    prediction = predict_extended(loaded, pd.DataFrame({"value": [0.25, 0.75]}))
                self.assertEqual(len(prediction), 2)
                if source_windows is not None:
                    for name, (inputs_before, targets_before) in source_windows.items():
                        np.testing.assert_array_equal(plan.partitions[name].window_inputs, inputs_before)
                        np.testing.assert_array_equal(plan.partitions[name].window_targets, targets_before)
                self.assertEqual(session.test_evaluation_count, 0)
                self.assertEqual(session.events, [])

    def test_tabular_mlp_num_layers_fit_export_and_reload_for_classification_and_regression(self):
        import torch

        for model_id in ("N01", "N02"):
            with self.subTest(model_id=model_id):
                config, snapshot, plan = self._build(model_id, _fixture_frame())
                config.parameters["num_layers"] = 3
                session = prepare_extended_experiment(config, snapshot=snapshot, sequence_plan=plan)
                hidden_linears = [
                    module
                    for module in session.fitted_model.estimator.module_.backbone.modules()
                    if isinstance(module, torch.nn.Linear)
                ]
                self.assertEqual(len(hidden_linears), 3)

                path = export_extended_model(session, self.root / f"{model_id}-three-layers")
                loaded = load_extended_model(path)
                self.assertEqual(loaded.parameters["num_layers"], 3)
                predictions = predict_extended(loaded, pd.DataFrame({"value": [0.25, 0.75]}))
                self.assertEqual(np.asarray(predictions).shape, (2,))
                self.assertNotIn("test", session.metrics)
                self.assertEqual(session.test_evaluation_count, 0)

    def test_hmm_seed_provenance_refit_statistics_and_cpu_bundle_round_trip(self):
        cases = ((17, 17), (None, 42))
        for requested_seed, expected_seed in cases:
            with self.subTest(requested_seed=requested_seed):
                config = _config("H01", str(self.source))
                if requested_seed is not None:
                    config.parameters["random_state"] = requested_seed
                snapshot, plan = build_extended_snapshot(config)
                winner = prepare_extended_experiment(config, snapshot=snapshot, sequence_plan=plan)
                self.assertEqual(winner.extended_training_metadata["effective_training_seed"], expected_seed)
                self.assertEqual(winner.refit_provenance["effective_training_seed"], expected_seed)
                self.assertEqual(winner.fitted_model.estimator.random_state, expected_seed)

                bad_provenance = copy.deepcopy(winner.refit_provenance)
                bad_provenance["effective_training_seed"] = expected_seed + 1
                with self.assertRaisesRegex(ExtendedExperimentError, "effective_training_seed does not match"):
                    refit_extended_session(
                        config,
                        snapshot=snapshot,
                        parameters=config.parameters,
                        refit_provenance=bad_provenance,
                        sequence_plan=plan,
                    )

                refit = refit_extended_session(
                    config,
                    snapshot=snapshot,
                    parameters=config.parameters,
                    refit_provenance=winner.refit_provenance,
                    sequence_plan=plan,
                )
                self.assertEqual(refit.extended_training_metadata["effective_training_seed"], expected_seed)
                self.assertEqual(refit.refit_provenance["effective_training_seed"], expected_seed)
                self.assertEqual(refit.fitted_model.estimator.random_state, expected_seed)

                fit_observations = np.concatenate(
                    [plan.train.observations, plan.validation.observations], axis=0
                )
                fit_lengths = np.concatenate([plan.train.lengths, plan.validation.lengths])
                direct_score = refit.fitted_model.estimator.score(
                    fit_observations, lengths=fit_lengths.tolist()
                ) / len(fit_observations)
                refit_metrics = refit.metrics["refit"]
                self.assertEqual(refit_metrics["n_observations"], len(fit_observations))
                self.assertEqual(refit_metrics["n_sequences"], len(fit_lengths))
                self.assertAlmostEqual(refit_metrics["log_likelihood_per_observation"], direct_score)
                self.assertNotIn("test", refit.metrics)

                if expected_seed == 17:
                    repeated = refit_extended_session(
                        config,
                        snapshot=snapshot,
                        parameters=config.parameters,
                        refit_provenance=winner.refit_provenance,
                        sequence_plan=plan,
                    )
                    for name in ("startprob_", "transmat_", "means_", "covars_"):
                        np.testing.assert_array_equal(
                            getattr(refit.fitted_model.estimator, name),
                            getattr(repeated.fitted_model.estimator, name),
                        )

                    evidence_dir = os.environ.get("PYML_S03_HMM_BUNDLE_DIR")
                    bundle_path = (
                        Path(evidence_dir) / "model.joblib"
                        if evidence_dir
                        else self.root / "H01-explicit-refit" / "model.joblib"
                    )
                    if evidence_dir:
                        bundle_path.parent.mkdir(parents=True, exist_ok=True)
                    export_extended_model(refit, bundle_path)
                    loaded = load_extended_model(bundle_path)
                    self.assertEqual(loaded.estimator.random_state, expected_seed)
                    for name in ("startprob_", "transmat_", "means_", "covars_"):
                        np.testing.assert_array_equal(
                            getattr(refit.fitted_model.estimator, name),
                            getattr(loaded.estimator, name),
                        )
                    predictions = predict_extended(
                        loaded,
                        {
                            "observations": plan.validation.observations[:4],
                            "lengths": [4],
                        },
                    )
                    self.assertEqual(np.asarray(predictions).shape, (4,))
                    if evidence_dir:
                        inputs_path = Path(evidence_dir).with_name("S03-H01-refit-oracle-inputs.npz")
                        np.savez(inputs_path, observations=fit_observations, lengths=fit_lengths)
                        print(f"S03_H01_BUNDLE={bundle_path}")
                        print(f"S03_H01_ORACLE_INPUTS={inputs_path}")

    def test_hmm_invalid_parameters_are_rejected_before_estimator_build(self):
        for parameters in (
            {"random_state": 1 << 32},
            {"algorithm": "greedy"},
            {"implementation": "fast"},
            {"verbose": 1},
            {"params": "w"},
            {"init_params": "e"},
        ):
            with self.subTest(parameters=parameters):
                config = _config("H01", str(self.source))
                config.parameters.update(parameters)
                snapshot, plan = build_extended_snapshot(config)
                with patch("pyml_workbench.extended_experiment.build_hmm_estimator") as builder:
                    with self.assertRaises(ExtendedModelError):
                        prepare_extended_experiment(config, snapshot=snapshot, sequence_plan=plan)
                builder.assert_not_called()

    def test_test_only_unknown_hmm_category_is_deferred_to_pure_final_computation(self):
        _, _, base_plan = self._build("H03", _fixture_frame())
        test_groups = set(base_plan.test.group_ids)
        frame = _fixture_frame()
        frame.loc[frame["group"].isin(test_groups), "kind"] = "test-only-symbol"
        config, snapshot, plan = self._build("H03", frame)
        session = prepare_extended_experiment(config, snapshot=snapshot, sequence_plan=plan)

        self.assertEqual(session.fitted_model.categorical_alphabet, ["a", "b"])
        self.assertEqual(plan.test.observation_encoding, "categorical_raw_unchecked")
        self.assertEqual(session.test_evaluation_count, 0)
        with self.assertRaisesRegex(ExtendedModelError, "unknown to the frozen train-only alphabet"):
            evaluate_extended_test(session)
        self.assertEqual(session.test_evaluation_count, 0)
        self.assertEqual(session.events, [])
        self.assertFalse(session.finalized)
        self.assertNotIn("test", session.metrics)
        self.assertNotIn("test-only-symbol", session.fitted_model.categorical_alphabet)

    def test_classification_test_only_target_is_not_used_until_final_computation(self):
        frame = _fixture_frame()
        frame["label"] = frame["label"].astype(object)
        frame.loc[144:, "label"] = "test-only-class"
        config = _config("N01", str(self.source))
        splits = {
            "train": np.arange(0, 108, dtype=np.int64),
            "validation": np.arange(108, 144, dtype=np.int64),
            "test": np.arange(144, 180, dtype=np.int64),
        }
        snapshot = _build_snapshot_from_frame(
            config,
            frame,
            source_path=self.source,
            source_sha256=_SOURCE_SHA,
            splits=splits,
        )
        session = prepare_extended_experiment(config, snapshot=snapshot)
        self.assertEqual([item["label"] for item in session.fitted_model.class_mapping], [0, 1])
        self.assertEqual(session.test_evaluation_count, 0)
        with self.assertRaisesRegex(ExtendedExperimentError, "test class.*unknown to the frozen train-only mapping"):
            evaluate_extended_test(session)
        self.assertEqual(session.test_evaluation_count, 0)
        self.assertEqual(session.events, [])
        self.assertNotIn("test", session.metrics)

    def test_deep_refit_rejects_mismatched_winner_plan_provenance(self):
        config, snapshot, plan = self._build("N06", _fixture_frame())
        winner = prepare_extended_experiment(config, snapshot=snapshot, sequence_plan=plan)
        provenance = copy.deepcopy(winner.refit_provenance)
        self.assertIsNotNone(provenance)
        provenance["sequence_plan_manifest_sha256"] = "0" * 64

        with self.assertRaisesRegex(ExtendedExperimentError, "SequencePlan digest does not match"):
            refit_extended_session(
                config,
                snapshot=snapshot,
                parameters=config.parameters,
                selected_epochs=winner.extended_training_metadata["selected_epochs"],
                refit_provenance=provenance,
                sequence_plan=plan,
            )

    def test_refit_provenance_freezes_selected_epoch_and_test_computation_is_stateless(self):
        config, snapshot, plan = self._build("N06", _fixture_frame())
        winner = prepare_extended_experiment(config, snapshot=snapshot, sequence_plan=plan)
        selected_epochs = winner.extended_training_metadata["selected_epochs"]
        refit = refit_extended_session(
            config,
            snapshot=snapshot,
            parameters=config.parameters,
            selected_epochs=selected_epochs,
            refit_provenance=winner.refit_provenance,
            sequence_plan=plan,
        )
        self.assertEqual(refit.fit_scope, "train_validation")
        self.assertEqual(refit.extended_training_metadata["selected_epochs"], selected_epochs)
        self.assertEqual(refit.refit_provenance["selected_epochs"], selected_epochs)

        computation_a = evaluate_extended_test(refit)
        computation_b = evaluate_extended_test(refit)
        self.assertEqual(computation_a.metrics, computation_b.metrics)
        self.assertEqual(computation_a.rows, computation_b.rows)
        self.assertEqual(refit.test_evaluation_count, 0)
        self.assertEqual(refit.events, [])
        self.assertFalse(refit.finalized)
        self.assertNotIn("test", refit.metrics)

    def test_deep_random_state_controls_weights_and_fresh_refit_reuses_effective_seed(self):
        config = _config("N06", str(self.source))
        config.parameters.pop("random_state")
        snapshot, plan = build_extended_snapshot(config)
        default_seed_fit = prepare_extended_experiment(config, snapshot=snapshot, sequence_plan=plan)
        self.assertEqual(default_seed_fit.extended_training_metadata["effective_training_seed"], config.split.seed)
        self.assertEqual(default_seed_fit.refit_provenance["effective_training_seed"], config.split.seed)
        self.assertNotIn("random_state", default_seed_fit.frozen_config["parameters"])
        curves = default_seed_fit.training_metadata["training_curves"]
        self.assertEqual(curves["schema_version"], 1)
        self.assertEqual(curves["model_id"], "N06")
        self.assertEqual(curves["fit_scope"], "train")
        self.assertTrue(curves["validation_available"])
        self.assertEqual(len(curves["epochs"]), len(curves["train_loss"]))
        self.assertEqual(len(curves["epochs"]), len(curves["validation_loss"]))
        self.assertEqual(curves["epochs"], list(range(1, len(curves["epochs"]) + 1)))
        self.assertTrue(np.isfinite(curves["train_loss"]).all())
        self.assertTrue(np.isfinite(curves["validation_loss"]).all())

        override_config = ExperimentConfig.from_dict(config.to_dict())
        override_config.parameters["random_state"] = config.split.seed + 1
        override_seed_fit = prepare_extended_experiment(override_config, snapshot=snapshot, sequence_plan=plan)
        self.assertEqual(
            override_seed_fit.extended_training_metadata["effective_training_seed"],
            config.split.seed + 1,
        )
        self.assertTrue(
            any(
                not np.array_equal(default_seed_fit.fitted_model.model_state_dict[name], override_seed_fit.fitted_model.model_state_dict[name])
                for name in default_seed_fit.fitted_model.model_state_dict
            ),
            "changing the effective training seed should change at least one fitted tensor",
        )

        bad_provenance = copy.deepcopy(default_seed_fit.refit_provenance)
        bad_provenance["effective_training_seed"] = config.split.seed + 1
        with self.assertRaisesRegex(ExtendedExperimentError, "effective_training_seed does not match"):
            refit_extended_session(
                config,
                snapshot=snapshot,
                parameters=config.parameters,
                selected_epochs=default_seed_fit.extended_training_metadata["selected_epochs"],
                refit_provenance=bad_provenance,
                sequence_plan=plan,
            )

        refit_a = refit_extended_session(
            config,
            snapshot=snapshot,
            parameters=config.parameters,
            selected_epochs=default_seed_fit.extended_training_metadata["selected_epochs"],
            refit_provenance=default_seed_fit.refit_provenance,
            sequence_plan=plan,
        )
        refit_b = refit_extended_session(
            config,
            snapshot=snapshot,
            parameters=config.parameters,
            selected_epochs=default_seed_fit.extended_training_metadata["selected_epochs"],
            refit_provenance=default_seed_fit.refit_provenance,
            sequence_plan=plan,
        )
        self.assertEqual(refit_a.extended_training_metadata["effective_training_seed"], config.split.seed)
        refit_curves = refit_a.training_metadata["training_curves"]
        self.assertEqual(refit_curves["fit_scope"], "train_validation")
        self.assertFalse(refit_curves["validation_available"])
        self.assertIsNone(refit_curves["validation_loss"])
        self.assertEqual(len(refit_curves["epochs"]), len(refit_curves["train_loss"]))
        self.assertTrue(np.isfinite(refit_curves["train_loss"]).all())
        self.assertEqual(set(refit_a.fitted_model.model_state_dict), set(refit_b.fitted_model.model_state_dict))
        for name in refit_a.fitted_model.model_state_dict:
            np.testing.assert_array_equal(
                refit_a.fitted_model.model_state_dict[name],
                refit_b.fitted_model.model_state_dict[name],
            )


if __name__ == "__main__":
    unittest.main()
