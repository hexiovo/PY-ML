from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import importlib.util
import os
import subprocess
import sys
import threading
import time
import unittest
from unittest.mock import patch

import numpy as np
from types import SimpleNamespace

from pyml_workbench.sequence_models import (
    ExtendedModelError,
    build_hmm_estimator,
    deep_cpu_context,
    fit_deep_validation,
    fit_hmm,
    hmm_test_outputs,
    refit_deep_fixed_epochs,
    score_hmm,
    _torch_module_type,
    resolve_deep_seed,
    validate_deep_inputs,
    validate_deep_targets,
    validate_extended_parameters,
)


class _TensorLike:
    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return np.array([1.0], dtype=np.float32)


class _FakeModule:
    def state_dict(self):
        return {"weight": _TensorLike()}


class _FakeEstimator:
    def __init__(self, *, best_epoch: int, losses: list[float]):
        self.callbacks_ = [("early_stopping", SimpleNamespace(best_epoch_=best_epoch))]
        self.history = [
            {"epoch": index + 1, "train_loss": loss + 0.5, "valid_loss": loss}
            for index, loss in enumerate(losses)
        ]
        self.module_ = _FakeModule()
        self.fit_args = None

    def fit(self, x, y):
        self.fit_args = (x, y)
        return self


class _HMMSpy:
    def __init__(self):
        self.fit_lengths = None
        self.score_lengths = []
        self.decode_lengths = None
        self.proba_lengths = None

    def fit(self, observations, *, lengths):
        self.fit_lengths = list(lengths)
        return self

    def score(self, observations, *, lengths):
        self.score_lengths.append(list(lengths))
        return float(len(observations)) * -0.5

    def decode(self, observations, *, lengths):
        self.decode_lengths = list(lengths)
        return -0.5 * len(observations), np.zeros(len(observations), dtype=np.int64)

    def predict_proba(self, observations, *, lengths):
        self.proba_lengths = list(lengths)
        return np.full((len(observations), 2), 0.5, dtype=np.float64)


class SequenceModelContractTests(unittest.TestCase):
    def test_importing_helper_modules_does_not_import_optional_model_libraries(self):
        code = (
            "import sys; import pyml_workbench.sequence_models; "
            "import pyml_workbench.extended_experiment; "
            "assert not any(name in sys.modules for name in ('torch', 'skorch', 'hmmlearn'))"
        )
        environment = os.environ.copy()
        result = subprocess.run([sys.executable, "-X", "utf8", "-c", code], capture_output=True, text=True, env=environment, check=False)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_parameters_reject_bad_bounds_and_unknown_fields_before_fit(self):
        for model_id, parameters in (
            ("H01", {"n_components": True}),
            ("N04", {"max_epochs": 101}),
            ("N04", {"max_epochs": 3.5}),
            ("N04", {"random_state": True}),
            ("N04", {"random_state": -1}),
            ("N04", {"random_state": 1 << 32}),
            ("N04", {"random_state": 3.5}),
            ("H01", {"random_state": True}),
            ("H01", {"random_state": -1}),
            ("H01", {"random_state": 1 << 32}),
            ("H01", {"random_state": 3.5}),
            ("N06", {"unrecognized": 2}),
        ):
            with self.subTest(parameters=parameters), self.assertRaises(ExtendedModelError):
                validate_extended_parameters(model_id, parameters)

        for model_id, valid_letters in (("H01", "stmc"), ("H02", "stmcw"), ("H03", "ste")):
            for value in (valid_letters, ""):
                parameters = {"params": value, "init_params": value, "verbose": False}
                validated = validate_extended_parameters(model_id, parameters)
                self.assertEqual(validated["params"], value)
                self.assertEqual(validated["init_params"], value)
        self.assertEqual(validate_extended_parameters("H01", {"algorithm": "map"})["algorithm"], "map")
        self.assertEqual(
            validate_extended_parameters("H01", {"implementation": "scaling"})["implementation"],
            "scaling",
        )
        for model_id, parameters in (
            ("H01", {"algorithm": "greedy"}),
            ("H01", {"implementation": "fast"}),
            ("H01", {"verbose": 1}),
            ("H01", {"params": "w"}),
            ("H01", {"init_params": "e"}),
            ("H02", {"params": "e"}),
            ("H03", {"params": "c"}),
        ):
            with self.subTest(model_id=model_id, parameters=parameters), self.assertRaises(ExtendedModelError):
                validate_extended_parameters(model_id, parameters)

        self.assertIsNone(validate_extended_parameters("N06")["random_state"])
        self.assertIsNone(validate_extended_parameters("N06", {"random_state": None})["random_state"])
        self.assertEqual(resolve_deep_seed({"random_state": None}, 17), 17)
        self.assertEqual(resolve_deep_seed({"random_state": 0}, 17), 0)
        for invalid_fallback in (True, -1, 1 << 32, 1.5):
            with self.subTest(fallback=invalid_fallback), self.assertRaises(ExtendedModelError):
                resolve_deep_seed({"random_state": None}, invalid_fallback)

    def test_deep_tensor_protocol_rejects_rowwise_or_nonfinite_window_inputs(self):
        values = np.ones((5, 10, 2), dtype=np.float32)
        self.assertEqual(validate_deep_inputs("N04", values).shape, (5, 10, 2))
        readonly = values.copy()
        readonly.setflags(write=False)
        validated = validate_deep_inputs("N04", readonly)
        self.assertTrue(validated.flags.writeable)
        self.assertIsNot(validated, readonly)
        with self.assertRaisesRegex(ExtendedModelError, r"shape \(batch, window, features\)"):
            validate_deep_inputs("N04", np.ones((5, 2), dtype=np.float32))
        values[0, 0, 0] = np.inf
        with self.assertRaisesRegex(ExtendedModelError, "must be finite"):
            validate_deep_inputs("N06", values)

    @unittest.skipUnless(importlib.util.find_spec("torch") is not None, "torch is not installed")
    def test_tabular_mlp_num_layers_controls_hidden_linear_count_and_parameter_count(self):
        torch, module_type = _torch_module_type()
        shared = {
            "architecture": "mlp",
            "input_size": 3,
            "output_size": 2,
            "hidden_size": 4,
        }
        one_layer = module_type(**shared, num_layers=1)
        three_layers = module_type(**shared, num_layers=3)

        def hidden_linears(module):
            return [item for item in module.backbone.modules() if isinstance(item, torch.nn.Linear)]

        self.assertEqual(len(hidden_linears(one_layer)), 1)
        self.assertEqual(len(hidden_linears(three_layers)), 3)
        self.assertGreater(
            sum(parameter.numel() for parameter in three_layers.parameters()),
            sum(parameter.numel() for parameter in one_layer.parameters()),
        )
        self.assertEqual(one_layer.head[-1].out_features, 2)
        self.assertEqual(three_layers.head[-1].out_features, 2)
        self.assertEqual(tuple(one_layer(torch.ones((2, 3))).shape), (2, 2))
        self.assertEqual(tuple(three_layers(torch.ones((2, 3))).shape), (2, 2))

    def test_classification_targets_are_int64_and_bound_to_train_mapping(self):
        values = validate_deep_targets("N01", np.array([0, 1, 0], dtype=np.int32), class_count=2)
        self.assertEqual(values.dtype, np.int64)
        readonly = np.array([0.5, 1.5], dtype=np.float32)
        readonly.setflags(write=False)
        writable_targets = validate_deep_targets("N02", readonly)
        self.assertTrue(writable_targets.flags.writeable)
        self.assertIsNot(writable_targets, readonly)
        with self.assertRaisesRegex(ExtendedModelError, "outside the frozen train-only mapping"):
            validate_deep_targets("N01", np.array([2], dtype=np.int64), class_count=2)

    def test_every_hmm_operation_receives_exact_partition_lengths(self):
        partition = SimpleNamespace(
            name="train",
            observations=np.zeros((9, 2), dtype=np.float32),
            lengths=np.array([4, 5], dtype=np.int64),
            observation_encoding="continuous_numeric",
        )
        model = _HMMSpy()

        fit_hmm(model, "H01", partition)
        self.assertEqual(model.fit_lengths, [4, 5])
        self.assertAlmostEqual(score_hmm(model, "H01", partition), -0.5)
        value, states, posterior = hmm_test_outputs(model, "H01", partition)
        self.assertAlmostEqual(value, -0.5)
        self.assertEqual(states.shape, (9,))
        self.assertEqual(posterior.shape, (9, 2))
        self.assertEqual(model.decode_lengths, [4, 5])
        self.assertEqual(model.proba_lengths, [4, 5])
        self.assertEqual(model.score_lengths, [[4, 5], [4, 5]])

    def test_hmm_random_state_uses_explicit_value_or_split_seed_fallback(self):
        if importlib.util.find_spec("hmmlearn") is None:
            self.skipTest("sequence extra is not installed")
        self.assertIsNone(validate_extended_parameters("H01")["random_state"])
        explicit = build_hmm_estimator(
            "H01", {"n_components": 2, "n_iter": 1, "random_state": 17}, seed=42
        )
        fallback = build_hmm_estimator(
            "H01", {"n_components": 2, "n_iter": 1, "random_state": None}, seed=42
        )
        omitted = build_hmm_estimator("H01", {"n_components": 2, "n_iter": 1}, seed=42)
        self.assertEqual(explicit.random_state, 17)
        self.assertEqual(fallback.random_state, 42)
        self.assertEqual(omitted.random_state, 42)

    def test_raw_hmm_observations_are_blocked_until_explicit_final_evaluation(self):
        partition = SimpleNamespace(
            name="test",
            observations=np.array([[1.0], [2.0]], dtype=object),
            lengths=np.array([2], dtype=np.int64),
            observation_encoding="continuous_raw_unchecked",
        )
        with self.assertRaisesRegex(ExtendedModelError, "cannot be used before final evaluation"):
            score_hmm(_HMMSpy(), "H01", partition)
        self.assertAlmostEqual(score_hmm(_HMMSpy(), "H01", partition, allow_raw=True), -0.5)

    def test_deep_early_stopping_best_epoch_matches_callback_threshold_rule(self):
        if importlib.util.find_spec("torch") is None or importlib.util.find_spec("skorch") is None:
            self.skipTest("deep extra is not installed")
        import skorch.callbacks

        callback = SimpleNamespace(best_epoch_=1)
        estimator = _FakeEstimator(best_epoch=1, losses=[1.0, 0.99995])
        x_train = np.ones((4, 3), dtype=np.float32)
        y_train = np.arange(4, dtype=np.float32)
        x_valid = np.ones((2, 3), dtype=np.float32)
        y_valid = np.arange(2, dtype=np.float32)
        with patch.object(skorch.callbacks, "EarlyStopping", return_value=callback) as callback_factory:
            with patch("pyml_workbench.sequence_models._deep_estimator", return_value=estimator):
                result = fit_deep_validation(
                    "N02",
                    {"max_epochs": 3},
                    seed=2,
                    train_x=x_train,
                    train_y=y_train,
                    validation_x=x_valid,
                    validation_y=y_valid,
                )
        self.assertEqual(result.best_epoch, 1)
        self.assertEqual(result.selected_epochs, 1)
        self.assertEqual(result.best_weight_policy, "early_stopping_abs_threshold_zero_load_best_valid_loss")
        self.assertEqual(result.training_curves["epochs"], [1, 2])
        np.testing.assert_allclose(result.training_curves["train_loss"], [1.5, 1.49995])
        np.testing.assert_allclose(result.training_curves["validation_loss"], [1.0, 0.99995])
        self.assertTrue(result.training_curves["validation_available"])
        kwargs = callback_factory.call_args.kwargs
        self.assertEqual(kwargs["threshold"], 0.0)
        self.assertEqual(kwargs["threshold_mode"], "abs")
        self.assertTrue(kwargs["load_best"])

    @unittest.skipUnless(importlib.util.find_spec("torch") is not None and importlib.util.find_spec("skorch") is not None, "deep extra is not installed")
    def test_real_deep_refit_uses_fresh_module_and_exact_epoch_count(self):
        import torch

        rng = np.random.default_rng(22)
        x = rng.normal(size=(12, 4, 2)).astype(np.float32)
        y = (x[:, -1, 0] + x[:, -1, 1]).astype(np.float32).reshape(-1, 1)
        validation_x = rng.normal(size=(4, 4, 2)).astype(np.float32)
        validation_y = (validation_x[:, -1, 0] + validation_x[:, -1, 1]).astype(np.float32).reshape(-1, 1)
        validation = fit_deep_validation(
            "N06",
            {"max_epochs": 2, "patience": 1, "batch_size": 4},
            seed=9,
            train_x=x,
            train_y=y,
            validation_x=validation_x,
            validation_y=validation_y,
        )
        before_threads = torch.get_num_threads()
        before_rng = torch.random.get_rng_state().clone()
        refit = refit_deep_fixed_epochs(
            "N06",
            {"max_epochs": 2, "batch_size": 4},
            seed=9,
            train_x=np.concatenate([x, validation_x]),
            train_y=np.concatenate([y, validation_y]),
            selected_epochs=2,
        )

        self.assertIsNot(validation.estimator.module_, refit.module_)
        self.assertEqual(len(refit.history), 2)
        self.assertIsNone(refit.train_split)
        self.assertEqual(torch.get_num_threads(), before_threads)
        self.assertTrue(torch.equal(torch.random.get_rng_state(), before_rng))

    @unittest.skipUnless(importlib.util.find_spec("torch") is not None, "torch is not installed")
    def test_deep_training_context_serializes_global_torch_changes(self):
        active = 0
        maximum = 0
        guard = threading.Lock()

        def task(seed):
            nonlocal active, maximum
            with deep_cpu_context(seed):
                with guard:
                    active += 1
                    maximum = max(maximum, active)
                time.sleep(0.03)
                with guard:
                    active -= 1

        with ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(task, (11, 12)))
        self.assertEqual(maximum, 1)


if __name__ == "__main__":
    unittest.main()
