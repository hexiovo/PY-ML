"""Lazy optional-model builders for HMM and neural sequence experiments.

The package's base install can import this module without importing torch,
skorch, or hmmlearn. Model constructors and numerical work import those
libraries only when their corresponding model is requested.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import importlib
import math
import random
import threading
from typing import Any, Iterator, Mapping

import numpy as np


HMM_MODEL_IDS = frozenset({"H01", "H02", "H03"})
TABULAR_DEEP_MODEL_IDS = frozenset({"N01", "N02"})
WINDOW_DEEP_MODEL_IDS = frozenset({"N04", "N06"})
DEEP_MODEL_IDS = TABULAR_DEEP_MODEL_IDS | WINDOW_DEEP_MODEL_IDS
EXTENDED_MODEL_IDS = HMM_MODEL_IDS | DEEP_MODEL_IDS
_MAX_RANDOM_STATE = (1 << 32) - 1

_DEEP_TRAIN_LOCK = threading.RLock()
_TORCH_MODULE_TYPE: type | None = None


class ExtendedModelError(ValueError):
    """Raised when an extended model's parameters or inputs violate its contract."""


class OptionalModelDependencyError(ImportError):
    """Raised when an optional model is requested without its extra installed."""


@dataclass(frozen=True)
class DeepFitResult:
    estimator: Any
    selected_epochs: int
    best_epoch: int
    best_weight_policy: str
    training_curves: dict[str, Any]


def validate_extended_parameters(
    model_id: str,
    parameters: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate and fill model-specific parameters without importing optional deps."""
    if model_id not in EXTENDED_MODEL_IDS:
        raise ExtendedModelError(f"Unsupported extended model id: {model_id!r}")
    requested = dict(parameters or {})
    if model_id in HMM_MODEL_IDS:
        defaults: dict[str, Any] = {
            "n_components": 3,
            "n_iter": 100,
            "tol": 0.01,
            "random_state": None,
        }
        if model_id == "H01":
            defaults.update(covariance_type="diag", min_covar=0.001)
            allowed = set(defaults) | {"algorithm", "implementation", "params", "init_params", "verbose"}
            choices = {"covariance_type": {"spherical", "diag", "full", "tied"}}
        elif model_id == "H02":
            defaults.update(n_mix=2, covariance_type="diag", min_covar=0.001)
            allowed = set(defaults) | {"algorithm", "implementation", "params", "init_params", "verbose"}
            choices = {"covariance_type": {"spherical", "diag", "full", "tied"}}
        else:
            allowed = set(defaults) | {"algorithm", "implementation", "params", "init_params", "verbose"}
            choices = {}
    else:
        defaults = {
            "hidden_size": 32,
            "num_layers": 1,
            "batch_size": 32,
            "learning_rate": 0.001,
            "weight_decay": 0.0,
            "max_epochs": 100,
            "patience": 10,
            "random_state": None,
        }
        allowed = set(defaults)
        choices = {}
    unknown = sorted(set(requested) - allowed)
    if unknown:
        raise ExtendedModelError(f"Unsupported parameters for {model_id}: {', '.join(unknown)}")
    values = defaults | requested
    for name in ("n_components", "n_iter", "n_mix", "hidden_size", "num_layers", "batch_size", "max_epochs", "patience"):
        if name in values:
            value = values[name]
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) <= 0:
                raise ExtendedModelError(f"{model_id} parameter {name} must be a positive integer")
            values[name] = int(value)
    if model_id in DEEP_MODEL_IDS and values["max_epochs"] > 100:
        raise ExtendedModelError(f"{model_id} parameter max_epochs cannot exceed 100")
    for name in ("random_state",):
        if name in values:
            value = values[name]
            if value is None:
                continue
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) < 0:
                raise ExtendedModelError(f"{model_id} parameter {name} must be a non-negative integer")
            if model_id in HMM_MODEL_IDS | DEEP_MODEL_IDS and int(value) > _MAX_RANDOM_STATE:
                raise ExtendedModelError(f"{model_id} parameter {name} must be between 0 and {_MAX_RANDOM_STATE}")
            values[name] = int(value)
    if model_id in HMM_MODEL_IDS:
        for name, options in (
            ("algorithm", {"viterbi", "map"}),
            ("implementation", {"log", "scaling"}),
        ):
            if name in values and (not isinstance(values[name], str) or values[name] not in options):
                raise ExtendedModelError(f"{model_id} parameter {name} must be one of {sorted(options)}")
        if "verbose" in values and not isinstance(values["verbose"], bool):
            raise ExtendedModelError(f"{model_id} parameter verbose must be a boolean")
        emission_parameters = {
            "H01": set("stmc"),
            "H02": set("stmcw"),
            "H03": set("ste"),
        }[model_id]
        for name in ("params", "init_params"):
            if name in values:
                value = values[name]
                if not isinstance(value, str) or not set(value).issubset(emission_parameters):
                    allowed_letters = "".join(sorted(emission_parameters))
                    raise ExtendedModelError(
                        f"{model_id} parameter {name} must contain only letters from {allowed_letters!r}"
                    )
    for name in ("tol", "min_covar", "learning_rate", "weight_decay"):
        if name in values:
            value = values[name]
            if isinstance(value, bool) or not isinstance(value, (int, float, np.number)) or not math.isfinite(float(value)):
                raise ExtendedModelError(f"{model_id} parameter {name} must be a finite number")
            must_be_positive = name in {"tol", "min_covar", "learning_rate"}
            if (must_be_positive and float(value) <= 0) or (not must_be_positive and float(value) < 0):
                qualifier = "positive" if must_be_positive else "non-negative"
                raise ExtendedModelError(f"{model_id} parameter {name} must be {qualifier}")
            values[name] = float(value)
    for name, options in choices.items():
        if values[name] not in options:
            raise ExtendedModelError(f"{model_id} parameter {name} must be one of {sorted(options)}")
    return values


def resolve_deep_seed(parameters: Mapping[str, Any], fallback_seed: Any) -> int:
    """Resolve the validated deep-model seed, falling back to the split seed."""
    requested = parameters.get("random_state")
    selected = fallback_seed if requested is None else requested
    if (
        isinstance(selected, bool)
        or not isinstance(selected, (int, np.integer))
        or int(selected) < 0
        or int(selected) > _MAX_RANDOM_STATE
    ):
        raise ExtendedModelError(f"Deep training seed must be an integer between 0 and {_MAX_RANDOM_STATE}")
    return int(selected)


def resolve_hmm_seed(parameters: Mapping[str, Any], fallback_seed: Any) -> int:
    """Resolve the HMM seed, using the split seed when random_state is omitted or None."""
    requested = parameters.get("random_state")
    selected = fallback_seed if requested is None else requested
    if (
        isinstance(selected, bool)
        or not isinstance(selected, (int, np.integer))
        or int(selected) < 0
        or int(selected) > _MAX_RANDOM_STATE
    ):
        raise ExtendedModelError(f"HMM training seed must be an integer between 0 and {_MAX_RANDOM_STATE}")
    return int(selected)


def build_hmm_estimator(
    model_id: str,
    parameters: Mapping[str, Any] | None = None,
    *,
    seed: int = 42,
    categorical_n_features: int | None = None,
) -> Any:
    """Create an hmmlearn estimator lazily with an explicit reproducible seed."""
    if model_id not in HMM_MODEL_IDS:
        raise ExtendedModelError(f"{model_id!r} is not an HMM model")
    values = validate_extended_parameters(model_id, parameters)
    values["random_state"] = resolve_hmm_seed(values, seed)
    if model_id == "H03":
        if isinstance(categorical_n_features, bool) or not isinstance(categorical_n_features, int) or categorical_n_features <= 0:
            raise ExtendedModelError("H03 requires the frozen train-only categorical_n_features")
        values["n_features"] = categorical_n_features
    try:
        hmm = importlib.import_module("hmmlearn.hmm")
    except ImportError as exc:
        raise OptionalModelDependencyError("HMM models require the 'sequence' extra (hmmlearn)") from exc
    constructor = {
        "H01": hmm.GaussianHMM,
        "H02": hmm.GMMHMM,
        "H03": hmm.CategoricalHMM,
    }[model_id]
    try:
        return constructor(**values)
    except (TypeError, ValueError) as exc:
        raise ExtendedModelError(f"Invalid {model_id} parameters: {exc}") from exc


def hmm_lengths(partition: Any) -> list[int]:
    """Return checked sequence lengths for every HMM operation."""
    lengths = np.asarray(getattr(partition, "lengths", None), dtype=np.int64)
    observations = getattr(partition, "observations", None)
    if observations is None or observations.ndim != 2:
        raise ExtendedModelError("HMM partition observations must have shape (observations, features)")
    if lengths.ndim != 1 or len(lengths) == 0 or np.any(lengths <= 0) or int(lengths.sum()) != len(observations):
        raise ExtendedModelError("HMM partition lengths must be positive and sum to its observation count")
    return [int(item) for item in lengths]


def hmm_observations(model_id: str, partition: Any, *, allow_raw: bool = False) -> np.ndarray:
    """Return model-shaped observations; unchecked raw partitions are final-test only."""
    if model_id not in HMM_MODEL_IDS:
        raise ExtendedModelError(f"{model_id!r} is not an HMM model")
    encoding = getattr(partition, "observation_encoding", None)
    if encoding in {"continuous_raw_unchecked", "categorical_raw_unchecked"} and not allow_raw:
        raise ExtendedModelError(f"Raw {getattr(partition, 'name', 'partition')} observations cannot be used before final evaluation")
    raw = getattr(partition, "observations", None)
    if raw is None:
        raise ExtendedModelError("HMM partition is missing observations")
    try:
        if model_id == "H03":
            result = np.asarray(raw, dtype=np.int64)
            if result.ndim != 2 or result.shape[1] != 1 or np.any(result < 0):
                raise ExtendedModelError("H03 observations must be encoded integer symbols with shape (n, 1)")
        else:
            result = np.asarray(raw, dtype=np.float32)
            if result.ndim != 2 or result.shape[1] < 1 or not np.isfinite(result).all():
                raise ExtendedModelError(f"{model_id} observations must be finite float32 with shape (n, features)")
    except (TypeError, ValueError, OverflowError) as exc:
        raise ExtendedModelError(f"{model_id} observations have an invalid emission value") from exc
    hmm_lengths(partition)
    return result


def fit_hmm(estimator: Any, model_id: str, partition: Any, *, allow_raw: bool = False) -> Any:
    """Fit an HMM while always forwarding the partition's exact lengths."""
    observations = hmm_observations(model_id, partition, allow_raw=allow_raw)
    lengths = hmm_lengths(partition)
    estimator.fit(observations, lengths=lengths)
    return estimator


def score_hmm(estimator: Any, model_id: str, partition: Any, *, allow_raw: bool = False) -> float:
    observations = hmm_observations(model_id, partition, allow_raw=allow_raw)
    lengths = hmm_lengths(partition)
    total = float(estimator.score(observations, lengths=lengths))
    value = total / int(lengths and sum(lengths))
    if not math.isfinite(value):
        raise ExtendedModelError("HMM log_likelihood_per_observation is not finite")
    return value


def hmm_test_outputs(
    estimator: Any,
    model_id: str,
    partition: Any,
    *,
    categorical_alphabet: tuple[Any, ...] | list[Any] | None = None,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Return final-only log likelihood, decoded states, and posterior rows."""
    lengths = hmm_lengths(partition)
    if model_id == "H03":
        # The alphabet is deliberately checked only after the caller has frozen
        # the winner and entered the single final test evaluation.
        if partition.observation_encoding == "categorical_raw_unchecked":
            if categorical_alphabet is None:
                raise ExtendedModelError("H03 final test evaluation requires the frozen train-only alphabet")
            from .sequence import _json_text, _normal_scalar

            alphabet = {
                _json_text(value): index
                for index, value in enumerate(categorical_alphabet)
            }
            encoded: list[int] = []
            for position, value in enumerate(np.asarray(partition.observations, dtype=object).reshape(-1)):
                safe = _normal_scalar(value, context=f"test categorical observation at position {position}")
                key = _json_text(safe)
                if key not in alphabet:
                    raise ExtendedModelError(
                        f"H03 final test observation {safe!r} at position {position} is unknown to the frozen train-only alphabet"
                    )
                encoded.append(alphabet[key])
            observations = np.asarray(encoded, dtype=np.int64).reshape(-1, 1)
        else:
            observations = hmm_observations(model_id, partition)
    else:
        observations = hmm_observations(model_id, partition, allow_raw=True)
    total = float(estimator.score(observations, lengths=lengths))
    per_observation = total / sum(lengths)
    _, states = estimator.decode(observations, lengths=lengths)
    posterior = estimator.predict_proba(observations, lengths=lengths)
    if not math.isfinite(per_observation) or len(states) != len(observations) or posterior.shape[0] != len(observations):
        raise ExtendedModelError("HMM returned invalid final test output")
    return per_observation, np.asarray(states), np.asarray(posterior)


def validate_deep_inputs(model_id: str, values: Any, *, name: str = "inputs") -> np.ndarray:
    """Validate required tabular or (batch, window, features) float32 shapes."""
    if model_id not in DEEP_MODEL_IDS:
        raise ExtendedModelError(f"{model_id!r} is not a neural model")
    try:
        result = np.asarray(values, dtype=np.float32)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ExtendedModelError(f"{model_id} {name} must be numeric and fit float32") from exc
    required_ndim = 3 if model_id in WINDOW_DEEP_MODEL_IDS else 2
    if result.ndim != required_ndim:
        shape = "(batch, window, features)" if required_ndim == 3 else "(batch, features)"
        raise ExtendedModelError(f"{model_id} {name} must have shape {shape}; received {result.shape}")
    if any(size <= 0 for size in result.shape):
        raise ExtendedModelError(f"{model_id} {name} dimensions must be positive")
    if not np.isfinite(result).all():
        raise ExtendedModelError(f"{model_id} {name} must be finite")
    # SequencePlan arrays are intentionally read-only.  Always hand the deep
    # estimator its own writable buffer so PyTorch never wraps a read-only
    # NumPy view as a tensor.
    return np.array(result, dtype=np.float32, order="C", copy=True)


def validate_deep_targets(model_id: str, values: Any, *, class_count: int | None = None, name: str = "targets") -> np.ndarray:
    if model_id == "N01":
        raw = np.asarray(values)
        if raw.ndim != 1 or len(raw) == 0 or not np.issubdtype(raw.dtype, np.integer):
            raise ExtendedModelError("N01 targets must be a non-empty one-dimensional encoded integer array")
        result = raw.astype(np.int64, copy=False)
        if np.any(result < 0) or (class_count is not None and np.any(result >= class_count)):
            raise ExtendedModelError("N01 target class IDs are outside the frozen train-only mapping")
        if class_count is not None and class_count < 2:
            raise ExtendedModelError("N01 training requires at least two train-only classes")
        return np.array(result, dtype=np.int64, order="C", copy=True)
    try:
        result = np.asarray(values, dtype=np.float32)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ExtendedModelError(f"{model_id} {name} must be numeric and fit float32") from exc
    if result.ndim == 1:
        result = result.reshape(-1, 1)
    if result.ndim != 2 or result.shape[1] != 1 or len(result) == 0 or not np.isfinite(result).all():
        raise ExtendedModelError(f"{model_id} {name} must be finite float32 with shape (n, 1)")
    return np.array(result, dtype=np.float32, order="C", copy=True)


def _torch_module_type() -> tuple[Any, type]:
    global _TORCH_MODULE_TYPE
    try:
        torch = importlib.import_module("torch")
    except ImportError as exc:
        raise OptionalModelDependencyError("Neural models require the 'deep' extra (torch and skorch)") from exc
    if _TORCH_MODULE_TYPE is None:
        class GenericTorchModule(torch.nn.Module):
            def __init__(
                self,
                *,
                architecture: str,
                input_size: int,
                output_size: int,
                hidden_size: int = 32,
                num_layers: int = 1,
            ):
                super().__init__()
                if architecture not in {"mlp", "lstm", "gru"}:
                    raise ValueError(f"Unknown architecture {architecture!r}")
                self.architecture = architecture
                if architecture == "mlp":
                    hidden_layers = [torch.nn.Linear(input_size, hidden_size)]
                    for _ in range(1, num_layers):
                        hidden_layers.extend(
                            [
                                torch.nn.ReLU(),
                                torch.nn.Linear(hidden_size, hidden_size),
                            ]
                        )
                    self.backbone = torch.nn.Sequential(*hidden_layers)
                    self.recurrent = None
                elif architecture == "lstm":
                    self.backbone = None
                    self.recurrent = torch.nn.LSTM(
                        input_size=input_size,
                        hidden_size=hidden_size,
                        num_layers=num_layers,
                        batch_first=True,
                    )
                else:
                    self.backbone = None
                    self.recurrent = torch.nn.GRU(
                        input_size=input_size,
                        hidden_size=hidden_size,
                        num_layers=num_layers,
                        batch_first=True,
                    )
                self.head = torch.nn.Sequential(torch.nn.ReLU(), torch.nn.Linear(hidden_size, output_size))

            def forward(self, values):
                if self.architecture == "mlp":
                    features = self.backbone(values)
                else:
                    outputs, _ = self.recurrent(values)
                    features = outputs[:, -1, :]
                return self.head(features)

        GenericTorchModule.__module__ = __name__
        GenericTorchModule.__qualname__ = "GenericTorchModule"
        globals()["GenericTorchModule"] = GenericTorchModule
        _TORCH_MODULE_TYPE = GenericTorchModule
    return torch, _TORCH_MODULE_TYPE


@contextmanager
def deep_cpu_context(seed: int | None = None) -> Iterator[Any]:
    """Serialize torch-global RNG/thread changes and restore caller state."""
    with _DEEP_TRAIN_LOCK:
        torch, _ = _torch_module_type()
        python_state = random.getstate()
        numpy_state = np.random.get_state()
        torch_state = torch.random.get_rng_state().clone()
        thread_count = int(torch.get_num_threads())
        try:
            torch.set_num_threads(1)
            if seed is not None:
                random.seed(int(seed))
                np.random.seed(int(seed) % (2**32))
                torch.manual_seed(int(seed))
            yield torch
        finally:
            torch.random.set_rng_state(torch_state)
            np.random.set_state(numpy_state)
            random.setstate(python_state)
            if int(torch.get_num_threads()) != thread_count:
                torch.set_num_threads(thread_count)


def _deep_estimator(
    model_id: str,
    params: Mapping[str, Any],
    *,
    input_size: int,
    output_size: int,
    train_split: Any,
    max_epochs: int,
    callbacks: list[Any],
) -> Any:
    torch, module_type = _torch_module_type()
    try:
        regressor = importlib.import_module("skorch.regressor").NeuralNetRegressor
        classifier = importlib.import_module("skorch.classifier").NeuralNetClassifier
        cls = classifier if model_id == "N01" else regressor
    except ImportError as exc:
        raise OptionalModelDependencyError("Neural models require the 'deep' extra (torch and skorch)") from exc
    architecture = {"N01": "mlp", "N02": "mlp", "N04": "lstm", "N06": "gru"}[model_id]
    return cls(
        module=module_type,
        module__architecture=architecture,
        module__input_size=int(input_size),
        module__output_size=int(output_size),
        module__hidden_size=int(params["hidden_size"]),
        module__num_layers=int(params["num_layers"]),
        criterion=torch.nn.CrossEntropyLoss if model_id == "N01" else torch.nn.MSELoss,
        optimizer=torch.optim.Adam,
        lr=float(params["learning_rate"]),
        optimizer__weight_decay=float(params["weight_decay"]),
        batch_size=int(params["batch_size"]),
        max_epochs=int(max_epochs),
        iterator_train__shuffle=True,
        train_split=train_split,
        callbacks=callbacks,
        device="cpu",
        verbose=0,
    )


def _state_dict_numpy(estimator: Any) -> dict[str, np.ndarray]:
    return {
        str(name): value.detach().cpu().numpy().copy()
        for name, value in estimator.module_.state_dict().items()
    }


def _training_curves_from_history(history: Any, model_id: str, fit_scope: str) -> dict[str, Any]:
    """Copy actual skorch epoch metrics into a JSON-safe, ordered record."""
    if fit_scope not in {"train", "train_validation"}:
        raise ExtendedModelError("Deep training-curve fit_scope must be 'train' or 'train_validation'")
    epochs: list[int] = []
    train_loss: list[float] = []
    validation_loss: list[float] = []
    rows = list(history)
    if not rows:
        raise ExtendedModelError("Deep fit produced no skorch history rows")
    for position, row in enumerate(rows, start=1):
        if not isinstance(row, Mapping):
            raise ExtendedModelError("Deep skorch history rows must be mappings")
        epoch = row.get("epoch")
        if isinstance(epoch, bool) or not isinstance(epoch, (int, np.integer)) or int(epoch) != position:
            raise ExtendedModelError("Deep skorch history epochs must be consecutive positive integers")
        epochs.append(int(epoch))
        value = row.get("train_loss")
        if isinstance(value, bool) or not isinstance(value, (int, float, np.number)) or not math.isfinite(float(value)):
            raise ExtendedModelError("Deep skorch history train_loss must be finite at every epoch")
        train_loss.append(float(value))
        if fit_scope == "train":
            value = row.get("valid_loss")
            if isinstance(value, bool) or not isinstance(value, (int, float, np.number)) or not math.isfinite(float(value)):
                raise ExtendedModelError("Deep skorch history valid_loss must be finite at every validation epoch")
            validation_loss.append(float(value))
    return {
        "schema_version": 1,
        "model_id": model_id,
        "fit_scope": fit_scope,
        "epochs": epochs,
        "train_loss": train_loss,
        "validation_loss": validation_loss if fit_scope == "train" else None,
        "validation_available": fit_scope == "train",
    }


def fit_deep_validation(
    model_id: str,
    parameters: Mapping[str, Any] | None,
    *,
    seed: int,
    train_x: Any,
    train_y: Any,
    validation_x: Any,
    validation_y: Any,
    class_count: int | None = None,
) -> DeepFitResult:
    """Fit with a fixed validation dataset and restore best validation weights."""
    if model_id not in DEEP_MODEL_IDS:
        raise ExtendedModelError(f"{model_id!r} is not a neural model")
    params = validate_extended_parameters(model_id, parameters)
    if model_id == "N01" and (isinstance(class_count, bool) or not isinstance(class_count, int) or class_count < 2):
        raise ExtendedModelError("N01 requires the frozen train-only class_count")
    x_train = validate_deep_inputs(model_id, train_x, name="train inputs")
    x_validation = validate_deep_inputs(model_id, validation_x, name="validation inputs")
    y_train = validate_deep_targets(model_id, train_y, class_count=class_count, name="train targets")
    y_validation = validate_deep_targets(model_id, validation_y, class_count=class_count, name="validation targets")
    if len(x_train) != len(y_train) or len(x_validation) != len(y_validation):
        raise ExtendedModelError(f"{model_id} input and target row counts do not match")
    if x_train.shape[1:] != x_validation.shape[1:]:
        raise ExtendedModelError(f"{model_id} training and validation input dimensions differ")
    try:
        from skorch.dataset import Dataset
        from skorch.callbacks import EarlyStopping
        from skorch.helper import predefined_split
    except ImportError as exc:
        raise OptionalModelDependencyError("Neural models require the 'deep' extra (torch and skorch)") from exc
    validation_set = Dataset(x_validation, y_validation)
    stopping = EarlyStopping(
        monitor="valid_loss",
        patience=params["patience"],
        threshold=0.0,
        threshold_mode="abs",
        load_best=True,
        lower_is_better=True,
    )
    net = None
    with deep_cpu_context(seed):
        net = _deep_estimator(
            model_id,
            params,
            input_size=x_train.shape[-1],
            output_size=class_count if model_id == "N01" else 1,
            train_split=predefined_split(validation_set),
            max_epochs=params["max_epochs"],
            callbacks=[("early_stopping", stopping)],
        )
        net.fit(x_train, y_train)
        training_curves = _training_curves_from_history(net.history, model_id, "train")
        valid_losses = list(zip(training_curves["validation_loss"], training_curves["epochs"]))
        early_stopping = next(
            (callback for name, callback in net.callbacks_ if name == "early_stopping"),
            None,
        )
        callback_epoch = getattr(early_stopping, "best_epoch_", None)
        if isinstance(callback_epoch, bool) or not isinstance(callback_epoch, (int, np.integer)):
            raise ExtendedModelError("EarlyStopping did not record a best_epoch_ for the restored validation weights")
        best_epoch = int(callback_epoch)
        best_row = next((loss for loss, epoch in valid_losses if epoch == best_epoch), None)
        if best_row is None:
            raise ExtendedModelError("EarlyStopping best_epoch_ is absent from the finite valid_loss history")
        # A zero absolute threshold makes the callback's best_epoch_ rule match
        # strict finite argmin, with ties retaining the first best epoch.
        net._pyml_best_state_dict = _state_dict_numpy(net)
    return DeepFitResult(
        estimator=net,
        selected_epochs=best_epoch,
        best_epoch=best_epoch,
        best_weight_policy="early_stopping_abs_threshold_zero_load_best_valid_loss",
        training_curves=training_curves,
    )


def refit_deep_fixed_epochs(
    model_id: str,
    parameters: Mapping[str, Any] | None,
    *,
    seed: int,
    train_x: Any,
    train_y: Any,
    selected_epochs: int,
    class_count: int | None = None,
) -> Any:
    """Fit a fresh CPU module/optimizer for fixed epochs without a validation split."""
    if model_id not in DEEP_MODEL_IDS:
        raise ExtendedModelError(f"{model_id!r} is not a neural model")
    params = validate_extended_parameters(model_id, parameters)
    if model_id == "N01" and (isinstance(class_count, bool) or not isinstance(class_count, int) or class_count < 2):
        raise ExtendedModelError("N01 requires the frozen train-only class_count")
    if isinstance(selected_epochs, bool) or not isinstance(selected_epochs, (int, np.integer)):
        raise ExtendedModelError("selected_epochs must be an integer from the frozen provenance")
    if not 1 <= int(selected_epochs) <= params["max_epochs"]:
        raise ExtendedModelError("selected_epochs must be between 1 and max_epochs")
    x_train = validate_deep_inputs(model_id, train_x, name="refit inputs")
    y_train = validate_deep_targets(model_id, train_y, class_count=class_count, name="refit targets")
    if len(x_train) != len(y_train):
        raise ExtendedModelError(f"{model_id} refit input and target row counts do not match")
    with deep_cpu_context(seed):
        net = _deep_estimator(
            model_id,
            params,
            input_size=x_train.shape[-1],
            output_size=class_count if model_id == "N01" else 1,
            train_split=None,
            max_epochs=int(selected_epochs),
            callbacks=[],
        )
        net.fit(x_train, y_train)
        net._pyml_training_curves = _training_curves_from_history(
            net.history, model_id, "train_validation"
        )
    return net


def deep_estimator_state(estimator: Any) -> dict[str, np.ndarray]:
    """Return a detached CPU state dict suitable for the raw-data-free model bundle."""
    with deep_cpu_context():
        return _state_dict_numpy(estimator)


def build_deep_from_state(
    model_id: str,
    parameters: Mapping[str, Any],
    *,
    input_size: int,
    output_size: int,
    state_dict: Mapping[str, Any],
) -> Any:
    """Rebuild a CPU inference module without retaining a skorch optimizer."""
    params = validate_extended_parameters(model_id, parameters)
    with deep_cpu_context():
        _, module_type = _torch_module_type()
        module = module_type(
            architecture={"N01": "mlp", "N02": "mlp", "N04": "lstm", "N06": "gru"}[model_id],
            input_size=int(input_size),
            output_size=int(output_size),
            hidden_size=params["hidden_size"],
            num_layers=params["num_layers"],
        )
        torch = importlib.import_module("torch")
        converted = {
            str(name): torch.as_tensor(np.asarray(value), device="cpu")
            for name, value in state_dict.items()
        }
        module.load_state_dict(converted, strict=True)
        module.eval()
    return module


def predict_deep(estimator: Any, model_id: str, values: Any, *, class_count: int | None = None) -> np.ndarray:
    """Predict with a skorch estimator or raw CPU module under the deep lock."""
    inputs = validate_deep_inputs(model_id, values, name="inference inputs")
    with deep_cpu_context():
        module = getattr(estimator, "module_", None)
        if module is None and hasattr(estimator, "state_dict"):
            module = estimator
        if module is not None:
            torch, _ = _torch_module_type()
            was_training = bool(module.training)
            module.eval()
            with torch.no_grad():
                result = module(torch.as_tensor(inputs, dtype=torch.float32, device="cpu")).detach().cpu().numpy()
            if was_training:
                module.train()
        elif callable(getattr(estimator, "predict", None)):
            result = np.asarray(estimator.predict(inputs))
        else:
            raise ExtendedModelError("Deep estimator does not expose a fitted CPU module or predict method")
    if model_id == "N01":
        if result.ndim != 2 or result.shape[1] != class_count:
            if result.ndim == 1 and len(result) == len(inputs):
                return result
            raise ExtendedModelError("N01 model output does not match the frozen class mapping")
        return result
    if result.ndim == 1:
        result = result.reshape(-1, 1)
    if result.ndim != 2 or result.shape[1] != 1 or not np.isfinite(result).all():
        raise ExtendedModelError(f"{model_id} model returned invalid regression predictions")
    return result.astype(np.float32, copy=False)


def predict_proba_deep(estimator: Any, model_id: str, values: Any, *, class_count: int) -> np.ndarray:
    if model_id != "N01":
        raise ExtendedModelError("predict_proba is available only for N01")
    logits = predict_deep(estimator, model_id, values, class_count=class_count)
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=1, keepdims=True)


def __getattr__(name: str) -> Any:
    """Resolve a torch-backed class only when a saved deep session needs it."""
    if name == "GenericTorchModule":
        _, module_type = _torch_module_type()
        return module_type
    raise AttributeError(name)

