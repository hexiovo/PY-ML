"""Budgeted validation search using genuine upstream optimization adapters."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from importlib.util import find_spec
import json
import os
from pathlib import Path
import re
import tempfile
import time
import traceback
from typing import Any
import uuid

import joblib
import numpy as np
from threadpoolctl import threadpool_limits

from .catalog import build_estimator, get_model
from .config import ExperimentConfig
from .experiment import DatasetSnapshot, ExperimentSession, _canonical_config, _json_digest, _session_training_curves, build_extended_snapshot, build_snapshot, prepare_experiment
from .objectives import ObjectiveSpec, score_session
from .sequence import load_owned_sequence_plan, save_owned_sequence_plan
from .sequence_reporting import sequence_plan_summary
from .search_space import SearchSpace, json_copy
from .sequence_models import EXTENDED_MODEL_IDS, HMM_MODEL_IDS, validate_extended_parameters

_ERROR_ID_RE = re.compile(r"^[0-9a-f]{32}$")


def atomic_joblib(path, value):
    path = Path(path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    try:
        joblib.dump(value, temporary, compress=3)
        with open(temporary, "r+b") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _atomic_session_plot_cache_manifest(session_path, session):
    from .plot_cache import (
        build_session_plot_cache_manifest,
        session_plot_cache_manifest_path,
    )

    manifest_path = session_plot_cache_manifest_path(session_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest = build_session_plot_cache_manifest(session)
    descriptor, temporary = tempfile.mkstemp(
        prefix=manifest_path.name + ".", suffix=".tmp", dir=manifest_path.parent
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(manifest, stream, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, manifest_path)
    finally:
        Path(temporary).unlink(missing_ok=True)


@dataclass
class SearchSpec:
    method: str = "grid"
    space: SearchSpace = field(default_factory=SearchSpace)
    objective: ObjectiveSpec = field(default_factory=ObjectiveSpec)
    max_fits: int = 50
    timeout_seconds: float = 1200.
    max_proposals: int = 250
    seed: int = 42

    def __post_init__(self):
        self.validate()

    def validate(self):
        self.space = SearchSpace.from_dict(self.space)
        self.objective = ObjectiveSpec.from_dict(self.objective)
        if self.method not in {"grid", "genetic", "annealing", "random", "tpe"}:
            raise ValueError("Unknown search method")
        if not self.space.fields and self.method != "grid":
            raise ValueError("Fixed-only parameter runs require method='grid'")
        optional_module = {"tpe": "optuna", "genetic": "pymoo"}.get(self.method)
        if optional_module is not None and find_spec(optional_module) is None:
            method_label = "TPE" if self.method == "tpe" else "genetic search"
            raise ValueError(
                f"{method_label} requires optional dependency '{optional_module}'; "
                "install the search extra with `uv sync --extra search` or "
                "`pip install pyml-workbench[search]`"
            )
        if any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in (self.max_fits, self.max_proposals)) or not np.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0 or not isinstance(self.seed, int) or self.seed < 0:
            raise ValueError("Search budgets and seed must be valid finite values")
        return self

    @classmethod
    def from_dict(cls, payload):
        if isinstance(payload, cls):
            payload = payload.to_dict()
        if not isinstance(payload, dict):
            raise ValueError("Search spec must be an object")
        data = dict(payload)
        data["space"] = SearchSpace.from_dict(data.get("space", {}))
        data["objective"] = ObjectiveSpec.from_dict(data.get("objective"))
        if "max_proposals" not in data:
            data["max_proposals"] = int(data.get("max_fits", 50)) * 5
        return cls(**data)

    def to_dict(self):
        return dict(method=self.method, space=self.space.to_dict(), objective=self.objective.to_dict(), max_fits=self.max_fits, timeout_seconds=self.timeout_seconds, max_proposals=self.max_proposals, seed=self.seed)


@dataclass
class TrialRecord:
    trial_id: str
    proposal_index: int
    fit_index: int | None
    status: str
    parameters: dict
    config_sha256: str | None = None
    cache_key: str | None = None
    objective_value: float | None = None
    direction: str = "max"
    metric: str = "balanced_accuracy"
    score_split: str = "validation"
    metrics: dict = field(default_factory=dict)
    duration_seconds: float = 0.
    cache_hit: bool = False
    error: str | None = None
    test_evaluation_count: int = 0
    session_path: str | None = None
    cached_from: str | None = None
    training_curves: dict[str, Any] | None = None
    error_id: str | None = None
    traceback: str | None = None

    def to_dict(self):
        return json_copy(asdict(self))

    @classmethod
    def from_dict(cls, value):
        return cls(**json_copy(value))


@dataclass
class SearchResume:
    records: list = field(default_factory=list)
    actual_fit_count: int = 0
    proposal_count: int = 0
    elapsed_seconds: float = 0.
    best_session_path: str | None = None
    snapshot_path: str | None = None
    fingerprint: str | None = None
    sequence_plan_path: str | None = None
    sequence_plan_receipt: dict | None = None
    sequence_plan_manifest: dict | None = None

    @classmethod
    def from_dict(cls, payload):
        if isinstance(payload, cls):
            return payload
        return cls(records=payload.get("trials", payload.get("records", [])), actual_fit_count=payload.get("actual_fit_count", 0), proposal_count=payload.get("proposal_count", 0), elapsed_seconds=payload.get("elapsed_seconds", 0.), best_session_path=payload.get("best_session_path"), snapshot_path=payload.get("snapshot_path"), fingerprint=payload.get("fingerprint"), sequence_plan_path=payload.get("sequence_plan_path"), sequence_plan_receipt=payload.get("sequence_plan_receipt"), sequence_plan_manifest=payload.get("sequence_plan_manifest"))


@dataclass
class SearchResult:
    job_id: str
    status: str
    stop_reason: str
    spec: SearchSpec
    config: ExperimentConfig
    snapshot: DatasetSnapshot
    trials: list[TrialRecord]
    winner: TrialRecord | None
    winner_session: ExperimentSession | None
    best_validation: dict | None
    actual_fit_count: int
    proposal_count: int
    elapsed_seconds: float
    fingerprint: str
    snapshot_path: str | None = None
    best_session_path: str | None = None
    test_evaluation_count: int = 0
    resume_mode: str = "restart_sampler_with_cache"
    sequence_plan: Any | None = None
    sequence_plan_path: str | None = None
    sequence_plan_receipt: dict | None = None

    def to_dict(self):
        return json_copy(dict(job_id=self.job_id, status=self.status, stop_reason=self.stop_reason, spec=self.spec.to_dict(), config=self.config.to_dict(), snapshot=self.snapshot.to_dict(), trials=[item.to_dict() for item in self.trials], winner=self.winner.to_dict() if self.winner else None, best_validation=self.best_validation, actual_fit_count=self.actual_fit_count, proposal_count=self.proposal_count, elapsed_seconds=self.elapsed_seconds, fingerprint=self.fingerprint, snapshot_path=self.snapshot_path, best_session_path=self.best_session_path, test_evaluation_count=0, resume_mode=self.resume_mode, sequence_plan_manifest=self.sequence_plan.to_dict() if self.sequence_plan is not None else None, sequence_plan_summary=sequence_plan_summary(self.sequence_plan), sequence_plan_path=self.sequence_plan_path, sequence_plan_receipt=self.sequence_plan_receipt))


class _StopSearch(Exception):
    pass


class _SearchEngine:
    def __init__(self, config, spec, snapshot, job_id, on_event, should_cancel, wait_for_dispatch, resume, artifact_dir, sequence_plan=None, error_handler=None):
        self.config, self.spec, self.snapshot, self.job_id = config, spec, snapshot, job_id
        self.sequence_plan = sequence_plan
        self.on_event, self.should_cancel, self.wait_for_dispatch = on_event, should_cancel, wait_for_dispatch
        self.error_handler = error_handler
        self.objective = spec.objective.resolve(config.task)
        spec.objective = self.objective
        self.fingerprint = _json_digest({"config": replace(config, output_dir=None).to_dict(), "space": spec.space.to_dict(), "objective": self.objective.to_dict(), "seed": spec.seed, "method": spec.method, "snapshot": snapshot.to_dict()})
        self.trials = []
        self.actual_fit_count = self.proposal_count = 0
        self.previous_elapsed = 0.
        self.started, self.paused_seconds = time.monotonic(), 0.
        self.cache, self.winner, self.winner_session = {}, None, None
        self.snapshot_path = self.best_session_path = None
        self.sequence_plan_path = None
        self.sequence_plan_receipt = None
        self.directory = Path(artifact_dir).expanduser().resolve() if artifact_dir else None
        self.stop_reason = "space_exhausted"
        if self.directory:
            source = Path(snapshot.source_path).resolve()
            if self.directory in {source, source.parent}:
                raise ValueError("Search artifact directory must be separate from input and its parent")
            self.directory.mkdir(parents=True, exist_ok=True)
        if resume is not None:
            resume = SearchResume.from_dict(resume)
            if resume.fingerprint != self.fingerprint or any(value < 0 for value in (resume.actual_fit_count, resume.proposal_count, resume.elapsed_seconds)):
                raise ValueError("Resume history does not match this search fingerprint/budget")
            expected_plan = self.sequence_plan.to_dict() if self.sequence_plan is not None else None
            if resume.sequence_plan_manifest != expected_plan:
                raise ValueError("Resume history sequence plan does not match this search")
            self.sequence_plan_path = resume.sequence_plan_path
            self.sequence_plan_receipt = resume.sequence_plan_receipt
            self.trials = [TrialRecord.from_dict(item.to_dict() if isinstance(item, TrialRecord) else item) for item in resume.records]
            self.actual_fit_count, self.proposal_count, self.previous_elapsed = resume.actual_fit_count, resume.proposal_count, resume.elapsed_seconds
            if self.actual_fit_count != max((item.fit_index or 0 for item in self.trials), default=0) or self.proposal_count != max((item.proposal_index for item in self.trials), default=0):
                raise ValueError("Resume counters do not match persisted trial receipts")
            for item in self.trials:
                if item.test_evaluation_count:
                    raise ValueError("Search history contains test evaluation")
                if item.cache_key:
                    digest = _canonical_config(replace(config, parameters=item.parameters, output_dir=None))[1]
                    if item.config_sha256 != digest or item.cache_key != _json_digest({"search": self.fingerprint, "config": digest}) or item.metric != self.objective.metric or item.direction != self.objective.direction or item.score_split != self.objective.split:
                        raise ValueError("Resume trial belongs to another configuration/objective")
                if item.cache_key and not item.cache_hit:
                    self.cache[item.cache_key] = item
                if item.status == "complete" and self.better(item):
                    self.winner = item
            self.best_session_path = resume.best_session_path or (self.winner.session_path if self.winner else None)
            if self.winner:
                if not self.best_session_path:
                    raise ValueError("Resume winner has no persisted fitted session")
                self.winner_session = joblib.load(self.best_session_path)
                self.verify_session(self.winner_session, self.winner)
                from .plot_cache import (
                    _verify_session_plot_cache_manifest,
                    session_plot_cache_manifest_path,
                )

                manifest_path = session_plot_cache_manifest_path(self.best_session_path)
                if manifest_path.is_file():
                    try:
                        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                        _verify_session_plot_cache_manifest(
                            {"plot_cache_manifest": manifest}, self.winner_session
                        )
                    except Exception as exc:
                        raise ValueError(
                            f"Resume winner plot-cache manifest is invalid: {exc}"
                        ) from exc
        if self.directory:
            snapshot_file = self.directory / "snapshot.joblib"
            if snapshot_file.exists():
                stored = joblib.load(snapshot_file)
                if not isinstance(stored, DatasetSnapshot):
                    raise ValueError("Existing snapshot artifact has invalid type")
                stored.verify(config)
                if stored.to_dict() != snapshot.to_dict():
                    raise ValueError("Existing artifact directory belongs to another snapshot")
            else:
                atomic_joblib(snapshot_file, snapshot)
            self.snapshot_path = str(snapshot_file)
            if self.sequence_plan is not None:
                plan_file = self.directory / "sequence_plan.joblib"
                receipt_file = self.directory / "sequence_plan_receipt.json"
                if plan_file.exists() or receipt_file.exists():
                    if not plan_file.is_file() or not receipt_file.is_file():
                        raise ValueError("Existing search sequence plan artifact is incomplete")
                    receipt = json.loads(receipt_file.read_text(encoding="utf-8"))
                    persisted = load_owned_sequence_plan(
                        plan_file,
                        receipt,
                        expected_manifest=self.sequence_plan.to_dict(),
                        config=config,
                        snapshot=snapshot,
                    )
                    if persisted.to_dict() != self.sequence_plan.to_dict():
                        raise ValueError("Existing search directory belongs to another sequence plan")
                    self.sequence_plan_receipt = receipt
                else:
                    self.sequence_plan_receipt = save_owned_sequence_plan(self.sequence_plan, plan_file)
                    receipt_file.write_text(json.dumps(self.sequence_plan_receipt, ensure_ascii=False, indent=2), encoding="utf-8")
                self.sequence_plan_path = str(plan_file)
            elif resume is not None and resume.sequence_plan_path:
                raise ValueError("Resume has a sequence plan but this search does not")

    def verify_session(self, session, record):
        if not isinstance(session, ExperimentSession) or session.frozen or session.finalized or session.test_evaluation_count or "test" in session.metrics:
            raise ValueError("Stored winner is not an untouched validation session")
        if session.frozen_config_sha256 != record.config_sha256 or session.snapshot_manifest != self.snapshot.to_dict():
            raise ValueError("Stored winner does not match history/snapshot")
        expected = replace(self.config, parameters=record.parameters, output_dir=None)
        if _canonical_config(expected)[1] != session.frozen_config_sha256 or session.fitted_model.config != session.frozen_config:
            raise ValueError("Stored winner configuration is inconsistent")
        session_plan = getattr(session, "sequence_plan", None)
        if (session_plan.to_dict() if session_plan is not None else None) != (self.sequence_plan.to_dict() if self.sequence_plan is not None else None):
            raise ValueError("Stored winner sequence plan is inconsistent")

    def elapsed(self):
        return self.previous_elapsed + time.monotonic() - self.started - self.paused_seconds

    def emit(self, name, record=None, **data):
        payload = {"type": "progress", "action": "search", "operation": "search", "job_id": self.job_id, "trial_id": record.trial_id if record else None, "name": name, "data": dict(data, actual_fit_count=self.actual_fit_count, proposal_count=self.proposal_count, elapsed_seconds=self.elapsed())}
        if record:
            payload["data"]["record"] = record.to_dict()
        payload = json_copy(payload)
        if self.on_event:
            self.on_event(payload)

    def better(self, record):
        if record.objective_value is None:
            return False
        if self.winner is None:
            return True
        return record.objective_value > self.winner.objective_value if self.objective.direction == "max" else record.objective_value < self.winner.objective_value

    def gate(self):
        if self.should_cancel and self.should_cancel():
            self.stop_reason = "cancelled"
            raise _StopSearch
        if self.wait_for_dispatch:
            before = time.monotonic()
            allowed = self.wait_for_dispatch()
            self.paused_seconds += time.monotonic() - before
            if not allowed:
                self.stop_reason = "cancelled"
                raise _StopSearch
        if self.should_cancel and self.should_cancel():
            self.stop_reason = "cancelled"
            raise _StopSearch
        if self.actual_fit_count >= self.spec.max_fits:
            self.stop_reason = "fit_limit"
        elif self.proposal_count >= self.spec.max_proposals:
            self.stop_reason = "proposal_limit"
        elif self.elapsed() >= self.spec.timeout_seconds:
            self.stop_reason = "time_limit"
        else:
            return
        raise _StopSearch

    def evaluate(self, candidate):
        self.gate()
        self.proposal_count += 1
        record = TrialRecord(str(uuid.uuid4()), self.proposal_count, None, "invalid", {}, direction=self.objective.direction, metric=self.objective.metric, score_split=self.objective.split)
        before = time.monotonic()
        try:
            record.parameters = self.spec.space.canonical(candidate, self.config.parameters)
            trial_config = replace(self.config, parameters=record.parameters, output_dir=None)
            if trial_config.model_id in EXTENDED_MODEL_IDS:
                validate_extended_parameters(trial_config.model_id, trial_config.parameters)
            else:
                estimator = build_estimator(trial_config.model_id, trial_config.parameters, seed=trial_config.split.seed)
                estimator._validate_params()
            record.config_sha256 = _canonical_config(trial_config)[1]
            record.cache_key = _json_digest({"search": self.fingerprint, "config": record.config_sha256})
        except (ValueError, TypeError) as exc:
            record.error = str(exc)
            self.trials.append(record)
            self.emit("proposal", record)
            return float("inf")
        self.emit("proposal", record)
        if record.cache_key in self.cache:
            cached = self.cache[record.cache_key]
            record.status, record.cache_hit, record.cached_from = "cached", True, cached.trial_id
            record.objective_value, record.metrics, record.error = cached.objective_value, json_copy(cached.metrics), cached.error
            record.training_curves = json_copy(cached.training_curves) if cached.training_curves is not None else None
            record.error_id, record.traceback = cached.error_id, cached.traceback
            self.trials.append(record)
            self.emit("cache_hit", record)
            return self.loss(record)
        self.actual_fit_count += 1
        record.fit_index, record.status = self.actual_fit_count, "running"
        # Callback errors propagate. The persistence owner must commit this receipt
        # before returning; no fit begins if recording the budget fails.
        self.emit("fit_started", record)
        session = None
        try:
            with threadpool_limits(limits=1):
                session = prepare_experiment(trial_config, snapshot=self.snapshot, sequence_plan=self.sequence_plan)
                record.training_curves = _session_training_curves(session)
                record.objective_value, record.metrics = score_session(session, self.objective, self.snapshot)
            if session.test_evaluation_count or "test" in session.metrics:
                raise RuntimeError("Search trial violated test isolation")
            record.status = "complete" if record.objective_value is not None else "unscorable"
        except Exception as exc:
            record.status, record.error = "failed", f"{type(exc).__name__}: {exc}"
            try:
                record.traceback = "".join(
                    traceback.format_exception(type(exc), exc, exc.__traceback__)
                )
            except Exception:
                record.traceback = record.error
            record.error_id = uuid.uuid4().hex
            if self.error_handler is not None:
                try:
                    logged = self.error_handler(
                        exc,
                        context={
                            "stage": "search_trial",
                            "job_id": self.job_id,
                            "trial_id": record.trial_id,
                        },
                    )
                    logged_id = getattr(logged, "error_id", None)
                    if isinstance(logged_id, str) and _ERROR_ID_RE.fullmatch(logged_id):
                        record.error_id = logged_id
                except Exception:
                    # Diagnostics must never change fit, budget, or cancellation behavior.
                    pass
        record.duration_seconds = time.monotonic() - before
        self.trials.append(record)
        self.cache[record.cache_key] = record
        if record.status == "complete" and self.better(record):
            self.winner, self.winner_session = record, session
            if self.directory:
                path = self.directory / "sessions" / f"{record.trial_id}.joblib"
                atomic_joblib(path, session)
                _atomic_session_plot_cache_manifest(path, session)
                record.session_path = self.best_session_path = str(path)
        self.emit("trial_failed" if record.status == "failed" else "trial_completed", record, best_session_path=self.best_session_path)
        return self.loss(record)

    def loss(self, record):
        if record.objective_value is None:
            return float("inf")
        return -record.objective_value if self.objective.direction == "max" else record.objective_value


def _adapter(engine):
    space, spec = engine.spec.space, engine.spec
    rng = np.random.default_rng(spec.seed)
    if spec.method == "grid":
        for candidate in space.grid():
            engine.evaluate(candidate)
    elif spec.method == "random":
        while True:
            engine.evaluate(space.sample(rng))
    elif spec.method == "annealing":
        from scipy.optimize import dual_annealing
        names, bounds, constants = [], [], {}
        for name, field_spec in space.fields.items():
            if field_spec["type"] == "real":
                names.append(name)
                low, high = field_spec["low"], field_spec["high"]
                bounds.append((np.log(low), np.log(high)) if field_spec.get("log") else (low, high))
            else:
                if name not in engine.config.parameters:
                    raise ValueError("Annealing needs explicit fixed values for each integer/choice field")
                constants[name] = engine.config.parameters[name]
                space._check(name, constants[name])
        if not names:
            raise ValueError("Annealing requires at least one real variable")
        def objective(vector):
            candidate = dict(constants)
            for name, value in zip(names, vector):
                candidate[name] = float(np.exp(value)) if space.fields[name].get("log") else float(value)
                candidate[name] = min(space.fields[name]["high"], max(space.fields[name]["low"], candidate[name]))
            return engine.evaluate(candidate)
        dual_annealing(objective, bounds=bounds, maxiter=spec.max_proposals, maxfun=spec.max_proposals, no_local_search=True, rng=rng)
    elif spec.method == "tpe":
        import optuna
        study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=spec.seed))
        while True:
            engine.gate()
            trial = study.ask()
            candidate = {}
            for name in space.order():
                description = space.fields[name]
                current = dict(engine.config.parameters, **space.fixed, **candidate)
                if not all(current.get(parent) in values for parent, values in description.get("when", {}).items()):
                    continue
                if description["type"] == "choice":
                    candidate[name] = trial.suggest_categorical(name, description["values"])
                elif description["type"] == "integer":
                    candidate[name] = trial.suggest_int(name, description["low"], description["high"])
                else:
                    candidate[name] = trial.suggest_float(name, description["low"], description["high"], log=description.get("log", False))
            study.tell(trial, engine.evaluate(candidate))
    else:
        from pymoo.core.problem import ElementwiseProblem
        from pymoo.core.mixed import MixedVariableGA
        from pymoo.core.variable import Choice, Integer, Real
        from pymoo.optimize import minimize
        variables = {}
        for name, description in space.fields.items():
            if description["type"] == "choice":
                variables[name] = Choice(options=description["values"])
            elif description["type"] == "integer":
                variables[name] = Integer(bounds=(description["low"], description["high"]))
            else:
                bounds = (description["low"], description["high"])
                variables[name] = Real(bounds=tuple(np.log(bounds)) if description.get("log") else bounds)
        class Problem(ElementwiseProblem):
            def __init__(self):
                super().__init__(vars=variables, n_obj=1)
            def _evaluate(self, values, out, *args, **kwargs):
                candidate = {name: float(np.exp(value)) if space.fields[name].get("log") else value for name, value in values.items()}
                out["F"] = engine.evaluate(candidate)
        minimize(Problem(), MixedVariableGA(pop_size=min(8, spec.max_fits)), termination=("n_eval", spec.max_proposals), seed=spec.seed, verbose=False)


def search(config, spec, *, snapshot=None, sequence_plan=None, job_id=None, on_event=None, should_cancel=None, wait_for_dispatch=None, resume=None, artifact_dir=None, error_handler=None):
    """Search on train/validation only. Callbacks are synchronous pure JSON events."""
    config = ExperimentConfig.from_dict(config) if isinstance(config, dict) else config
    config = ExperimentConfig.from_dict(config.to_dict())
    spec = SearchSpec.from_dict(spec)
    spec.objective = spec.objective.resolve(config.task)
    if get_model(config.model_id)["task"] != config.task:
        raise ValueError("Model/task mismatch")
    resumed = SearchResume.from_dict(resume) if resume is not None else None
    if snapshot is None and resumed is not None and resumed.snapshot_path:
        snapshot = joblib.load(resumed.snapshot_path)
    if snapshot is None:
        if config.model_id in EXTENDED_MODEL_IDS:
            snapshot, built_plan = build_extended_snapshot(
                config,
                objective_labels_column=spec.objective.objective_labels_column,
            )
            if sequence_plan is not None and built_plan is not None and sequence_plan.to_dict() != built_plan.to_dict():
                raise ValueError("Supplied sequence plan differs from the source-derived plan")
            sequence_plan = sequence_plan or built_plan
        else:
            snapshot = build_snapshot(config, objective_labels_column=spec.objective.objective_labels_column)
    if config.model_id in {"H01", "H02", "H03", "N04", "N06"} and sequence_plan is None and resumed is not None:
        if not resumed.sequence_plan_path or not resumed.sequence_plan_receipt:
            raise ValueError("Resumed sequence search is missing its owned plan receipt")
        sequence_plan = load_owned_sequence_plan(
            resumed.sequence_plan_path,
            resumed.sequence_plan_receipt,
            expected_manifest=resumed.sequence_plan_manifest,
            config=config,
            snapshot=snapshot,
        )
    if config.model_id in {"H01", "H02", "H03", "N04", "N06"} and sequence_plan is None:
        raise ValueError("Sequence-model search requires an owned SequencePlan")
    if sequence_plan is not None:
        sequence_plan.verify(config, snapshot)
    snapshot.verify(config)
    if snapshot.manifest.get("objective_labels_column") != spec.objective.objective_labels_column:
        raise ValueError("Objective labels do not match this dataset snapshot")
    spec.space.validate_for(config, snapshot)
    if config.task == "anomaly detection":
        labels = snapshot.objective_labels
        if labels is None:
            raise ValueError("Independent anomaly labels were not loaded")
        scored = labels.iloc[snapshot.splits["train" if spec.objective.split == "train_exploratory" else "validation"]]
        if spec.objective.label_mapping:
            scored = scored.map(lambda value: spec.objective.label_mapping.get(str(value)))
        if scored.isna().any() or set(scored.unique()) != {-1, 1}:
            raise ValueError("Objective partition must contain normal (+1) and anomaly (-1) labels")
    engine = _SearchEngine(config, spec, snapshot, job_id or str(uuid.uuid4()), on_event, should_cancel, wait_for_dispatch, resumed, artifact_dir, sequence_plan=sequence_plan, error_handler=error_handler)
    engine.emit("search_started", snapshot=snapshot.to_dict(), snapshot_path=engine.snapshot_path, fingerprint=engine.fingerprint, spec=spec.to_dict())
    try:
        _adapter(engine)
    except _StopSearch:
        pass
    status = "cancelled" if engine.stop_reason == "cancelled" else "complete" if engine.winner else "no_valid_trial"
    result = SearchResult(engine.job_id, status, engine.stop_reason, spec, config, snapshot, engine.trials, engine.winner, engine.winner_session, engine.winner.metrics if engine.winner and engine.winner.score_split == "validation" else None, engine.actual_fit_count, engine.proposal_count, engine.elapsed(), engine.fingerprint, engine.snapshot_path, engine.best_session_path, sequence_plan=sequence_plan, sequence_plan_path=engine.sequence_plan_path, sequence_plan_receipt=engine.sequence_plan_receipt)
    engine.emit("search_finished", result=result.to_dict())
    return result


def load_search_result(payload):
    """Reconstitute a JSON summary from its fingerprinted local snapshot/session."""
    if isinstance(payload, SearchResult):
        return payload
    config = ExperimentConfig.from_dict(payload["config"])
    spec = SearchSpec.from_dict(payload["spec"])
    if not payload.get("snapshot_path"):
        raise ValueError("Search result has no persisted snapshot path")
    snapshot = joblib.load(payload["snapshot_path"])
    if not isinstance(snapshot, DatasetSnapshot):
        raise ValueError("Persisted search snapshot has unexpected type")
    snapshot.verify(config)
    if snapshot.to_dict() != payload["snapshot"]:
        raise ValueError("Search summary does not match its snapshot")
    sequence_plan = None
    sequence_plan_path = payload.get("sequence_plan_path")
    sequence_plan_receipt = payload.get("sequence_plan_receipt")
    if config.model_id in {"H01", "H02", "H03", "N04", "N06"}:
        if not sequence_plan_path or not sequence_plan_receipt:
            raise ValueError("Search result is missing its owned sequence plan receipt")
        sequence_plan = load_owned_sequence_plan(
            sequence_plan_path,
            sequence_plan_receipt,
            expected_manifest=payload.get("sequence_plan_manifest"),
            config=config,
            snapshot=snapshot,
        )
    elif payload.get("sequence_plan_manifest") is not None:
        raise ValueError("Search result has an unexpected sequence plan")
    if "sequence_plan_summary" in payload and payload.get("sequence_plan_summary") != sequence_plan_summary(sequence_plan):
        raise ValueError("Search summary sequence counts differ from its owned SequencePlan")
    engine = _SearchEngine(config, spec, snapshot, payload["job_id"], None, None, None, payload, None, sequence_plan=sequence_plan)
    if payload.get("winner") != (engine.winner.to_dict() if engine.winner else None):
        raise ValueError("Search summary winner differs from its trial history")
    return SearchResult(payload["job_id"], payload["status"], payload["stop_reason"], spec, config, snapshot, engine.trials, engine.winner, engine.winner_session, engine.winner.metrics if engine.winner and engine.winner.score_split == "validation" else None, engine.actual_fit_count, engine.proposal_count, engine.previous_elapsed, engine.fingerprint, payload["snapshot_path"], engine.best_session_path, sequence_plan=sequence_plan, sequence_plan_path=sequence_plan_path, sequence_plan_receipt=sequence_plan_receipt)
