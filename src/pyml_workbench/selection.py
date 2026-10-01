"""Explicit validation winner freeze, 80% refit and durable single final test."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, replace
import json
import os
from pathlib import Path
import time
import uuid

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from sklearn.metrics import balanced_accuracy_score

from .config import ExperimentConfig
from .experiment import DatasetSnapshot, ExperimentError, ExperimentSession, _canonical_config, _json_digest, _session_training_curves, _table_digest, _target_for_task, _write_artifacts, evaluate_test, freeze_experiment, prepare_experiment
from .search import SearchResult, atomic_joblib
from .search_space import json_copy
from .sequence import SequencePlan


@dataclass
class FrozenSelection:
    job_id: str
    winner_trial_id: str
    config: ExperimentConfig
    snapshot: DatasetSnapshot
    metadata: dict
    selection_sha256: str
    sequence_plan: SequencePlan | None = None

    def verify(self):
        self.snapshot.verify(self.config)
        if _json_digest(self.metadata) != self.selection_sha256 or self.metadata["config"] != self.config.to_dict() or self.metadata["snapshot"] != self.snapshot.to_dict():
            raise ExperimentError("Frozen selection was modified")
        plan_manifest = self.sequence_plan.to_dict() if self.sequence_plan is not None else None
        if self.metadata.get("sequence_plan_manifest") != plan_manifest:
            raise ExperimentError("Frozen selection sequence plan was modified")
        if self.sequence_plan is not None:
            self.sequence_plan.verify(self.config, self.snapshot)
        if self.job_id != self.metadata["job_id"] or self.winner_trial_id != self.metadata["winner_trial_id"]:
            raise ExperimentError("Frozen winner identity was modified")

    def to_dict(self):
        return json_copy(dict(self.metadata, selection_sha256=self.selection_sha256))


@dataclass
class FinalSelection:
    final_run_id: str
    session_path: str
    state: str
    selection: dict
    session: ExperimentSession
    refit_seconds: float
    cached: bool = False
    artifact_paths: dict | None = None

    def to_dict(self):
        return json_copy(dict(final_run_id=self.final_run_id, session_id=self.final_run_id, session_path=self.session_path, state=self.state, selection=self.selection, fit_scope="train_validation", selection_validation=self.selection["validation_metrics"], final_refit=self.session.metrics.get("refit"), final_test=self.session.metrics.get("test"), training_curves=_session_training_curves(self.session), refit_seconds=self.refit_seconds, test_evaluation_count=self.session.test_evaluation_count, cached=self.cached, frozen_config_sha256=self.session.frozen_config_sha256, capabilities=self.session.fitted_model.capabilities, audit=self.session.result.audit if self.session.result else None, artifact_paths=self.artifact_paths or {}))


FinalSelectionResult = FinalSelection


def freeze_selection(search_result: SearchResult, *, winner_trial_id=None) -> FrozenSelection:
    winner, session = search_result.winner, search_result.winner_session
    if winner is None or session is None or winner.status != "complete" or winner.score_split != "validation" or winner.objective_value is None:
        raise ExperimentError("Only a valid independent-validation winner can be frozen")
    if winner_trial_id is not None and winner_trial_id != winner.trial_id:
        raise ExperimentError("Requested winner does not match the selected validation trial")
    if session.frozen or session.finalized or session.test_evaluation_count or "test" in session.metrics:
        raise ExperimentError("Winner has already entered a final-test workflow")
    config = replace(search_result.config, parameters=json_copy(winner.parameters), output_dir=None)
    snapshot = search_result.snapshot
    snapshot.verify(config)
    sequence_plan = search_result.sequence_plan
    if config.model_id in {"H01", "H02", "H03", "N04", "N06"}:
        if sequence_plan is None:
            raise ExperimentError("Sequence-model winner has no owned SequencePlan")
        sequence_plan.verify(config, snapshot)
    elif sequence_plan is not None:
        raise ExperimentError("Tabular model winner unexpectedly carries a SequencePlan")
    if session.snapshot_manifest != snapshot.to_dict() or _canonical_config(config)[1] != winner.config_sha256 or session.fitted_model.frozen_config_sha256 != winner.config_sha256 or session.config.to_dict() != config.to_dict() or session.fitted_model.config != config.to_dict():
        raise ExperimentError("Winner fitted configuration/snapshot differs from its search record")
    if (getattr(session, "sequence_plan", None).to_dict() if getattr(session, "sequence_plan", None) is not None else None) != (sequence_plan.to_dict() if sequence_plan is not None else None):
        raise ExperimentError("Winner session does not match the search SequencePlan")
    metadata = dict(job_id=search_result.job_id, winner_trial_id=winner.trial_id, config=config.to_dict(), snapshot=snapshot.to_dict(), search_fingerprint=search_result.fingerprint, objective=search_result.spec.objective.to_dict(), objective_value=winner.objective_value, validation_metrics=json_copy(winner.metrics), fit_scope="train_validation", sequence_plan_manifest=sequence_plan.to_dict() if sequence_plan is not None else None)
    extended_metadata = getattr(session, "extended_training_metadata", None)
    if config.model_id in {"H01", "H02", "H03", "N01", "N02", "N04", "N06"}:
        refit_provenance = getattr(session, "refit_provenance", None)
        if not isinstance(extended_metadata, dict) or not isinstance(refit_provenance, dict):
            raise ExperimentError("Extended winner is missing its training provenance")
        # Persist the exact helper contract expected by prepare_experiment's
        # final refit path, not the session's training-metadata wrapper.
        metadata["extended_provenance"] = json_copy(refit_provenance)
    return FrozenSelection(search_result.job_id, winner.trial_id, config, snapshot, metadata, _json_digest(metadata), sequence_plan)


@contextmanager
def _lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    marker = path.with_name(path.name + ".lock")
    try:
        descriptor = os.open(marker, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise ExperimentError("Final session is busy or has an interrupted lock") from exc
    os.close(descriptor)
    try:
        yield
    finally:
        marker.unlink(missing_ok=True)


def _emit(callback, name, selection, envelope=None, **data):
    if callback:
        callback(json_copy(dict(type="progress", action="selection", operation="selection", job_id=selection.job_id, trial_id=selection.winner_trial_id, final_run_id=envelope["session_id"] if envelope else None, name=name, data=data)))


def _refresh_plot_cache_manifest(envelope, *, allow_create=False):
    session = envelope.get("session")
    if session is None:
        envelope.pop("plot_cache_manifest", None)
        return
    if "plot_cache_manifest" not in envelope and not allow_create:
        return
    from .plot_cache import build_session_plot_cache_manifest

    envelope["plot_cache_manifest"] = build_session_plot_cache_manifest(session)


def _verify_envelope(path, envelope, selection):
    if not isinstance(envelope, dict) or envelope.get("kind") != "pyml-workbench-internal-session" or envelope.get("schema_version") != 1 or envelope.get("session_path") != str(path):
        raise ExperimentError("File/path is not the expected final session")
    if envelope.get("selection_sha256") != selection.selection_sha256 or envelope.get("selection") != selection.to_dict():
        raise ExperimentError("Final session belongs to another selection")
    if envelope.get("state") in {"refitting", "refit_failed", "testing", "test_failed"}:
        raise ExperimentError("Final refit/test was interrupted or failed; automatic retry is prohibited")
    session = envelope.get("session")
    if not isinstance(session, ExperimentSession) or not session.frozen or session.fit_scope != "train_validation":
        raise ExperimentError("Final session has invalid fitted state")
    if "plot_cache_manifest" in envelope:
        from .plot_cache import _verify_session_plot_cache_manifest

        try:
            _verify_session_plot_cache_manifest(envelope, session)
        except Exception as exc:
            raise ExperimentError(f"Final session plot-cache manifest is invalid: {exc}") from exc
    freeze_experiment(session, selection.config)
    if envelope.get("frozen_config_sha256") != session.frozen_config_sha256 or session.snapshot_manifest != selection.snapshot.to_dict():
        raise ExperimentError("Final session configuration or data/split identity differs")
    expected_target = _target_for_task(selection.snapshot.target, selection.config.task, session.fitted_model.estimator)
    if _table_digest(session.features) != _table_digest(selection.snapshot.features) or _table_digest(session.target) != _table_digest(expected_target):
        raise ExperimentError("Final session data was modified")
    expected_train = np.concatenate([selection.snapshot.splits["train"], selection.snapshot.splits["validation"]])
    if not np.array_equal(session.splits["train"], expected_train) or len(session.splits["validation"]) or not np.array_equal(session.splits["test"], selection.snapshot.splits["test"]):
        raise ExperimentError("Final refit/test positions were modified")
    if not isinstance(envelope.get("session_id"), str) or envelope.get("state") not in {"frozen", "tested"}:
        raise ExperimentError("Final session identity/state is invalid")
    session_plan = getattr(session, "sequence_plan", None)
    if (session_plan.to_dict() if session_plan is not None else None) != (selection.sequence_plan.to_dict() if selection.sequence_plan is not None else None):
        raise ExperimentError("Final session SequencePlan differs from the frozen selection")
    if envelope["state"] == "frozen" and (session.finalized or session.result is not None or "test" in session.metrics):
        raise ExperimentError("Final session test state is inconsistent")
    if envelope["state"] == "tested" and (not session.finalized or session.result is None):
        raise ExperimentError("Final session is missing its cached test result")


def _refit(selection, path, on_event, should_cancel):
    selection.verify()
    source = Path(selection.snapshot.source_path).resolve()
    if path == source or path == source.parent or path.suffix != ".joblib":
        raise ExperimentError("Final session must use a separate .joblib file")
    if path.exists():
        envelope = joblib.load(path)
        _verify_envelope(path, envelope, selection)
        return envelope, True
    if should_cancel and should_cancel():
        raise ExperimentError("Final refit cancelled before starting")
    envelope = dict(kind="pyml-workbench-internal-session", schema_version=1, session_id=str(uuid.uuid4()), session_path=str(path), state="refitting", session=None, frozen_config_sha256=_canonical_config(selection.config)[1], selection=selection.to_dict(), selection_sha256=selection.selection_sha256, fit_scope="train_validation", refit_seconds=0.)
    atomic_joblib(path, envelope)
    started = time.monotonic()
    try:
        _emit(on_event, "refit_started", selection, envelope, selection_snapshot=selection.to_dict())
        with threadpool_limits(limits=1):
            extended_provenance = selection.metadata.get("extended_provenance")
            session = prepare_experiment(
                selection.config,
                snapshot=selection.snapshot,
                sequence_plan=selection.sequence_plan,
                _fit_scope="train_validation",
                selected_epochs=extended_provenance.get("selected_epochs") if isinstance(extended_provenance, dict) else None,
                refit_provenance=extended_provenance,
            )
        session.metrics["selection_validation"] = json_copy(selection.metadata["validation_metrics"])
        session.metrics["selection"].update(mode="parameters chosen using independent validation before 80% refit", winner_trial_id=selection.winner_trial_id, selection_sha256=selection.selection_sha256)
        session.fitted_model.preprocessing["selection_sha256"] = selection.selection_sha256
        freeze_experiment(session, selection.config)
        envelope.update(session=session, state="frozen", refit_seconds=time.monotonic() - started)
        _refresh_plot_cache_manifest(envelope, allow_create=True)
        atomic_joblib(path, envelope)
    except Exception:
        envelope.update(state="refit_failed", refit_seconds=time.monotonic() - started)
        atomic_joblib(path, envelope)
        raise
    _emit(on_event, "refit_completed", selection, envelope, refit_seconds=envelope["refit_seconds"], final_run_id=envelope["session_id"], session_path=str(path))
    return envelope, False


def _result(envelope, cached):
    return FinalSelection(envelope["session_id"], envelope["session_path"], envelope["state"], envelope["selection"], envelope["session"], envelope["refit_seconds"], cached, envelope.get("artifact_paths", {}))


def refit_selected(selection, *, session_path, on_event=None, should_cancel=None):
    """Refit once without reading test, then leave a compatible frozen envelope."""
    path = Path(session_path).expanduser().resolve()
    with _lock(path):
        envelope, cached = _refit(selection, path, on_event, should_cancel)
        return _result(envelope, cached)


def finalize_selected(selection, *, session_path, on_event=None, should_cancel=None):
    """Explicit final action; repeat calls only load the same durable test cache."""
    path = Path(session_path).expanduser().resolve()
    with _lock(path):
        envelope, cached = _refit(selection, path, on_event, should_cancel)
        if envelope["state"] == "tested":
            return _result(envelope, True)
        if should_cancel and should_cancel():
            raise ExperimentError("Final evaluation cancelled before accessing test; frozen refit is preserved")
        envelope["state"] = "testing"
        _refresh_plot_cache_manifest(envelope)
        atomic_joblib(path, envelope)
        try:
            _emit(on_event, "test_started", selection, envelope, session_path=str(path), test_permission_consumed=True)
            with threadpool_limits(limits=1):
                evaluate_test(envelope["session"], export_artifacts=False)
            session = envelope["session"]
            if selection.config.task == "anomaly detection" and session.test_evaluation_count:
                expected = selection.snapshot.objective_labels.iloc[selection.snapshot.splits["test"]]
                mapping = selection.metadata["objective"].get("label_mapping")
                if mapping:
                    expected = expected.map(lambda value: mapping.get(str(value)))
                if expected.isna().any() or not set(expected.unique()) <= {-1, 1}:
                    raise ExperimentError("Final anomaly objective labels are invalid")
                if set(expected.unique()) == {-1, 1}:
                    predictions = session.result.results.loc[session.result.results["split"] == "test", "y_pred"].to_numpy()
                    session.metrics["test"]["balanced_accuracy"] = float(balanced_accuracy_score(expected.to_numpy(), predictions))
                else:
                    session.metrics["test"]["objective_note"] = "Final partition has only one objective label class; balanced accuracy was not reported"
            envelope["state"] = "tested"
            _refresh_plot_cache_manifest(envelope)
            atomic_joblib(path, envelope)
        except Exception:
            envelope["state"] = "test_failed"
            _refresh_plot_cache_manifest(envelope)
            atomic_joblib(path, envelope)
            raise
        result = _result(envelope, False)
        _emit(on_event, "test_completed", selection, envelope, result=result.to_dict())
        return result


def export_selected(final_result, output_dir):
    """Export only the fitted bundle, metrics/config and result tables."""
    path = Path(final_result.session_path).expanduser().resolve()
    with _lock(path):
        envelope = joblib.load(path)
        if envelope.get("session_path") != str(path) or envelope.get("session_id") != final_result.final_run_id or envelope.get("selection") != final_result.selection or envelope.get("state") not in {"frozen", "tested"}:
            raise ExperimentError("Export does not match the stored final session")
        session = envelope["session"]
        freeze_experiment(session)
        if "plot_cache_manifest" in envelope:
            from .plot_cache import _verify_session_plot_cache_manifest

            try:
                _verify_session_plot_cache_manifest(envelope, session)
            except Exception as exc:
                raise ExperimentError(f"Final session plot-cache manifest is invalid: {exc}") from exc
        paths = _write_artifacts(output_dir, session.source_path, session.frozen_config, session.frozen_config_sha256, session.metrics, session.result.results if session.result else pd.DataFrame(session.rows), session.fitted_model, training_curves=_session_training_curves(session))
        selection_path = Path(paths["output_dir"]) / "selection.json"
        selection_path.write_text(json.dumps(final_result.selection, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
        paths["selection"] = str(selection_path)
        envelope["artifact_paths"] = paths
        _refresh_plot_cache_manifest(envelope)
        atomic_joblib(path, envelope)
        final_result.artifact_paths = paths
        return paths
