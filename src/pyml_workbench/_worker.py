"""UTF-8 JSON-lines QProcess actions and local, durable fitted sessions.

Internal sessions contain holdout data. Only export's FittedModel is a user model.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, redirect_stdout
import json
import os
from pathlib import Path
import sys
import tempfile
from threading import Lock
import traceback
from typing import Any
import uuid

import joblib
import numpy as np
import pandas as pd

from pyml_workbench.config import ExperimentConfig
from pyml_workbench.diagnostics import LogSession
from pyml_workbench.batch import (
    create_search_jobs,
    export_training_exploration,
    finalize_frozen_search,
    freeze_search_winner,
    get_job_summary,
    request_job_control,
    run_search_jobs,
)
from pyml_workbench.data import load_dataset
from pyml_workbench.experiment import (
    ExperimentSession, _canonical_config, _session_training_curves, _write_artifacts, evaluate_test,
    freeze_experiment, load_model, prepare_experiment,
)
from pyml_workbench.plot_cache import (
    _verify_session_plot_cache_manifest,
    build_session_plot_cache_manifest,
)
from pyml_workbench.sequence_reporting import sequence_plan_summary


class SessionError(ValueError):
    """The requested session is invalid or belongs to another training run."""


_EMIT_LOCK = Lock()


def _emit(context: dict[str, Any], event_type: str, **payload: Any) -> None:
    event = {key: value for key, value in context.items() if not key.startswith("_")}
    event.update(type=event_type, **payload)
    with _EMIT_LOCK:
        print(json.dumps(event, ensure_ascii=False, allow_nan=False), file=context.get("_stdout", sys.stdout), flush=True)


def _config(path: Path) -> ExperimentConfig:
    return ExperimentConfig.from_dict(json.loads(path.read_text(encoding="utf-8-sig")))


@contextmanager
def _session_lock(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = path.with_name(path.name + ".lock")
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError as exc:
        raise SessionError("Session is busy or has an interrupted worker lock; do not start another action") from exc
    try:
        os.write(descriptor, str(os.getpid()).encode("ascii"))
        os.close(descriptor)
        yield
    finally:
        lock.unlink(missing_ok=True)


def _save(path: Path, envelope: dict[str, Any]) -> None:
    if (
        envelope.get("kind") == "pyml-workbench-internal-session"
        and ("plot_cache_manifest" in envelope or envelope.get("state") == "trained")
    ):
        envelope["plot_cache_manifest"] = build_session_plot_cache_manifest(
            envelope.get("session")
        )
    descriptor, temporary = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    try:
        joblib.dump(envelope, temporary, compress=3)
        with open(temporary, "r+b") as saved:
            os.fsync(saved.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def _load(path: Path, expected_id: str, config: ExperimentConfig) -> dict[str, Any]:
    value = joblib.load(path)
    if not isinstance(value, dict) or value.get("kind") != "pyml-workbench-internal-session" or value.get("schema_version") != 1:
        raise SessionError("File is not an internal fitted session")
    if value.get("session_id") != expected_id or value.get("session_path") != str(path):
        raise SessionError("Session identity/path does not match the requested training run")
    session = value.get("session")
    if not isinstance(session, ExperimentSession):
        raise SessionError("Session payload has an unexpected type")
    snapshot, digest = _canonical_config(session.config)
    if (snapshot != session.frozen_config or digest != session.frozen_config_sha256
            or digest != value.get("frozen_config_sha256")
            or session.fitted_model.config != snapshot
            or session.fitted_model.frozen_config_sha256 != digest
            or _canonical_config(config)[1] != digest):
        raise SessionError("Configuration changed since training; discard this session and retrain")
    state = value.get("state")
    if state not in {"trained", "frozen", "testing", "tested", "test_failed"}:
        raise SessionError("Session state is invalid")
    if state == "trained" and (session.frozen or session.finalized or session.result is not None or "test" in session.metrics):
        raise SessionError("Unfrozen session contains final-test state")
    if state in {"frozen", "tested"} and not session.frozen:
        raise SessionError("Session freeze state is inconsistent")
    if state == "frozen" and (session.finalized or session.result is not None or "test" in session.metrics):
        raise SessionError("Frozen session already contains final-test state")
    if state == "tested" and (not session.finalized or session.result is None):
        raise SessionError("Session has no cached final-test result")
    if state == "tested" and (session.result.config != snapshot or session.result.model.frozen_config_sha256 != digest):
        raise SessionError("Cached result does not belong to the fitted snapshot")
    if "plot_cache_manifest" in value:
        try:
            _verify_session_plot_cache_manifest(value, session)
        except Exception as exc:
            raise SessionError(f"Session plot-cache integrity manifest is invalid: {exc}") from exc
    return value


def _summary(envelope: dict[str, Any]) -> dict[str, Any]:
    session = envelope["session"]
    sequence_summary = sequence_plan_summary(getattr(session, "sequence_plan", None))
    return {
        "session_path": envelope["session_path"], "state": envelope["state"],
        "metrics": session.metrics, "events": list(session.events),
        "training_curves": _session_training_curves(session),
        "sequence_summary": sequence_summary,
        "frozen_config_sha256": session.frozen_config_sha256,
        "capabilities": session.fitted_model.capabilities,
        "test_evaluation_count": session.test_evaluation_count,
        "audit": session.result.audit if session.result is not None else None,
    }


def _session_action(args, context: dict[str, Any]) -> None:
    path = args.session.expanduser().resolve()
    config = _config(args.config)
    with _session_lock(path):
        callback = lambda name: _emit(context, "progress", name=name)
        if args.action == "train":
            if path.exists():
                raise FileExistsError("Session path already exists; select a new training session path")
            if path == Path(config.dataset.source_path).expanduser().resolve():
                raise SessionError("Session path cannot replace the input dataset")
            _emit(context, "progress", name="preparing")
            session = prepare_experiment(config, on_event=callback)
            envelope = {
                "kind": "pyml-workbench-internal-session", "schema_version": 1,
                "session_id": context["session_id"], "session_path": str(path),
                "frozen_config_sha256": session.frozen_config_sha256,
                "state": "trained", "session": session,
            }
            _save(path, envelope)
            _emit(context, "result", **_summary(envelope), cached=False)
            return
        envelope = _load(path, args.session_id, config)
        session = envelope["session"]
        state = envelope["state"]
        if state in {"testing", "test_failed"}:
            raise SessionError("Final test was interrupted or failed; this session cannot be tested or relabelled again")
        if args.action == "freeze":
            freeze_experiment(session, config)
            if state == "trained":
                envelope["state"] = "frozen"
                _save(path, envelope)
            _emit(context, "result", **_summary(envelope), cached=state != "trained")
        elif args.action == "test":
            if state == "trained":
                raise SessionError("Freeze this fitted session before final test")
            cached = state == "tested"
            if not cached:
                # Persist consumption before test access: a killed process cannot
                # reload the pre-test state and silently inspect test twice.
                envelope["state"] = "testing"
                _save(path, envelope)
                try:
                    evaluate_test(session, on_event=callback, export_artifacts=False)
                except Exception:
                    envelope["state"] = "test_failed"
                    _save(path, envelope)
                    raise
                envelope["state"] = "tested"
                _save(path, envelope)
            _emit(context, "result", **_summary(envelope), cached=cached)
        else:
            if state == "trained":
                raise SessionError("Freeze this fitted session before exporting")
            artifacts = _write_artifacts(
                args.output, session.source_path, session.frozen_config,
                session.frozen_config_sha256, session.metrics,
                pd.DataFrame(session.rows), session.fitted_model,
                training_curves=_session_training_curves(session),
            )
            _emit(context, "result", **_summary(envelope), artifact_paths=artifacts)


def _inspect(args, context):
    fitted = load_model(args.model)
    _emit(context, "result", **fitted.manifest())


def _inference(args, context):
    fitted = load_model(args.model)
    operation = args.operation
    if not fitted.capabilities.get(operation, False):
        raise ValueError(f"已加载模型不支持 {operation}，未运行推理")
    dataset = load_dataset(args.data, sheet_name=args.sheet)
    sequence_model = fitted.model_id in {"H01", "H02", "H03", "N04", "N06"}
    if sequence_model and operation != "predict":
        raise ValueError(f"{fitted.model_id} 的 CSV 序列适配器只支持 predict，未运行推理")
    missing = [name for name in fitted.feature_columns if name not in dataset.frame.columns]
    if missing:
        raise ValueError(f"新数据缺少模型需要的特征列：{missing}")
    values = dataset.frame.loc[:, fitted.feature_columns]
    output_path = args.output.expanduser().resolve()
    if output_path.exists():
        raise FileExistsError(f"为防止覆盖现有文件，拒绝写入：{output_path}")
    _emit(context, "progress", name="inference", rows=len(dataset.frame))
    if sequence_model:
        from .sequence_inference import sequence_inference_frame

        output = sequence_inference_frame(fitted, dataset.frame)
        for column in ("window_source_row_positions", "window_source_indexes"):
            if column in output.columns:
                output[column] = output[column].map(
                    lambda item: json.dumps(item, ensure_ascii=False, allow_nan=False)
                )
    else:
        array = np.asarray(getattr(fitted, operation)(values))
        if array.ndim == 1:
            names = {"predict": "prediction", "decision_function": "decision_score", "score_samples": "score_sample"}
            output = pd.DataFrame({names.get(operation, "component_1"): array})
        else:
            prefix = {"transform": "component", "predict_proba": "probability", "decision_function": "decision"}.get(operation, "prediction")
            output = pd.DataFrame(array, columns=[f"{prefix}_{index + 1}" for index in range(array.shape[1])])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8-sig", newline="") as handle:
        output.to_csv(handle, index=False)
    _emit(context, "result", rows=len(output), path=str(output_path), frozen_config_sha256=fitted.frozen_config_sha256)


def _batch_emit(context: dict[str, Any], event: dict[str, Any]) -> None:
    _emit(
        context,
        str(event.get("type", "progress")),
        **{key: value for key, value in event.items() if key != "type"},
    )


def _batch_summary(outcome: dict[str, Any]) -> dict[str, Any]:
    result = outcome.get("result") or {}
    winner = result.get("winner") or {}
    return {
        "job_id": outcome.get("job_id"),
        "status": outcome.get("status"),
        "error": outcome.get("error"),
        "error_id": outcome.get("error_id"),
        "traceback": outcome.get("traceback"),
        "diagnostic_errors": outcome.get("diagnostic_errors", []),
        "stop_reason": result.get("stop_reason"),
        "actual_fit_count": result.get("actual_fit_count"),
        "proposal_count": result.get("proposal_count"),
        "elapsed_seconds": result.get("elapsed_seconds"),
        "winner_trial_id": winner.get("trial_id"),
        "winner_value": winner.get("objective_value"),
        "winner_parameters": winner.get("parameters"),
        "winner_training_curves": winner.get("training_curves"),
        "sequence_summary": result.get("sequence_plan_summary"),
    }


def _batch_action(
    args,
    context: dict[str, Any],
    *,
    log_session: LogSession | None = None,
) -> None:
    if args.action == "batch-run-request":
        requests = json.loads(args.requests.read_text(encoding="utf-8-sig"))
        if not isinstance(requests, list):
            raise ValueError("批量请求文件必须是 JSON 数组")
        jobs = create_search_jobs(
            args.history,
            args.root,
            [(item["config"], item["spec"]) for item in requests],
        )
        _emit(
            context,
            "result",
            phase="queued",
            jobs=[
                {
                    "job_id": job["job_id"], "status": job["status"],
                    "model_id": job["model_id"], "task": job["task"],
                    "artifact_dir": job["artifact_dir"],
                }
                for job in jobs
            ],
        )
        outcomes = run_search_jobs(
            args.history,
            [job["job_id"] for job in jobs],
            max_workers=args.parallel_workers,
            on_event=lambda _job_id, event: _batch_emit(context, event),
            error_handler=log_session.record_error if log_session is not None else None,
        )
        _emit(
            context,
            "result",
            phase="finished",
            outcomes=[_batch_summary(item) for item in outcomes],
            summaries=[
                {key: value for key, value in get_job_summary(args.history, job["job_id"]).items()
                 if key in {"job_id", "status", "actual_fit_count", "proposal_count", "elapsed_seconds", "test_permission", "sequence_summary", "winner_training_curves"}}
                for job in jobs
            ],
        )
        return
    if args.action == "batch-run":
        outcomes = run_search_jobs(
            args.history,
            args.job_id,
            max_workers=args.parallel_workers,
            on_event=lambda _job_id, event: _batch_emit(context, event),
            error_handler=log_session.record_error if log_session is not None else None,
        )
        _emit(
            context,
            "result",
            phase="finished",
            outcomes=[_batch_summary(item) for item in outcomes],
            summaries=[
                {key: value for key, value in get_job_summary(args.history, job_id).items()
                 if key in {"job_id", "status", "actual_fit_count", "proposal_count", "elapsed_seconds", "test_permission", "sequence_summary", "winner_training_curves"}}
                for job_id in args.job_id
            ],
        )
        return
    if args.action == "batch-freeze":
        selection = freeze_search_winner(args.history, args.job_id)
        _emit(context, "result", phase="selection_frozen", job_id=args.job_id,
              selection=selection.to_dict())
        return
    if args.action == "batch-export-exploration":
        result = export_training_exploration(
            args.history, args.job_id, output_dir=args.output,
        )
        _emit(context, "result", phase="training_exploration_exported", result=result)
        return
    if args.action == "batch-finalize":
        result = finalize_frozen_search(
            args.history,
            args.job_id,
            output_dir=args.output,
            on_event=lambda event: _batch_emit(context, event),
        )
        _emit(context, "result", phase="finalized", job_id=args.job_id,
              final_selection=result.to_dict())
        return
    if args.action == "batch-control":
        sequence = request_job_control(
            args.history,
            args.job_id,
            args.control,
            worker_active=args.worker_active,
        )
        _emit(context, "result", phase="control_requested", job_id=args.job_id,
              control=args.control, event_sequence=sequence)
        return
    raise ValueError(f"Unsupported batch action: {args.action}")


class _Parser(argparse.ArgumentParser):
    def error(self, message):
        raise ValueError(message)


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="backslashreplace")
    raw = sys.argv[1:] if argv is None else argv
    context = {"action": raw[0] if raw else None, "operation": raw[0] if raw else None, "session_id": None, "_stdout": sys.stdout}
    log_session = LogSession(
        log_root=os.environ.get("PYML_LOG_ROOT") or None,
        session_id=os.environ.get("PYML_LOG_SESSION_ID") or uuid.uuid4().hex,
        role="worker",
        context={"stage": context.get("operation")},
    )
    try:
        parser = _Parser(description="pyml-workbench QProcess worker")
        subparsers = parser.add_subparsers(dest="action", required=True)
        for action in ("train", "freeze", "test", "export"):
            child = subparsers.add_parser(action)
            child.add_argument("--config", required=True, type=Path)
            child.add_argument("--session", required=True, type=Path)
            if action != "train":
                child.add_argument("--session-id", required=True)
            if action == "export":
                child.add_argument("--output", required=True, type=Path)
        child = subparsers.add_parser("inference")
        child.add_argument("--operation", required=True, choices=("predict", "predict_proba", "decision_function", "score_samples", "transform"))
        child.add_argument("--model", required=True, type=Path)
        child.add_argument("--data", required=True, type=Path)
        child.add_argument("--sheet", default=None)
        child.add_argument("--output", required=True, type=Path)
        child = subparsers.add_parser("inspect")
        child.add_argument("--model", required=True, type=Path)
        child = subparsers.add_parser("batch-run-request")
        child.add_argument("--history", required=True, type=Path)
        child.add_argument("--root", required=True, type=Path)
        child.add_argument("--requests", required=True, type=Path)
        child.add_argument("--parallel-workers", type=int, choices=(1, 2), default=1)
        child = subparsers.add_parser("batch-run")
        child.add_argument("--history", required=True, type=Path)
        child.add_argument("--job-id", required=True, nargs="+")
        child.add_argument("--parallel-workers", type=int, choices=(1, 2), default=1)
        child = subparsers.add_parser("batch-freeze")
        child.add_argument("--history", required=True, type=Path)
        child.add_argument("--job-id", required=True)
        child = subparsers.add_parser("batch-export-exploration")
        child.add_argument("--history", required=True, type=Path)
        child.add_argument("--job-id", required=True)
        child.add_argument("--output", type=Path, default=None)
        child = subparsers.add_parser("batch-finalize")
        child.add_argument("--history", required=True, type=Path)
        child.add_argument("--job-id", required=True)
        child.add_argument("--output", type=Path, default=None)
        child = subparsers.add_parser("batch-control")
        child.add_argument("--history", required=True, type=Path)
        child.add_argument("--job-id", required=True)
        child.add_argument("--control", required=True, choices=("pause", "resume", "cancel"))
        child.add_argument("--worker-active", action="store_true")
        args = parser.parse_args(raw)
        context.update(action=args.action, operation=getattr(args, "operation", args.action), session_id=getattr(args, "session_id", None))
        if args.action == "train":
            context["session_id"] = str(uuid.uuid4())
        with redirect_stdout(sys.stderr):
            if args.action in {"train", "freeze", "test", "export"}:
                _session_action(args, context)
            elif args.action.startswith("batch-"):
                _batch_action(args, context, log_session=log_session)
            elif args.action == "inspect":
                _inspect(args, context)
            else:
                _inference(args, context)
        return 0
    except Exception as exc:
        traceback_text = "".join(
            traceback.format_exception(type(exc), exc, exc.__traceback__)
        )
        try:
            record = log_session.record_error(
                exc,
                context={"stage": context.get("operation")},
            )
        except BaseException:
            record = None
        error_id = record.error_id if record is not None else uuid.uuid4().hex
        try:
            _emit(
                context,
                "error",
                error_type=type(exc).__name__,
                message=str(exc),
                error_id=error_id,
                traceback=traceback_text,
            )
        except BaseException:
            pass
        try:
            sys.stderr.write(traceback_text)
            sys.stderr.flush()
        except BaseException:
            pass
        return 1
    finally:
        log_session.close()


if __name__ == "__main__":
    raise SystemExit(main())
