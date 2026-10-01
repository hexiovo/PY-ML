"""Durable queue orchestration for validation-only parameter searches."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import math
import os
import re
import threading
import time
import traceback
import uuid
from typing import Any, Iterator

import joblib
import pandas as pd
from threadpoolctl import threadpool_limits

from .config import ExperimentConfig
from .catalog import get_model
from .experiment import DatasetSnapshot, ExperimentError, _json_digest, _session_training_curves, _write_artifacts, build_extended_snapshot, build_snapshot, load_model
from .history import BudgetExhausted, DispatchStopped, HistoryError, HistoryStore
from .search import SearchResume, SearchResult, SearchSpec, TrialRecord, load_search_result as restore_search_result, search
from .selection import (
    FinalSelection,
    FrozenSelection,
    export_selected,
    finalize_selected,
    freeze_selection,
)
from .sequence import SequencePlan, load_owned_sequence_plan, save_owned_sequence_plan
from .sequence_reporting import sequence_plan_summary_from_manifest
from .sequence_models import EXTENDED_MODEL_IDS


MAX_ACTUAL_FITS = 50
MAX_PROPOSALS = 250
MAX_ACTIVE_SECONDS = 20 * 60
SEARCH_THREADS_PER_WORKER = 1
_ERROR_ID_RE = re.compile(r"^[0-9a-f]{32}$")


class BatchError(RuntimeError):
    """Raised when a queued search cannot safely be created, resumed, or frozen."""


def _finite_objective_score(value: Any) -> float | None:
    """Normalize a persisted score for comparison, rejecting null and non-finite values."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    normalized = float(value)
    return normalized if math.isfinite(normalized) else None


def _rank_validation_group_entries(entries: list[dict[str, Any]]) -> dict[str, int | None]:
    """Assign competition ranks to finite validation scores within comparison groups.

    Ties share a rank (1, 1, 3). Training-exploratory and unscored entries return
    ``None`` and are left for the caller to label and place after ranked entries.
    """
    ranks: dict[str, int | None] = {}
    grouped: dict[str, list[dict[str, Any]]] = {}
    for entry in entries:
        job_id = str(entry["job_id"])
        ranks[job_id] = None
        score = _finite_objective_score(entry.get("score"))
        if (
            entry.get("score_split", "validation") != "validation"
            or entry.get("direction") not in {"min", "max"}
            or score is None
        ):
            continue
        group_key = str(entry["comparison_group_key"])
        grouped.setdefault(group_key, []).append(
            entry | {"_normalized_score": score, "job_id": job_id}
        )

    for group_entries in grouped.values():
        direction = group_entries[0]["direction"]
        group_entries.sort(key=lambda item: (
            -item["_normalized_score"] if direction == "max" else item["_normalized_score"],
            str(item.get("created_at", "")),
            str(item["job_id"]),
        ))
        previous_score: float | None = None
        previous_rank: int | None = None
        for position, entry in enumerate(group_entries, start=1):
            score = entry["_normalized_score"]
            if previous_score is None or score != previous_score:
                previous_score = score
                previous_rank = position
            ranks[entry["job_id"]] = previous_rank
    return ranks


def _load_json(value: str, field: str) -> dict[str, Any]:
    try:
        decoded = json.loads(value)
    except (TypeError, json.JSONDecodeError) as exc:
        raise BatchError(f"Stored {field} is not valid JSON: {exc}") from exc
    if not isinstance(decoded, dict):
        raise BatchError(f"Stored {field} must be a JSON object")
    return decoded


def _validated_request(
    config: ExperimentConfig | dict[str, Any],
    spec: SearchSpec | dict[str, Any],
    *,
    snapshot: DatasetSnapshot | None = None,
    sequence_plan: SequencePlan | None = None,
) -> tuple[ExperimentConfig, SearchSpec, DatasetSnapshot, SequencePlan | None]:
    config, spec = _validated_search_definition(config, spec)
    if snapshot is None:
        if config.model_id in EXTENDED_MODEL_IDS:
            current, built_plan = build_extended_snapshot(
                config, objective_labels_column=spec.objective.objective_labels_column
            )
            if sequence_plan is not None and built_plan is not None and sequence_plan.to_dict() != built_plan.to_dict():
                raise BatchError("Supplied sequence plan differs from the source-derived plan")
            sequence_plan = sequence_plan or built_plan
        else:
            current = build_snapshot(config, objective_labels_column=spec.objective.objective_labels_column)
    else:
        current = snapshot
    _validate_request_snapshot(config, spec, current)
    if config.model_id in {"H01", "H02", "H03", "N04", "N06"}:
        if sequence_plan is None:
            raise BatchError("Sequence search requires its owned SequencePlan")
        sequence_plan.verify(config, current)
    elif sequence_plan is not None:
        raise BatchError("Tabular model search cannot accept a SequencePlan")
    return config, spec, current, sequence_plan


def _validated_search_definition(
    config: ExperimentConfig | dict[str, Any],
    spec: SearchSpec | dict[str, Any],
) -> tuple[ExperimentConfig, SearchSpec]:
    config = ExperimentConfig.from_dict(config) if isinstance(config, dict) else config
    if not isinstance(config, ExperimentConfig):
        raise TypeError("config must be an ExperimentConfig or JSON-compatible dictionary")
    config = ExperimentConfig.from_dict(replace(config, output_dir=None).to_dict())
    if get_model(config.model_id)["task"] != config.task:
        raise BatchError("Selected model is incompatible with the configured task")
    spec = SearchSpec.from_dict(spec)
    if spec.max_fits > MAX_ACTUAL_FITS:
        raise BatchError(f"A search may use at most {MAX_ACTUAL_FITS} actual fits")
    if spec.max_proposals > MAX_PROPOSALS:
        raise BatchError(f"A search may use at most {MAX_PROPOSALS} proposals")
    if spec.timeout_seconds > MAX_ACTIVE_SECONDS:
        raise BatchError(f"A search may use at most {MAX_ACTIVE_SECONDS // 60} active minutes")
    spec.objective = spec.objective.resolve(config.task)
    spec.space.validate_for(config)
    if spec.method == "grid":
        spec.space.grid()
    return config, spec


def _validate_request_snapshot(
    config: ExperimentConfig,
    spec: SearchSpec,
    snapshot: DatasetSnapshot,
) -> None:
    snapshot.verify(config)
    if snapshot.manifest.get("objective_labels_column") != spec.objective.objective_labels_column:
        raise BatchError("Objective labels do not match the queued dataset snapshot")
    if config.task == "anomaly detection":
        labels = snapshot.objective_labels
        if labels is None:
            raise BatchError("Anomaly search requires an independent objective-label column")
        split_name = "train" if spec.objective.split == "train_exploratory" else "validation"
        selected = labels.iloc[snapshot.splits[split_name]]
        if spec.objective.label_mapping:
            selected = selected.map(lambda value: spec.objective.label_mapping.get(str(value)))
        if selected.isna().any() or set(selected.unique()) != {-1, 1}:
            raise BatchError("The scored anomaly partition must contain both -1 and +1 labels")


def _queue_search_job(
    store: HistoryStore,
    artifact_root: Path,
    config: ExperimentConfig,
    spec: SearchSpec,
    snapshot: DatasetSnapshot,
    sequence_plan: SequencePlan | None = None,
) -> dict[str, Any]:
    from .owned_snapshot import save_owned_snapshot

    job_id = uuid.uuid4().hex
    artifact_dir = artifact_root / "jobs" / job_id
    snapshot_path = artifact_dir / "snapshot.joblib"
    sequence_plan_path = artifact_dir / "sequence_plan.joblib"
    sequence_plan_receipt_path = artifact_dir / "sequence_plan_receipt.json"
    source = Path(snapshot.source_path).resolve()
    if artifact_dir.resolve() in {source, source.parent} or snapshot_path.resolve() == source:
        raise BatchError("Search artifacts cannot replace the input file or its parent")
    artifact_dir.parent.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(exist_ok=False)
    try:
        receipt = save_owned_snapshot(snapshot, snapshot_path)
        plan_receipt = None
        if sequence_plan is not None:
            sequence_plan.verify(config, snapshot)
            plan_receipt = save_owned_sequence_plan(sequence_plan, sequence_plan_path)
            sequence_plan_receipt_path.write_text(json.dumps(plan_receipt, ensure_ascii=False, indent=2), encoding="utf-8")
        return store.create_job(
            job_id=job_id,
            dataset_id=snapshot.manifest["source_sha256"],
            task=config.task,
            model_id=config.model_id,
            config=config.to_dict(),
            snapshot=snapshot.to_dict(),
            split_summary={
                "seed": config.split.seed,
                "split_sha256": snapshot.manifest["split_sha256"],
                "positions": snapshot.manifest["split_positions"],
            },
            search_spec=spec.to_dict(),
            budget={
                "max_actual_fits": spec.max_fits,
                "max_proposals": spec.max_proposals,
                "max_active_seconds": float(spec.timeout_seconds),
                "parallel_workers_default": 1,
                "parallel_workers_max": 2,
                "threads_per_worker": SEARCH_THREADS_PER_WORKER,
                "trial_test_evaluation_count": 0,
            },
            artifact_dir=artifact_dir,
            fingerprint=_expected_fingerprint(config, spec, snapshot),
            snapshot_receipt=receipt,
            sequence_plan_receipt=plan_receipt,
        )
    except Exception:
        existing = store.get_job(job_id)
        if existing is None or Path(existing["artifact_dir"]).expanduser().resolve() != artifact_dir.resolve():
            snapshot_path.unlink(missing_ok=True)
            sequence_plan_path.unlink(missing_ok=True)
            sequence_plan_receipt_path.unlink(missing_ok=True)
            try:
                artifact_dir.rmdir()
            except OSError:
                pass
        raise


def create_search_job(
    history_path: str | Path,
    artifact_root: str | Path,
    config: ExperimentConfig | dict[str, Any],
    spec: SearchSpec | dict[str, Any],
) -> dict[str, Any]:
    """Preflight and queue one search without fitting an estimator."""
    config, spec, snapshot, sequence_plan = _validated_request(config, spec)
    root = Path(artifact_root).expanduser().resolve()
    if root.exists() and not root.is_dir():
        raise BatchError(f"Artifact root is not a directory: {root}")
    root.mkdir(parents=True, exist_ok=True)
    store = HistoryStore(history_path)
    return _queue_search_job(store, root, config, spec, snapshot, sequence_plan)


def create_search_jobs(
    history_path: str | Path,
    artifact_root: str | Path,
    requests: list[tuple[ExperimentConfig | dict[str, Any], SearchSpec | dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Preflight and queue several independent model/space requests in order."""
    if not requests:
        raise BatchError("At least one search request is required")
    prepared = [_validated_request(config, spec) for config, spec in requests]
    root = Path(artifact_root).expanduser().resolve()
    if root.exists() and not root.is_dir():
        raise BatchError(f"Artifact root is not a directory: {root}")
    root.mkdir(parents=True, exist_ok=True)
    store = HistoryStore(history_path)
    jobs = []
    for config, spec, snapshot, sequence_plan in prepared:
        jobs.append(_queue_search_job(store, root, config, spec, snapshot, sequence_plan))
    return jobs


@contextmanager
def _runner_lock(artifact_dir: Path) -> Iterator[None]:
    artifact_dir.mkdir(parents=True, exist_ok=True)
    lock_path = artifact_dir / "runner.lock"
    while True:
        try:
            descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError as exc:
            try:
                pid = int(lock_path.read_text(encoding="ascii").strip())
                os.kill(pid, 0)
            except (OSError, ValueError):
                lock_path.unlink(missing_ok=True)
                continue
            raise BatchError(f"Search job is already running in process {pid}") from exc
        else:
            try:
                os.write(descriptor, str(os.getpid()).encode("ascii"))
                os.close(descriptor)
                break
            except Exception:
                os.close(descriptor)
                lock_path.unlink(missing_ok=True)
                raise
    try:
        yield
    finally:
        lock_path.unlink(missing_ok=True)


class _ActiveClock:
    """Persist approximate active-search time while a worker is inside a fit."""

    def __init__(self, store: HistoryStore, job_id: str, elapsed: float, run_id: str, slot_id: int):
        self.store, self.job_id = store, job_id
        self.run_id, self.slot_id = run_id, slot_id
        self._lock = threading.Lock()
        self._total = float(elapsed)
        self._segment_start = time.monotonic()
        self._paused = True
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._heartbeat, name="pyml-batch-clock", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def elapsed(self) -> float:
        with self._lock:
            return self._total if self._paused else self._total + time.monotonic() - self._segment_start

    def pause(self) -> None:
        with self._lock:
            if not self._paused:
                self._total += time.monotonic() - self._segment_start
                self._paused = True
        self.store.touch_elapsed(self.job_id, self.elapsed())

    def resume(self) -> None:
        with self._lock:
            if self._paused:
                self._segment_start = time.monotonic()
                self._paused = False

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=3)
        self.store.touch_elapsed(self.job_id, self.elapsed())

    def _heartbeat(self) -> None:
        while not self._stop.wait(2.0):
            with self._lock:
                elapsed = None if self._paused else self._total + time.monotonic() - self._segment_start
            try:
                self.store.heartbeat_worker_slot(self.run_id, self.slot_id, self.job_id)
                if elapsed is not None:
                    self.store.touch_elapsed(self.job_id, elapsed)
            except (HistoryError, OSError):
                # Search callbacks still persist exact elapsed values at safe points.
                continue


def _rebuild_resume(store: HistoryStore, job: dict[str, Any]) -> SearchResume | None:
    if not job.get("fingerprint"):
        return None
    records: list[TrialRecord] = []
    trial_rows = store.list_trials(job["job_id"])
    running_rows = [row for row in trial_rows if row["status"] == "running"]
    actual_fit_count = int(job["actual_fit_count"])
    persisted_fit_count = sum(int(row["actual_fit_count"]) for row in trial_rows)
    if len(running_rows) > 1:
        raise BatchError(
            "Cannot resume a search job with multiple running trial receipts; "
            "one worker may fit only one trial at a time."
        )
    if persisted_fit_count != actual_fit_count:
        raise BatchError(
            "Cannot resume search job because its aggregate fit counter does not "
            f"match persisted trial receipts ({actual_fit_count} != {persisted_fit_count})."
        )
    if running_rows and int(running_rows[0]["actual_fit_count"]) != 1:
        raise BatchError(
            "Cannot resume a running trial without exactly one persisted fit receipt."
        )

    for row in trial_rows:
        if row["status"] == "running":
            raw = _load_json(row["result_json"] or "{}", "running trial")
            raw.update(
                status="failed",
                # The per-trial SQLite count is only 0/1. The job counter is
                # the globally ordered fit number and the running receipt is
                # the most recently budgeted fit for this serial search job.
                fit_index=actual_fit_count,
                error="Worker stopped after the fit was budgeted; this fit remains consumed.",
                duration_seconds=max(0.0, float(raw.get("duration_seconds", 0.0))),
                test_evaluation_count=0,
            )
            store.finish_trial(
                job_id=job["job_id"], trial_id=row["trial_id"], status="failed",
                objective_value=raw.get("objective_value"), metrics=raw.get("metrics", {}),
                result=raw, error=raw["error"], duration_seconds=raw["duration_seconds"],
                cache_key=row["cache_key"],
            )
            row = store.get_trial(job["job_id"], row["trial_id"])
        raw = _load_json(row["result_json"] or "{}", "trial result")
        if row["status"] == "proposed":
            # Preserve the proposal receipt/index without pretending an estimator
            # fit happened. It is deliberately excluded from the persisted cache.
            raw.update(
                status="invalid",
                fit_index=None,
                cache_key=None,
                objective_value=None,
                error=raw.get("error") or "Worker stopped before this proposal entered estimator.fit.",
                test_evaluation_count=0,
            )
        elif row["status"] == "succeeded":
            raw["status"] = "complete"
        elif row["status"] in {"cached", "cached_failure"}:
            raw["status"] = "cached"
            raw["cache_hit"] = True
        elif row["status"] == "rejected":
            raw["status"] = "invalid"
        else:
            raw["status"] = row["status"]
        raw["test_evaluation_count"] = 0
        records.append(TrialRecord.from_dict(raw))
    sequence_plan_receipt = None
    sequence_plan_manifest = None
    if job.get("sequence_plan_receipt_json"):
        sequence_plan_receipt = _load_json(job["sequence_plan_receipt_json"], "owned sequence plan receipt")
        sequence_plan_manifest = sequence_plan_receipt.get("manifest")
    return SearchResume(
        records=records,
        actual_fit_count=actual_fit_count,
        proposal_count=int(job["proposal_count"]),
        elapsed_seconds=float(job.get("elapsed_seconds", 0.0)),
        best_session_path=_best_session_path(store, job),
        snapshot_path=str(Path(job["artifact_dir"]) / "snapshot.joblib"),
        fingerprint=job["fingerprint"],
        sequence_plan_path=(str(Path(job["artifact_dir"]) / "sequence_plan.joblib") if sequence_plan_receipt else None),
        sequence_plan_receipt=sequence_plan_receipt,
        sequence_plan_manifest=sequence_plan_manifest,
    )


def _lease_heartbeat(store: HistoryStore, run_id: str):
    stop = threading.Event()

    def heartbeat() -> None:
        while not stop.wait(2.0):
            try:
                store.heartbeat_batch_run(run_id)
            except HistoryError:
                return

    thread = threading.Thread(target=heartbeat, name="pyml-batch-lease", daemon=True)
    thread.start()

    def close() -> None:
        stop.set()
        thread.join(timeout=3)

    return close


def _best_session_path(store: HistoryStore, job: dict[str, Any]) -> str | None:
    for row in reversed(store.list_trials(job["job_id"])):
        raw = _load_json(row["result_json"] or "{}", "trial result")
        path = raw.get("session_path")
        if path and Path(path).is_file():
            # Search resume verifies the session against the actual winner record.
            result = _latest_search_result(store, job["job_id"])
            winner = result.get("winner") if result else None
            if winner and winner.get("trial_id") == raw.get("trial_id"):
                return path
    result = _latest_search_result(store, job["job_id"])
    if result and result.get("best_session_path"):
        return result["best_session_path"]
    try:
        config = ExperimentConfig.from_dict(_load_json(job["config_json"], "config"))
        spec = SearchSpec.from_dict(_load_json(job["search_json"], "search spec"))
        direction = spec.objective.resolve(config.task).direction
    except (BatchError, ValueError):
        direction = "max"
    best: dict[str, Any] | None = None
    for row in store.list_trials(job["job_id"]):
        raw = _load_json(row["result_json"] or "{}", "trial result")
        value = raw.get("objective_value")
        if raw.get("status") != "complete" or value is None or not raw.get("session_path"):
            continue
        if best is None or (value > best["objective_value"] if direction == "max" else value < best["objective_value"]):
            best = raw
    return best.get("session_path") if best else None


def _latest_search_result(store: HistoryStore, job_id: str) -> dict[str, Any] | None:
    events = store.list_events(job_id)
    for event in reversed(events):
        if event["event_type"] == "search_finished":
            payload = _load_json(event["payload_json"], "search result event")
            result = payload.get("result")
            return result if isinstance(result, dict) else None
    return None


def _run_search_job_impl(
    history_path: str | Path, job_id: str, *, run_id: str, on_event=None,
    error_handler=None,
) -> SearchResult:
    """Run/resume one durable search job; every fit receipt commits synchronously."""
    store = HistoryStore(history_path)
    job = store.get_job(job_id)
    if job is None:
        raise HistoryError(f"Unknown job_id: {job_id}")
    if job["status"] in {"completed", "cancelled", "failed"}:
        result = load_search_result(history_path, job_id)
        if result is None:
            raise BatchError(f"Terminal job {job_id} has no persisted search result")
        return result
    if job["status"] == "cancel_requested":
        store.set_job_status(job_id, "cancelled", reason="cancelled before worker dispatch")
        raise DispatchStopped(f"Job {job_id} was cancelled before dispatch")
    artifact_dir = Path(job["artifact_dir"]).expanduser().resolve()
    with _runner_lock(artifact_dir):
        job = store.get_job(job_id)
        if job is None:
            raise HistoryError(f"Unknown job_id: {job_id}")
        try:
            if job["status"] == "paused":
                store.request_control(job_id, "resume")
                job = store.get_job(job_id)
                if job is None:
                    raise HistoryError(f"Unknown job_id: {job_id}")
            config = ExperimentConfig.from_dict(_load_json(job["config_json"], "config"))
            spec = SearchSpec.from_dict(_load_json(job["search_json"], "search spec"))
            config, spec = _validated_search_definition(config, spec)
            snapshot = _load_owned_job_snapshot(job, config, spec)
            sequence_plan = _load_owned_job_sequence_plan(job, config, snapshot)
            if not job.get("fingerprint"):
                store.set_search_fingerprint(job_id, _expected_fingerprint(config, spec, snapshot))
                job = store.get_job(job_id)
                if job is None:
                    raise HistoryError(f"Unknown job_id: {job_id}")
            if job["control_request"] == "cancel":
                store.set_job_status(job_id, "cancelled", reason="cancelled before search started")
                raise DispatchStopped(f"Job {job_id} was cancelled before dispatch")
            resume = _rebuild_resume(store, job) if job["fingerprint"] else None
        except DispatchStopped:
            raise
        except Exception as exc:
            current = store.get_job(job_id)
            if current and current["status"] not in {"cancelled", "completed", "failed"}:
                store.set_job_status(
                    job_id,
                    "failed",
                    reason=f"Preflight failed before fit: {type(exc).__name__}: {exc}",
                )
            raise

        def event_callback(event: dict[str, Any]) -> None:
            name = event.get("name")
            data = event.get("data") or {}
            record = data.get("record")
            elapsed = data.get("elapsed_seconds")
            if name == "search_started":
                store.set_search_fingerprint(job_id, data.get("fingerprint", ""))
                store.touch_elapsed(job_id, elapsed)
            elif name == "proposal" and record:
                store.record_proposal(
                    job_id=job_id,
                    trial_id=record["trial_id"],
                    parameters=record.get("parameters", {}),
                    objective_direction=record.get("direction", "max"),
                    max_proposals=spec.max_proposals,
                    config_sha256=record.get("config_sha256"),
                    cache_key=record.get("cache_key"),
                    result=record,
                )
                store.touch_elapsed(job_id, elapsed)
            elif name == "fit_started" and record:
                store.mark_fit_started(
                    job_id=job_id,
                    trial_id=record["trial_id"],
                    max_actual_fits=spec.max_fits,
                    elapsed_seconds=elapsed,
                )
            elif name == "cache_hit" and record:
                source_trial_id = record.get("cached_from")
                if not source_trial_id:
                    raise HistoryError("Search cache receipt is missing its source trial")
                store.record_cache_hit(
                    job_id=job_id,
                    trial_id=record["trial_id"],
                    source_trial_id=source_trial_id,
                    cache_key=record.get("cache_key"),
                    result=record,
                )
                store.touch_elapsed(job_id, elapsed)
            elif name in {"trial_completed", "trial_failed"} and record:
                status_map = {"complete": "succeeded", "failed": "failed", "unscorable": "unscorable"}
                status = status_map.get(record.get("status"))
                if status is None:
                    raise HistoryError(f"Unexpected completed trial status: {record.get('status')!r}")
                store.finish_trial(
                    job_id=job_id,
                    trial_id=record["trial_id"],
                    status=status,
                    objective_value=record.get("objective_value"),
                    metrics=record.get("metrics", {}),
                    result=record,
                    error=record.get("error"),
                    duration_seconds=record.get("duration_seconds", 0.0),
                    cache_key=record.get("cache_key"),
                )
                store.touch_elapsed(job_id, elapsed)
            elif name == "search_finished":
                result_data = data.get("result")
                if not isinstance(result_data, dict):
                    raise HistoryError("Search completion event is missing its result")
                store.persist_search_result(job_id=job_id, result=result_data)
            else:
                store.record_search_progress(
                    job_id=job_id,
                    event_type=str(name or "search_progress"),
                    payload=event,
                    elapsed_seconds=elapsed,
                )
            if on_event is not None:
                on_event(event)

        def should_cancel() -> bool:
            try:
                return store.control_request(job_id) == "cancel"
            except HistoryError:
                return True

        def wait_for_dispatch() -> bool:
            request = store.control_request(job_id)
            if request == "cancel":
                return False
            if request != "pause":
                return True
            clock.pause()
            current = store.get_job(job_id)
            if current and current["status"] == "pausing":
                store.set_job_status(job_id, "paused", reason="safe point reached")
            while True:
                request = store.control_request(job_id)
                if request == "cancel":
                    return False
                if request is None:
                    clock.resume()
                    store.set_job_status(job_id, "running", reason="resumed at safe point")
                    return True
                time.sleep(0.4)

        slot_id = store.claim_worker_slot(run_id, job_id)
        clock = _ActiveClock(store, job_id, float(job.get("elapsed_seconds", 0.0)), run_id, slot_id)
        clock.start()
        clock.resume()
        try:
            result = search(
                config,
                spec,
                snapshot=snapshot,
                sequence_plan=sequence_plan,
                job_id=job_id,
                on_event=event_callback,
                should_cancel=should_cancel,
                wait_for_dispatch=wait_for_dispatch,
                resume=resume,
                artifact_dir=artifact_dir,
                error_handler=error_handler,
            )
            terminal = "cancelled" if result.status == "cancelled" else "completed"
            store.set_job_status(job_id, terminal, reason=result.stop_reason)
            return result
        except Exception as exc:
            current = store.get_job(job_id)
            if current and current["status"] not in {"cancelled", "completed"}:
                store.set_job_status(job_id, "failed", reason=f"{type(exc).__name__}: {exc}")
            raise
        finally:
            clock.stop()
            store.release_worker_slot(run_id, slot_id, job_id)


def run_search_job(
    history_path: str | Path, job_id: str, *, on_event=None, error_handler=None
) -> SearchResult:
    """Run a single job under the default one-worker queue limit."""
    store = HistoryStore(history_path)
    run_id = store.begin_batch_run(1)
    close_heartbeat = _lease_heartbeat(store, run_id)
    try:
        return _run_search_job_impl(
            history_path, job_id, run_id=run_id, on_event=on_event,
            error_handler=error_handler,
        )
    finally:
        close_heartbeat()
        store.end_batch_run(run_id)


def run_search_jobs(
    history_path: str | Path,
    job_ids: list[str],
    *,
    max_workers: int = 1,
    on_event=None,
    error_handler=None,
) -> list[dict[str, Any]]:
    """Run queued combinations with one worker by default and at most two."""
    if not job_ids or len(set(job_ids)) != len(job_ids):
        raise BatchError("Provide one or more distinct queued job IDs")
    if isinstance(max_workers, bool) or max_workers not in {1, 2}:
        raise BatchError("parallel_workers must be 1 or 2")

    store = HistoryStore(history_path)
    run_id = store.begin_batch_run(max_workers)
    close_heartbeat = _lease_heartbeat(store, run_id)

    def run_one(job_id: str) -> dict[str, Any]:
        try:
            result = _run_search_job_impl(
                history_path,
                job_id,
                run_id=run_id,
                on_event=(lambda event: on_event(job_id, event)) if on_event else None,
                error_handler=error_handler,
            )
            diagnostic_errors = [
                {
                    "error_id": trial.error_id,
                    "error_type": (
                        trial.error.split(":", 1)[0]
                        if trial.error and ":" in trial.error
                        else "TrialError"
                    ),
                    "message": trial.error or "Search trial failed",
                    "traceback": trial.traceback or "",
                    "trial_id": trial.trial_id,
                }
                for trial in result.trials
                if trial.status == "failed"
            ]
            return {
                "job_id": job_id,
                "status": "completed" if result.status != "cancelled" else "cancelled",
                "result": result.to_dict(),
                "diagnostic_errors": diagnostic_errors,
            }
        except Exception as exc:
            try:
                traceback_text = "".join(
                    traceback.format_exception(type(exc), exc, exc.__traceback__)
                )
            except Exception:
                traceback_text = f"{type(exc).__name__}: {exc}"
            error_id = uuid.uuid4().hex
            if error_handler is not None:
                try:
                    logged = error_handler(
                        exc,
                        context={"stage": "batch_search", "job_id": job_id},
                    )
                    logged_id = getattr(logged, "error_id", None)
                    if isinstance(logged_id, str) and _ERROR_ID_RE.fullmatch(logged_id):
                        error_id = logged_id
                except Exception:
                    # Search failures remain reportable even when diagnostics are unavailable.
                    pass
            return {
                "job_id": job_id,
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
                "error_id": error_id,
                "traceback": traceback_text,
                "diagnostic_errors": [
                    {
                        "error_id": error_id,
                        "error_type": type(exc).__name__,
                        "message": f"{type(exc).__name__}: {exc}",
                        "traceback": traceback_text,
                    }
                ],
            }

    try:
        if max_workers == 1:
            return [run_one(job_id) for job_id in job_ids]
        outputs: dict[str, dict[str, Any]] = {}
        with threadpool_limits(limits=1):
            with ThreadPoolExecutor(max_workers=2, thread_name_prefix="pyml-search") as executor:
                futures = {executor.submit(run_one, job_id): job_id for job_id in job_ids}
                for future in as_completed(futures):
                    job_id = futures[future]
                    try:
                        outputs[job_id] = future.result()
                    except Exception as exc:
                        outputs[job_id] = {"job_id": job_id, "status": "failed", "error": f"{type(exc).__name__}: {exc}"}
        return [outputs[job_id] for job_id in job_ids]
    finally:
        close_heartbeat()
        store.end_batch_run(run_id)


def run_pending_search_jobs(
    history_path: str | Path, *, max_workers: int = 1, on_event=None,
    error_handler=None,
) -> list[dict[str, Any]]:
    store = HistoryStore(history_path)
    jobs = [job["job_id"] for job in store.list_jobs(status="queued")]
    if not jobs:
        return []
    return run_search_jobs(
        history_path, jobs, max_workers=max_workers, on_event=on_event,
        error_handler=error_handler,
    )


def run_batch(
    history_path: str | Path,
    artifact_root: str | Path,
    requests: list[tuple[ExperimentConfig | dict[str, Any], SearchSpec | dict[str, Any]]],
    *,
    max_workers: int = 1,
    on_event=None,
    error_handler=None,
) -> dict[str, Any]:
    """Preflight, queue, and run independent search requests in one call.

    The returned job IDs and outcomes are also persisted in ``history_path``;
    callers can recover full trial and selection records with
    :func:`get_job_summary` and :func:`load_search_result`.
    """
    jobs = create_search_jobs(history_path, artifact_root, requests)
    outcomes = run_search_jobs(
        history_path,
        [job["job_id"] for job in jobs],
        max_workers=max_workers,
        on_event=on_event,
        error_handler=error_handler,
    )
    return {
        "history_path": str(Path(history_path).expanduser().resolve()),
        "jobs": [
            {
                "job_id": job["job_id"],
                "status": job["status"],
                "task": job["task"],
                "model_id": job["model_id"],
                "artifact_dir": job["artifact_dir"],
            }
            for job in jobs
        ],
        "results": outcomes,
    }


def request_job_control(history_path: str | Path, job_id: str, action: str, *, worker_active: bool = True) -> int:
    """Persist pause/resume/cancel; queued work can be cancelled without a worker."""
    store = HistoryStore(history_path)
    job = store.get_job(job_id)
    if job is None:
        raise HistoryError(f"Unknown job_id: {job_id}")
    sequence = store.request_control(job_id, action)
    if action == "cancel" and not worker_active:
        store.set_job_status(job_id, "cancelled", reason="cancelled before worker dispatch")
    return sequence


def _expected_fingerprint(config: ExperimentConfig, spec: SearchSpec, snapshot: DatasetSnapshot) -> str:
    return _json_digest({
        "config": replace(config, output_dir=None).to_dict(),
        "space": spec.space.to_dict(),
        "objective": spec.objective.resolve(config.task).to_dict(),
        "seed": spec.seed,
        "method": spec.method,
        "snapshot": snapshot.to_dict(),
    })


def _load_owned_job_snapshot(
    job: dict[str, Any],
    config: ExperimentConfig,
    spec: SearchSpec,
) -> DatasetSnapshot:
    from .owned_snapshot import load_owned_snapshot

    artifact_dir = Path(job["artifact_dir"]).expanduser().resolve()
    snapshot_path = artifact_dir / "snapshot.joblib"
    if not snapshot_path.is_file():
        raise BatchError(
            "Owned search snapshot is missing or invalid; requeue this legacy job before fitting."
        )
    expected_manifest = _load_json(job["snapshot_json"], "queued snapshot")
    snapshot_json = job["snapshot_json"]
    if hashlib.sha256(snapshot_json.encode("utf-8")).hexdigest() != job["snapshot_sha256"]:
        raise BatchError("Queued snapshot manifest checksum does not match its history receipt")
    receipt_json = job.get("snapshot_receipt_json")
    try:
        if receipt_json:
            receipt = _load_json(receipt_json, "owned snapshot receipt")
            snapshot = load_owned_snapshot(
                snapshot_path,
                receipt,
                expected_manifest=expected_manifest,
                config=config,
            )
        else:
            # Early S02 jobs may have a valid snapshot written when search first
            # started. Accept that verified owned file; never rebuild from source.
            snapshot = joblib.load(snapshot_path)
            if not isinstance(snapshot, DatasetSnapshot):
                raise BatchError("Legacy owned search snapshot has an unexpected type")
            snapshot.verify(config)
            if snapshot.to_dict() != expected_manifest:
                raise ExperimentError("Legacy owned snapshot differs from the queued manifest")
    except Exception as exc:
        raise BatchError(f"Owned search snapshot failed verification: {exc}") from exc
    if snapshot.manifest.get("source_sha256") != job["dataset_id"]:
        raise BatchError("Owned search snapshot source identity differs from its queued job")
    split = _load_json(job["split_json"], "queued split")
    if (
        split.get("split_sha256") != snapshot.manifest.get("split_sha256")
        or split.get("seed") != config.split.seed
        or split.get("positions") != snapshot.manifest.get("split_positions")
    ):
        raise BatchError("Owned search snapshot split differs from its queued job")
    _validate_request_snapshot(config, spec, snapshot)
    expected_fingerprint = _expected_fingerprint(config, spec, snapshot)
    if job.get("fingerprint") and job["fingerprint"] != expected_fingerprint:
        raise BatchError("Owned search snapshot/config/protocol fingerprint differs from its queued job")
    return snapshot


def _load_owned_job_sequence_plan(
    job: dict[str, Any],
    config: ExperimentConfig,
    snapshot: DatasetSnapshot,
) -> SequencePlan | None:
    receipt_json = job.get("sequence_plan_receipt_json")
    is_sequence_model = config.model_id in {"H01", "H02", "H03", "N04", "N06"}
    if not receipt_json:
        if is_sequence_model or snapshot.manifest.get("sequence_plan_sha256") is not None:
            raise BatchError("Queued sequence search is missing its owned plan receipt")
        return None
    if not is_sequence_model:
        raise BatchError("Queued tabular search unexpectedly contains a SequencePlan")
    artifact_dir = Path(job["artifact_dir"]).expanduser().resolve()
    plan_path = artifact_dir / "sequence_plan.joblib"
    try:
        receipt = _load_json(receipt_json, "owned sequence plan receipt")
        plan = load_owned_sequence_plan(
            plan_path,
            receipt,
            config=config,
            snapshot=snapshot,
        )
    except Exception as exc:
        raise BatchError(f"Owned sequence plan failed verification: {exc}") from exc
    if plan.plan_sha256 != snapshot.manifest.get("sequence_plan_sha256"):
        raise BatchError("Owned sequence plan identity differs from the queued snapshot")
    return plan


def load_search_result(history_path: str | Path, job_id: str) -> SearchResult | None:
    """Rebuild a typed result from SQLite metadata and local fit/snapshot artifacts."""
    store = HistoryStore(history_path)
    job = store.get_job(job_id)
    if job is None:
        raise HistoryError(f"Unknown job_id: {job_id}")
    payload = _latest_search_result(store, job_id)
    if payload is None:
        return None
    config = ExperimentConfig.from_dict(_load_json(job["config_json"], "config"))
    spec = SearchSpec.from_dict(_load_json(job["search_json"], "search spec"))
    config, spec = _validated_search_definition(config, spec)
    artifact_dir = Path(job["artifact_dir"]).expanduser().resolve()
    snapshot_path = Path(payload.get("snapshot_path") or (artifact_dir / "snapshot.joblib"))
    snapshot = _load_owned_job_snapshot(job, config, spec)
    sequence_plan = _load_owned_job_sequence_plan(job, config, snapshot)
    if snapshot_path.resolve() != (artifact_dir / "snapshot.joblib").resolve():
        raise BatchError("Search result references a snapshot outside its owning job directory")
    if payload.get("fingerprint") != job.get("fingerprint") or payload.get("fingerprint") != _expected_fingerprint(config, spec, snapshot):
        raise ExperimentError("Search result fingerprint differs from its queued config/objective/space/data")
    payload["snapshot_path"] = str(snapshot_path)
    if sequence_plan is not None:
        expected_plan_path = artifact_dir / "sequence_plan.joblib"
        expected_receipt = _load_json(job["sequence_plan_receipt_json"], "owned sequence plan receipt")
        if Path(str(payload.get("sequence_plan_path", ""))).expanduser().resolve() != expected_plan_path.resolve():
            raise BatchError("Search result references a SequencePlan outside its owning job directory")
        if payload.get("sequence_plan_manifest") != sequence_plan.to_dict() or payload.get("sequence_plan_receipt") != expected_receipt:
            raise ExperimentError("Search result sequence plan differs from its queued receipt")
        payload["sequence_plan_path"] = str(expected_plan_path)
    elif payload.get("sequence_plan_manifest") is not None or payload.get("sequence_plan_receipt") is not None:
        raise BatchError("Tabular search result unexpectedly references a SequencePlan")
    restored = restore_search_result(payload)
    if not isinstance(restored, SearchResult):
        raise BatchError("Search result summary could not be restored")
    return restored


def freeze_search_winner(history_path: str | Path, job_id: str) -> FrozenSelection:
    """Persist an explicit validation-winner freeze before any refit/test action."""
    store = HistoryStore(history_path)
    job = store.get_job(job_id)
    if job is None or job["status"] != "completed":
        raise BatchError("Only a completed search job can freeze its validation winner")
    artifact_dir = Path(job["artifact_dir"]).expanduser().resolve()
    selection_path = artifact_dir / "frozen-selection.joblib"
    if selection_path.exists():
        selection = joblib.load(selection_path)
        if not isinstance(selection, FrozenSelection):
            raise BatchError("Stored frozen selection has an unexpected type")
        selection.verify()
        if selection.metadata.get("objective", {}).get("split") != "validation":
            raise BatchError("Only an independent-validation winner can be frozen")
        return selection
    result = load_search_result(history_path, job_id)
    if result is None:
        raise BatchError("Search result is not available")
    selection = freeze_selection(result)
    from .search import atomic_joblib

    atomic_joblib(selection_path, selection)
    store.record_search_progress(
        job_id=job_id,
        event_type="selection_frozen",
        payload=selection.to_dict(),
        elapsed_seconds=result.elapsed_seconds,
    )
    return selection


def export_training_exploration(
    history_path: str | Path,
    job_id: str,
    *,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Export an existing train-exploratory winner without refitting or scoring holdouts."""
    store = HistoryStore(history_path)
    job = store.get_job(job_id)
    if job is None or job["status"] != "completed":
        raise BatchError("Only a completed search job can export a training exploration")
    try:
        config = ExperimentConfig.from_dict(_load_json(job["config_json"], "config"))
        spec = SearchSpec.from_dict(_load_json(job["search_json"], "search spec"))
        objective = spec.objective.resolve(config.task)
    except (BatchError, ValueError) as exc:
        raise BatchError(f"Stored search configuration cannot be exported: {exc}") from exc
    if objective.split != "train_exploratory":
        raise BatchError("Training exploration export requires objective split='train_exploratory'")

    result = load_search_result(history_path, job_id)
    if result is None or result.winner is None or result.winner_session is None:
        raise BatchError("This training exploration has no scoreable fitted winner to export")
    winner, session = result.winner, result.winner_session
    if (
        result.best_validation is not None
        or winner.status != "complete"
        or winner.objective_value is None
        or winner.score_split != "train_exploratory"
        or session.fit_scope != "train"
        or session.test_evaluation_count != 0
        or session.finalized
        or session.result is not None
        or "test" in session.metrics
    ):
        raise BatchError("Stored winner is not an untouched training-only exploratory fit")

    target = Path(output_dir).expanduser().resolve() if output_dir is not None else (
        Path(job["artifact_dir"]).expanduser().resolve() / "training-exploration-export"
    )
    manifest_path = target / "training-exploration.json"
    if target.exists():
        if not manifest_path.is_file():
            raise BatchError(f"Export directory already exists without a matching training manifest: {target}")
        try:
            manifest = _load_json(
                manifest_path.read_text(encoding="utf-8"), "training exploration manifest"
            )
            paths = manifest["artifact_paths"]
            cached_match = (
                manifest.get("schema_version") == 1
                and manifest.get("job_id") == job_id
                and manifest.get("winner_trial_id") == winner.trial_id
                and manifest.get("search_fingerprint") == result.fingerprint
                and manifest.get("fit_scope") == "train"
                and manifest.get("score_split") == "train_exploratory"
                and manifest.get("test_evaluation_count") == 0
                and Path(manifest.get("output_dir", "")).resolve() == target
                and all(Path(value).is_file() for key, value in paths.items() if key != "output_dir")
            )
        except (BatchError, KeyError, TypeError, ValueError, OSError):
            cached_match = False
            paths = {}
        if not cached_match:
            raise BatchError(f"Export directory already exists and does not match this training exploration: {target}")
        try:
            reloaded = load_model(paths["model"])
        except (KeyError, ExperimentError) as exc:
            raise BatchError(f"Cached training model could not be reloaded: {exc}") from exc
        if reloaded.frozen_config_sha256 != session.frozen_config_sha256:
            raise BatchError("Cached exported model does not match the training winner configuration")
        return {
            "job_id": job_id,
            "status": "cached",
            "winner_trial_id": winner.trial_id,
            "score_split": "train_exploratory",
            "fit_scope": "train",
            "actual_fit_count": result.actual_fit_count,
            "test_evaluation_count": 0,
            "artifact_paths": paths,
        }

    artifact_paths = _write_artifacts(
        target,
        session.source_path,
        session.frozen_config,
        session.frozen_config_sha256,
        session.metrics,
        session.result.results if session.result else pd.DataFrame(session.rows),
        session.fitted_model,
        training_curves=_session_training_curves(session),
    )
    reloaded = load_model(artifact_paths["model"])
    if reloaded.frozen_config_sha256 != session.frozen_config_sha256:
        raise BatchError("Exported model does not match the training winner configuration")
    manifest = {
        "schema_version": 1,
        "job_id": job_id,
        "winner_trial_id": winner.trial_id,
        "search_fingerprint": result.fingerprint,
        "objective": objective.to_dict(),
        "score_split": "train_exploratory",
        "fit_scope": "train",
        "actual_fit_count": result.actual_fit_count,
        "proposal_count": result.proposal_count,
        "validation_ranked": False,
        "test_evaluation_count": 0,
        "output_dir": str(target),
        "artifact_paths": artifact_paths | {"manifest": str(manifest_path)},
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8"
    )
    artifact_paths["manifest"] = str(manifest_path)
    store.record_search_progress(
        job_id=job_id,
        event_type="training_exploration_exported",
        payload=manifest,
        elapsed_seconds=result.elapsed_seconds,
    )
    return {
        "job_id": job_id,
        "status": "exported",
        "winner_trial_id": winner.trial_id,
        "score_split": "train_exploratory",
        "fit_scope": "train",
        "actual_fit_count": result.actual_fit_count,
        "test_evaluation_count": 0,
        "artifact_paths": artifact_paths,
    }


def finalize_frozen_search(
    history_path: str | Path,
    job_id: str,
    *,
    output_dir: str | Path | None = None,
    on_event=None,
) -> FinalSelection:
    """Refit the explicitly frozen winner, then claim and consume the one test."""
    store = HistoryStore(history_path)
    job = store.get_job(job_id)
    if job is None:
        raise HistoryError(f"Unknown job_id: {job_id}")
    artifact_dir = Path(job["artifact_dir"]).expanduser().resolve()
    selection_path = artifact_dir / "frozen-selection.joblib"
    if not selection_path.is_file():
        raise BatchError("Freeze a validation winner before final refit/test")
    selection = joblib.load(selection_path)
    if not isinstance(selection, FrozenSelection):
        raise BatchError("Stored frozen selection has an unexpected type")
    selection.verify()
    if selection.metadata.get("objective", {}).get("split") != "validation":
        raise BatchError("Only an independent-validation winner can be finalized")
    final_session_path = artifact_dir / "final-session.joblib"
    consumed_session_id: str | None = None

    def on_selection_event(event: dict[str, Any]) -> None:
        nonlocal consumed_session_id
        name = event.get("name")
        data = event.get("data") or {}
        final_run_id = event.get("final_run_id")
        if name == "test_started":
            if not final_run_id:
                raise HistoryError("Final test event is missing its final-session identity")
            store.consume_test_permission(job_id=job_id, final_session_id=final_run_id)
            consumed_session_id = final_run_id
        elif name == "test_completed":
            result_payload = data.get("result")
            if not final_run_id or not isinstance(result_payload, dict):
                raise HistoryError("Final test completion is missing its durable result")
            store.finish_test(job_id=job_id, final_session_id=final_run_id, result=result_payload)
        store.record_search_progress(job_id=job_id, event_type=str(name or "selection_progress"), payload=event)
        if on_event is not None:
            on_event(event)

    try:
        result = finalize_selected(selection, session_path=final_session_path, on_event=on_selection_event)
        if result.session.test_evaluation_count not in {0, 1}:
            raise ExperimentError("Final selection has an invalid test evaluation count")
        if output_dir is not None:
            paths = export_selected(result, output_dir)
            store.record_search_progress(
                job_id=job_id,
                event_type="selection_exported",
                payload={"artifact_paths": paths},
            )
        return result
    except Exception as exc:
        if consumed_session_id:
            permit = store.get_test_permission(job_id)
            if permit and permit["state"] == "consumed":
                store.finish_test(
                    job_id=job_id,
                    final_session_id=consumed_session_id,
                    error=f"{type(exc).__name__}: {exc}",
                )
        raise


def get_job_summary(history_path: str | Path, job_id: str) -> dict[str, Any]:
    store = HistoryStore(history_path)
    job = store.get_job(job_id)
    if job is None:
        raise HistoryError(f"Unknown job_id: {job_id}")
    events = store.list_events(job_id)
    artifact_paths = None
    for event in reversed(events):
        if event["event_type"] not in {"training_exploration_exported", "selection_exported"}:
            continue
        payload = _load_json(event["payload_json"], "export event")
        artifact_paths = payload.get("artifact_paths")
        if artifact_paths:
            break
    search_result = _latest_search_result(store, job_id)
    sequence_summary = search_result.get("sequence_plan_summary") if search_result else None
    if sequence_summary is None and job.get("sequence_plan_receipt_json"):
        receipt = _load_json(job["sequence_plan_receipt_json"], "owned sequence plan receipt")
        sequence_summary = sequence_plan_summary_from_manifest(receipt.get("manifest"))
    winner = (search_result or {}).get("winner") or {}
    return {
        "job_id": job_id,
        "status": job["status"],
        "control_request": job["control_request"],
        "actual_fit_count": int(job["actual_fit_count"]),
        "proposal_count": int(job["proposal_count"]),
        "elapsed_seconds": float(job.get("elapsed_seconds", 0.0)),
        "fingerprint": job.get("fingerprint"),
        "trials": store.list_trials(job_id),
        "events": events,
        "artifact_paths": artifact_paths,
        "test_permission": store.get_test_permission(job_id),
        "search_result": search_result,
        "sequence_summary": sequence_summary,
        "winner_training_curves": winner.get("training_curves"),
    }
