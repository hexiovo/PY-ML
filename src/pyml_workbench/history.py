"""Durable, transactionally ordered history for S02 batch searches.

The store records metadata and JSON-safe results only. Fitted sessions and
dataset snapshots live in per-job artifact directories and are referenced by
fingerprint/path rather than copied into SQLite.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import time
from typing import Any, Iterator
import uuid


class HistoryError(RuntimeError):
    """Raised when a batch history operation violates its durable contract."""


class BudgetExhausted(HistoryError):
    """Raised before dispatch when a proposal or fit budget is exhausted."""


class DispatchStopped(HistoryError):
    """Raised when pause/cancel state prevents dispatching another trial."""


def _json_text(value: Any, *, field: str) -> str:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise HistoryError(f"{field} must be finite JSON data: {exc}") from exc


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _required_text(value: str, field: str) -> str:
    if not isinstance(value, str) or not value.strip() or chr(0) in value:
        raise HistoryError(f"{field} must be a non-empty string")
    return value


def _row_dict(row: sqlite3.Row | None) -> dict[str, Any] | None:
    return dict(row) if row is not None else None


class HistoryStore:
    """SQLite event and trial store with a monotonic, durable event sequence.

    Each operation opens its own connection so the same database can be used
    from the scheduler and worker processes. BEGIN IMMEDIATE serializes writes;
    mark_fit_started commits fit consumption before its caller enters fit code.
    """

    SCHEMA_VERSION = 4

    def __init__(self, path: str | Path):
        self.path = Path(path).expanduser().resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout=30000")
        connection.execute("PRAGMA foreign_keys=ON")
        return connection

    @contextmanager
    def _reader(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            yield connection
        finally:
            connection.close()

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        connection = self._connect()
        try:
            connection.execute("BEGIN IMMEDIATE")
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _initialize(self) -> None:
        connection = self._connect()
        try:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA synchronous=FULL")
            version = int(connection.execute("PRAGMA user_version").fetchone()[0])
            if version not in (0, 1, 2, 3, self.SCHEMA_VERSION):
                raise HistoryError(
                    f"Unsupported history schema {version}; expected {self.SCHEMA_VERSION}"
                )
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY,
                    dataset_id TEXT NOT NULL,
                    task TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    config_json TEXT NOT NULL DEFAULT '{}',
                    snapshot_json TEXT NOT NULL,
                    snapshot_sha256 TEXT NOT NULL,
                    snapshot_receipt_json TEXT,
                    sequence_plan_receipt_json TEXT,
                    split_json TEXT NOT NULL,
                    split_sha256 TEXT NOT NULL,
                    search_json TEXT NOT NULL,
                    budget_json TEXT NOT NULL,
                    fingerprint TEXT,
                    elapsed_seconds REAL NOT NULL DEFAULT 0 CHECK (elapsed_seconds >= 0),
                    artifact_dir TEXT NOT NULL UNIQUE,
                    status TEXT NOT NULL CHECK (status IN (
                        'queued', 'running', 'pausing', 'paused',
                        'cancel_requested', 'cancelled', 'completed', 'failed'
                    )),
                    control_request TEXT CHECK (
                        control_request IS NULL OR control_request IN ('pause', 'cancel')
                    ),
                    actual_fit_count INTEGER NOT NULL DEFAULT 0 CHECK (actual_fit_count >= 0),
                    proposal_count INTEGER NOT NULL DEFAULT 0 CHECK (proposal_count >= 0),
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS trials (
                    job_id TEXT NOT NULL REFERENCES jobs(job_id) ON DELETE CASCADE,
                    trial_id TEXT NOT NULL,
                    config_sha256 TEXT NOT NULL,
                    cache_key TEXT,
                    parameters_json TEXT NOT NULL,
                    objective_direction TEXT NOT NULL,
                    status TEXT NOT NULL CHECK (status IN (
                        'proposed', 'running', 'succeeded', 'failed', 'rejected',
                        'unscorable', 'cached', 'cached_failure'
                    )),
                    objective_value REAL,
                    metrics_json TEXT,
                    result_json TEXT,
                    error_text TEXT,
                    actual_fit_count INTEGER NOT NULL DEFAULT 0 CHECK (actual_fit_count >= 0),
                    cache_hit INTEGER NOT NULL DEFAULT 0 CHECK (cache_hit IN (0, 1)),
                    cached_from_trial_id TEXT,
                    event_sequence INTEGER,
                    started_at TEXT,
                    finished_at TEXT,
                    duration_seconds REAL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY (job_id, trial_id)
                );
                CREATE INDEX IF NOT EXISTS trials_cache_lookup
                    ON trials(job_id, config_sha256, status, cache_hit);
                CREATE INDEX IF NOT EXISTS trials_job_sequence
                    ON trials(job_id, event_sequence);
                CREATE TABLE IF NOT EXISTS events (
                    sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                    job_id TEXT NOT NULL REFERENCES jobs(job_id) ON DELETE CASCADE,
                    trial_id TEXT,
                    event_type TEXT NOT NULL,
                    at TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS events_by_job_sequence
                    ON events(job_id, sequence);
                CREATE TABLE IF NOT EXISTS test_permissions (
                    job_id TEXT PRIMARY KEY REFERENCES jobs(job_id) ON DELETE CASCADE,
                    final_session_id TEXT NOT NULL,
                    state TEXT NOT NULL CHECK (state IN ('consumed', 'completed', 'failed')),
                    consumed_at TEXT NOT NULL,
                    finished_at TEXT,
                    result_json TEXT,
                    error_text TEXT
                );
                CREATE TABLE IF NOT EXISTS batch_scheduler (
                    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
                    run_id TEXT NOT NULL UNIQUE,
                    max_workers INTEGER NOT NULL CHECK (max_workers IN (1, 2)),
                    pid INTEGER NOT NULL,
                    heartbeat_at REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS batch_worker_slots (
                    slot_id INTEGER PRIMARY KEY CHECK (slot_id IN (0, 1)),
                    run_id TEXT NOT NULL,
                    job_id TEXT NOT NULL REFERENCES jobs(job_id) ON DELETE CASCADE,
                    pid INTEGER NOT NULL,
                    heartbeat_at REAL NOT NULL,
                    UNIQUE(run_id, job_id)
                );
                """
            )
            # Additive migration for databases produced by the first S02 draft.
            job_columns = {row["name"] for row in connection.execute("PRAGMA table_info(jobs)")}
            if "config_json" not in job_columns:
                connection.execute("ALTER TABLE jobs ADD COLUMN config_json TEXT NOT NULL DEFAULT '{}'")
            if "fingerprint" not in job_columns:
                connection.execute("ALTER TABLE jobs ADD COLUMN fingerprint TEXT")
            if "snapshot_receipt_json" not in job_columns:
                connection.execute("ALTER TABLE jobs ADD COLUMN snapshot_receipt_json TEXT")
            if "sequence_plan_receipt_json" not in job_columns:
                connection.execute("ALTER TABLE jobs ADD COLUMN sequence_plan_receipt_json TEXT")
            if "elapsed_seconds" not in job_columns:
                connection.execute("ALTER TABLE jobs ADD COLUMN elapsed_seconds REAL NOT NULL DEFAULT 0")
            trial_columns = {row["name"] for row in connection.execute("PRAGMA table_info(trials)")}
            if "cache_key" not in trial_columns:
                connection.execute("ALTER TABLE trials ADD COLUMN cache_key TEXT")
            connection.execute(f"PRAGMA user_version={self.SCHEMA_VERSION}")
        finally:
            connection.close()

    def _append_event(
        self,
        connection: sqlite3.Connection,
        job_id: str,
        event_type: str,
        payload: Any,
        *,
        trial_id: str | None = None,
    ) -> int:
        cursor = connection.execute(
            "INSERT INTO events(job_id, trial_id, event_type, at, payload_json) "
            "VALUES (?, ?, ?, ?, ?)",
            (job_id, trial_id, event_type, _now(), _json_text(payload, field="event payload")),
        )
        return int(cursor.lastrowid)

    def create_job(
        self,
        *,
        job_id: str,
        dataset_id: str,
        task: str,
        model_id: str,
        snapshot: Any,
        split_summary: Any,
        search_spec: Any,
        budget: Any,
        artifact_dir: str | Path,
        config: Any = None,
        fingerprint: str | None = None,
        snapshot_receipt: Any = None,
        sequence_plan_receipt: Any = None,
    ) -> dict[str, Any]:
        job_id = _required_text(job_id, "job_id")
        dataset_id = _required_text(dataset_id, "dataset_id")
        task = _required_text(task, "task")
        model_id = _required_text(model_id, "model_id")
        snapshot_json = _json_text(snapshot, field="snapshot")
        split_json = _json_text(split_summary, field="split_summary")
        search_json = _json_text(search_spec, field="search_spec")
        config_json = _json_text(config or {}, field="config")
        budget_json = _json_text(budget, field="budget")
        snapshot_receipt_json = (
            None if snapshot_receipt is None
            else _json_text(snapshot_receipt, field="snapshot_receipt")
        )
        sequence_plan_receipt_json = (
            None if sequence_plan_receipt is None
            else _json_text(sequence_plan_receipt, field="sequence_plan_receipt")
        )
        if sequence_plan_receipt is not None and snapshot_receipt is None:
            raise HistoryError("An owned sequence plan receipt requires an owned snapshot receipt")
        if fingerprint is not None and (
            not isinstance(fingerprint, str)
            or len(fingerprint) != 64
            or any(character not in "0123456789abcdef" for character in fingerprint)
        ):
            raise HistoryError("fingerprint must be a lowercase SHA-256 digest")
        output = Path(artifact_dir).expanduser().resolve()
        if output.exists():
            snapshot_file = output / "snapshot.joblib"
            expected_artifacts = {"snapshot.joblib"}
            if sequence_plan_receipt is not None:
                expected_artifacts.update({"sequence_plan.joblib", "sequence_plan_receipt.json"})
            if (
                snapshot_receipt is None
                or not output.is_dir()
                or snapshot_file.is_symlink()
                or not snapshot_file.is_file()
                or {item.name for item in output.iterdir()} != expected_artifacts
            ):
                raise HistoryError(f"Artifact directory already exists; refusing overwrite: {output}")
        elif snapshot_receipt is not None:
            raise HistoryError("An owned snapshot receipt requires its persisted artifact directory")
        created = _now()
        with self._transaction() as connection:
            try:
                connection.execute(
                    """
                    INSERT INTO jobs(
                        job_id, dataset_id, task, model_id, config_json,
                        snapshot_json, snapshot_sha256, snapshot_receipt_json, sequence_plan_receipt_json,
                        split_json, split_sha256, search_json, budget_json,
                        fingerprint, artifact_dir, status, created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'queued', ?, ?)
                    """,
                    (
                        job_id, dataset_id, task, model_id, config_json, snapshot_json,
                        _sha256_text(snapshot_json), snapshot_receipt_json, sequence_plan_receipt_json,
                        split_json, _sha256_text(split_json), search_json, budget_json,
                        fingerprint, str(output), created, created,
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise HistoryError(f"Job ID or artifact directory already exists: {exc}") from exc
            sequence = self._append_event(
                connection,
                job_id,
                "job_created",
                {
                    "dataset_id": dataset_id,
                    "model_id": model_id,
                    "snapshot_sha256": _sha256_text(snapshot_json),
                    "split_sha256": _sha256_text(split_json),
                    "artifact_dir": str(output),
                },
            )
        job = self.get_job(job_id)
        if job is None:
            raise HistoryError(f"Job disappeared after creation: {job_id}")
        return job | {"event_sequence": sequence}

    def record_proposal(
        self,
        *,
        job_id: str,
        trial_id: str,
        parameters: Any,
        objective_direction: str,
        max_proposals: int,
        config_sha256: str | None = None,
        cache_key: str | None = None,
        result: Any = None,
    ) -> dict[str, Any]:
        job_id = _required_text(job_id, "job_id")
        trial_id = _required_text(trial_id, "trial_id")
        objective_direction = _required_text(objective_direction, "objective_direction")
        parameters_json = _json_text(parameters, field="parameters")
        calculated_hash = config_sha256 or _sha256_text(parameters_json)
        if len(calculated_hash) != 64 or any(character not in "0123456789abcdef" for character in calculated_hash):
            raise HistoryError("config_sha256 must be a lowercase SHA-256 digest")
        if cache_key is not None:
            cache_key = _required_text(cache_key, "cache_key")
        result_json = None if result is None else _json_text(result, field="trial result")
        now = _now()
        exhausted = False
        with self._transaction() as connection:
            job = connection.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
            if job is None:
                raise HistoryError(f"Unknown job_id: {job_id}")
            # A proposal is already inside the current safe-point interval. A
            # pause/cancel arriving now takes effect at the next dispatch gate.
            if job["status"] in {"paused", "cancelled", "completed", "failed"}:
                raise DispatchStopped(f"Job {job_id} is {job['status']}")
            if max_proposals <= 0:
                raise HistoryError("max_proposals must be positive")
            if int(job["proposal_count"]) >= max_proposals:
                self._append_event(
                    connection, job_id, "proposal_budget_exhausted",
                    {"max_proposals": max_proposals},
                )
                exhausted = True
            else:
                try:
                    connection.execute(
                        """
                        INSERT INTO trials(
                            job_id, trial_id, config_sha256, cache_key, parameters_json,
                            objective_direction, status, result_json, created_at, updated_at
                        ) VALUES (?, ?, ?, ?, ?, ?, 'proposed', ?, ?, ?)
                        """,
                        (job_id, trial_id, calculated_hash, cache_key, parameters_json,
                         objective_direction, result_json, now, now),
                    )
                except sqlite3.IntegrityError as exc:
                    raise HistoryError(f"Duplicate trial_id for job {job_id}: {trial_id}") from exc
                connection.execute(
                    "UPDATE jobs SET proposal_count=proposal_count+1, updated_at=? WHERE job_id=?",
                    (now, job_id),
                )
                sequence = self._append_event(
                    connection, job_id, "trial_proposed",
                    {"config_sha256": calculated_hash, "cache_key": cache_key,
                     "parameters": json.loads(parameters_json)},
                    trial_id=trial_id,
                )
                connection.execute(
                    "UPDATE trials SET event_sequence=?, updated_at=? WHERE job_id=? AND trial_id=?",
                    (sequence, now, job_id, trial_id),
                )
        if exhausted:
            raise BudgetExhausted(f"Job {job_id} reached {max_proposals} proposals")
        trial = self.get_trial(job_id, trial_id)
        if trial is None:
            raise HistoryError(f"Trial disappeared after proposal: {trial_id}")
        return trial

    def find_cached_trial(
        self, job_id: str, config_sha256: str | None = None, *, cache_key: str | None = None
    ) -> dict[str, Any] | None:
        lookup = cache_key or config_sha256
        if lookup is None:
            raise HistoryError("A cache key or configuration digest is required")
        column = "cache_key" if cache_key is not None else "config_sha256"
        with self._reader() as connection:
            row = connection.execute(
                f"""
                SELECT * FROM trials
                WHERE job_id=? AND {column}=? AND cache_hit=0
                    AND status IN ('succeeded', 'failed', 'unscorable')
                ORDER BY event_sequence DESC LIMIT 1
                """,
                (job_id, lookup),
            ).fetchone()
        return _row_dict(row)

    def record_cache_hit(
        self, *, job_id: str, trial_id: str, source_trial_id: str,
        cache_key: str | None = None, result: Any = None,
    ) -> dict[str, Any]:
        result_json = None if result is None else _json_text(result, field="cached trial result")
        now = _now()
        with self._transaction() as connection:
            source = connection.execute(
                "SELECT * FROM trials WHERE job_id=? AND trial_id=?",
                (job_id, source_trial_id),
            ).fetchone()
            if source is None or source["status"] not in {"succeeded", "failed", "unscorable"}:
                raise HistoryError("A cache hit must point to a completed trial in the same job")
            target = connection.execute(
                "SELECT * FROM trials WHERE job_id=? AND trial_id=?", (job_id, trial_id)
            ).fetchone()
            if target is None or target["status"] != "proposed":
                raise HistoryError("A cache hit must finalize a previously recorded proposal")
            if target["config_sha256"] != source["config_sha256"]:
                raise HistoryError("Cache source does not match the proposed configuration")
            effective_cache_key = cache_key or target["cache_key"]
            if effective_cache_key is None or effective_cache_key != source["cache_key"]:
                raise HistoryError("Cache source does not match the search fingerprint/configuration key")
            sequence = self._append_event(
                connection, job_id, "cache_hit",
                {"source_trial_id": source_trial_id, "config_sha256": source["config_sha256"]},
                trial_id=trial_id,
            )
            connection.execute(
                """
                UPDATE trials SET status=?, objective_value=?, metrics_json=?, result_json=?,
                    error_text=?, actual_fit_count=0, cache_hit=1, cached_from_trial_id=?,
                    event_sequence=?, finished_at=?, duration_seconds=0, updated_at=?
                WHERE job_id=? AND trial_id=?
                """,
                (
                    "cached_failure" if source["status"] == "failed" else "cached",
                    source["objective_value"], source["metrics_json"],
                    result_json if result_json is not None else source["result_json"],
                    source["error_text"], source_trial_id, sequence, now, now, job_id, trial_id,
                ),
            )
        trial = self.get_trial(job_id, trial_id)
        if trial is None:
            raise HistoryError(f"Cache trial disappeared after recording: {trial_id}")
        return trial

    def mark_fit_started(
        self, *, job_id: str, trial_id: str, max_actual_fits: int,
        elapsed_seconds: float | None = None,
    ) -> int:
        """Commit the fit budget and event before caller enters estimator.fit."""
        now = _now()
        exhausted = False
        with self._transaction() as connection:
            job = connection.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
            trial = connection.execute(
                "SELECT * FROM trials WHERE job_id=? AND trial_id=?", (job_id, trial_id)
            ).fetchone()
            if job is None or trial is None:
                raise HistoryError("Cannot start a fit for an unknown job/trial")
            if job["status"] in {"paused", "cancelled", "completed", "failed"}:
                raise DispatchStopped(f"Job {job_id} is {job['status']}")
            if trial["status"] != "proposed" or int(trial["actual_fit_count"]) != 0:
                raise HistoryError(f"Trial {trial_id} has already entered a fit or is not proposed")
            if max_actual_fits <= 0:
                raise HistoryError("max_actual_fits must be positive")
            current = int(job["actual_fit_count"])
            if current >= max_actual_fits:
                self._append_event(
                    connection, job_id, "fit_budget_exhausted",
                    {"max_actual_fits": max_actual_fits, "actual_fit_count": current},
                    trial_id=trial_id,
                )
                exhausted = True
            else:
                sequence = self._append_event(
                    connection, job_id, "fit_started",
                    {"actual_fit_number": current + 1, "config_sha256": trial["config_sha256"]},
                    trial_id=trial_id,
                )
                connection.execute(
                    """UPDATE jobs SET actual_fit_count=actual_fit_count+1, status='running',
                        elapsed_seconds=MAX(elapsed_seconds, COALESCE(?, elapsed_seconds)), updated_at=?
                    WHERE job_id=?""",
                    (elapsed_seconds, now, job_id),
                )
                connection.execute(
                    "UPDATE trials SET status='running', actual_fit_count=1, event_sequence=?, "
                    "started_at=?, updated_at=? WHERE job_id=? AND trial_id=?",
                    (sequence, now, now, job_id, trial_id),
                )
        if exhausted:
            raise BudgetExhausted(f"Job {job_id} reached {max_actual_fits} actual fits")
        return sequence

    def finish_trial(
        self,
        *,
        job_id: str,
        trial_id: str,
        status: str,
        objective_value: float | None = None,
        metrics: Any = None,
        result: Any = None,
        error: str | None = None,
        duration_seconds: float | None = None,
        cache_key: str | None = None,
    ) -> int:
        if status not in {"succeeded", "failed", "rejected", "unscorable"}:
            raise HistoryError("Trial finish status must be succeeded, failed, rejected, or unscorable")
        if duration_seconds is not None and duration_seconds < 0:
            raise HistoryError("duration_seconds cannot be negative")
        if status == "succeeded" and objective_value is None:
            raise HistoryError("A successful scored trial requires objective_value")
        metrics_json = None if metrics is None else _json_text(metrics, field="metrics")
        result_json = None if result is None else _json_text(result, field="result")
        now = _now()
        with self._transaction() as connection:
            trial = connection.execute(
                "SELECT status FROM trials WHERE job_id=? AND trial_id=?",
                (job_id, trial_id),
            ).fetchone()
            if trial is None or trial["status"] != "running":
                raise HistoryError(f"Trial {trial_id} is not running")
            sequence = self._append_event(
                connection, job_id, f"trial_{status}",
                {"objective_value": objective_value, "error": error},
                trial_id=trial_id,
            )
            connection.execute(
                """
                UPDATE trials SET status=?, objective_value=?, metrics_json=?,
                    result_json=?, error_text=?, finished_at=?, duration_seconds=?,
                    event_sequence=?, cache_key=COALESCE(?, cache_key), updated_at=?
                WHERE job_id=? AND trial_id=?
                """,
                (
                    status, objective_value, metrics_json, result_json, error,
                    now, duration_seconds, sequence, cache_key, now, job_id, trial_id,
                ),
            )
        return sequence

    def finish_unfit_trial(
        self, *, job_id: str, trial_id: str, status: str, result: Any, error: str | None = None
    ) -> int:
        """Close a rejected/unscorable proposal which never entered estimator.fit."""
        if status not in {"rejected", "unscorable"}:
            raise HistoryError("An unfit proposal must be rejected or unscorable")
        result_json = _json_text(result, field="trial result")
        objective_value = result.get("objective_value") if isinstance(result, dict) else None
        metrics = result.get("metrics") if isinstance(result, dict) else None
        metrics_json = None if metrics is None else _json_text(metrics, field="metrics")
        cache_key = result.get("cache_key") if isinstance(result, dict) else None
        now = _now()
        with self._transaction() as connection:
            trial = connection.execute(
                "SELECT status FROM trials WHERE job_id=? AND trial_id=?",
                (job_id, trial_id),
            ).fetchone()
            if trial is None or trial["status"] != "proposed":
                raise HistoryError(f"Trial {trial_id} is not an unstarted proposal")
            sequence = self._append_event(
                connection, job_id, f"trial_{status}",
                {"objective_value": objective_value, "error": error}, trial_id=trial_id,
            )
            connection.execute(
                """
                UPDATE trials SET status=?, objective_value=?, metrics_json=?, result_json=?,
                    error_text=?, cache_key=COALESCE(?, cache_key), finished_at=?,
                    duration_seconds=0, event_sequence=?, updated_at=?
                WHERE job_id=? AND trial_id=?
                """,
                (status, objective_value, metrics_json, result_json, error, cache_key,
                 now, sequence, now, job_id, trial_id),
            )
        return sequence

    def set_search_fingerprint(self, job_id: str, fingerprint: str) -> int:
        fingerprint = _required_text(fingerprint, "fingerprint")
        if len(fingerprint) != 64 or any(character not in "0123456789abcdef" for character in fingerprint):
            raise HistoryError("fingerprint must be a lowercase SHA-256 digest")
        now = _now()
        with self._transaction() as connection:
            job = connection.execute("SELECT fingerprint FROM jobs WHERE job_id=?", (job_id,)).fetchone()
            if job is None:
                raise HistoryError(f"Unknown job_id: {job_id}")
            if job["fingerprint"] not in {None, fingerprint}:
                raise HistoryError("Search fingerprint changed; this job cannot resume with another spec/data")
            connection.execute(
                "UPDATE jobs SET fingerprint=?, status='running', updated_at=? WHERE job_id=?",
                (fingerprint, now, job_id),
            )
            return self._append_event(connection, job_id, "search_started", {"fingerprint": fingerprint})

    def touch_elapsed(self, job_id: str, elapsed_seconds: float | None) -> None:
        if elapsed_seconds is None:
            return
        elapsed = float(elapsed_seconds)
        if elapsed < 0 or elapsed != elapsed or elapsed == float("inf"):
            raise HistoryError("elapsed_seconds must be finite and non-negative")
        with self._transaction() as connection:
            cursor = connection.execute(
                "UPDATE jobs SET elapsed_seconds=MAX(elapsed_seconds, ?), updated_at=? WHERE job_id=?",
                (elapsed, _now(), job_id),
            )
            if cursor.rowcount != 1:
                raise HistoryError(f"Unknown job_id: {job_id}")

    def begin_batch_run(self, max_workers: int) -> str:
        if isinstance(max_workers, bool) or max_workers not in {1, 2}:
            raise HistoryError("Batch worker count must be 1 or 2")
        now = time.time()
        run_id = hashlib.sha256(f"{os.getpid()}:{now}:{uuid.uuid4()}".encode()).hexdigest()
        with self._transaction() as connection:
            active = connection.execute("SELECT * FROM batch_scheduler WHERE singleton=1").fetchone()
            if active is not None and float(active["heartbeat_at"]) >= now - 15:
                raise HistoryError(
                    f"Batch queue is already active in process {active['pid']} with {active['max_workers']} worker(s)"
                )
            connection.execute("DELETE FROM batch_worker_slots")
            connection.execute("DELETE FROM batch_scheduler WHERE singleton=1")
            connection.execute(
                "INSERT INTO batch_scheduler(singleton, run_id, max_workers, pid, heartbeat_at) VALUES (1, ?, ?, ?, ?)",
                (run_id, max_workers, os.getpid(), now),
            )
        return run_id

    def heartbeat_batch_run(self, run_id: str) -> None:
        with self._transaction() as connection:
            cursor = connection.execute(
                "UPDATE batch_scheduler SET heartbeat_at=? WHERE singleton=1 AND run_id=?",
                (time.time(), run_id),
            )
            if cursor.rowcount != 1:
                raise HistoryError("Batch scheduler lease expired")

    def end_batch_run(self, run_id: str) -> None:
        with self._transaction() as connection:
            connection.execute("DELETE FROM batch_worker_slots WHERE run_id=?", (run_id,))
            connection.execute("DELETE FROM batch_scheduler WHERE singleton=1 AND run_id=?", (run_id,))

    def claim_worker_slot(self, run_id: str, job_id: str) -> int:
        now = time.time()
        with self._transaction() as connection:
            scheduler = connection.execute(
                "SELECT * FROM batch_scheduler WHERE singleton=1 AND run_id=?", (run_id,)
            ).fetchone()
            if scheduler is None:
                raise HistoryError("Batch scheduler lease is missing")
            connection.execute(
                "UPDATE batch_scheduler SET heartbeat_at=? WHERE singleton=1 AND run_id=?",
                (now, run_id),
            )
            used = {
                int(row["slot_id"])
                for row in connection.execute("SELECT slot_id FROM batch_worker_slots WHERE run_id=?", (run_id,))
            }
            slot = next((candidate for candidate in range(int(scheduler["max_workers"])) if candidate not in used), None)
            if slot is None:
                raise HistoryError("Configured batch worker limit is already in use")
            connection.execute(
                "INSERT INTO batch_worker_slots(slot_id, run_id, job_id, pid, heartbeat_at) VALUES (?, ?, ?, ?, ?)",
                (slot, run_id, job_id, os.getpid(), now),
            )
            return slot

    def heartbeat_worker_slot(self, run_id: str, slot_id: int, job_id: str) -> None:
        now = time.time()
        with self._transaction() as connection:
            cursor = connection.execute(
                "UPDATE batch_worker_slots SET heartbeat_at=? WHERE run_id=? AND slot_id=? AND job_id=?",
                (now, run_id, slot_id, job_id),
            )
            if cursor.rowcount != 1:
                raise HistoryError("Batch worker slot lease expired")
            connection.execute(
                "UPDATE batch_scheduler SET heartbeat_at=? WHERE singleton=1 AND run_id=?",
                (now, run_id),
            )

    def release_worker_slot(self, run_id: str, slot_id: int, job_id: str) -> None:
        with self._transaction() as connection:
            connection.execute(
                "DELETE FROM batch_worker_slots WHERE run_id=? AND slot_id=? AND job_id=?",
                (run_id, slot_id, job_id),
            )

    def record_search_progress(
        self, *, job_id: str, event_type: str, payload: Any, elapsed_seconds: float | None = None
    ) -> int:
        if elapsed_seconds is not None and (elapsed_seconds < 0 or not float(elapsed_seconds) < float("inf")):
            raise HistoryError("elapsed_seconds must be finite and non-negative")
        now = _now()
        with self._transaction() as connection:
            if connection.execute("SELECT 1 FROM jobs WHERE job_id=?", (job_id,)).fetchone() is None:
                raise HistoryError(f"Unknown job_id: {job_id}")
            sequence = self._append_event(connection, job_id, event_type, payload)
            if elapsed_seconds is not None:
                connection.execute(
                    "UPDATE jobs SET elapsed_seconds=MAX(elapsed_seconds, ?), updated_at=? WHERE job_id=?",
                    (float(elapsed_seconds), now, job_id),
                )
            return sequence

    def persist_search_result(self, *, job_id: str, result: dict[str, Any]) -> int:
        """Sync the complete result so interruption recovery includes all proposals."""
        trials = result.get("trials", [])
        if not isinstance(trials, list):
            raise HistoryError("Search result trials must be a list")
        with self._transaction() as connection:
            job = connection.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
            if job is None:
                raise HistoryError(f"Unknown job_id: {job_id}")
            for record in trials:
                trial_id = _required_text(str(record.get("trial_id", "")), "trial_id")
                row = connection.execute(
                    "SELECT status FROM trials WHERE job_id=? AND trial_id=?", (job_id, trial_id)
                ).fetchone()
                if row is None:
                    raise HistoryError(f"Search result contains unrecorded proposal {trial_id}")
                status_map = {
                    "complete": "succeeded", "failed": "failed", "invalid": "rejected",
                    "unscorable": "unscorable", "cached": "cached",
                }
                status = status_map.get(record.get("status"))
                if status is None:
                    raise HistoryError(f"Unsupported search trial status: {record.get('status')!r}")
                objective_value = record.get("objective_value")
                metrics_json = _json_text(record.get("metrics", {}), field="trial metrics")
                result_json = _json_text(record, field="trial result")
                now = _now()
                sequence = self._append_event(
                    connection, job_id, "trial_result_synced",
                    {"trial_id": trial_id, "status": status}, trial_id=trial_id,
                )
                connection.execute(
                    """
                    UPDATE trials SET status=?, objective_value=?, metrics_json=?, result_json=?,
                        error_text=?, cache_key=COALESCE(?, cache_key), cache_hit=?,
                        cached_from_trial_id=?, event_sequence=?,
                        finished_at=COALESCE(finished_at, ?),
                        duration_seconds=COALESCE(duration_seconds, ?), updated_at=?
                    WHERE job_id=? AND trial_id=?
                    """,
                    (
                        status, objective_value, metrics_json, result_json, record.get("error"),
                        record.get("cache_key"), int(bool(record.get("cache_hit"))),
                        record.get("cached_from"), sequence, now,
                        float(record.get("duration_seconds", 0.0)), now, job_id, trial_id,
                    ),
                )
            now = _now()
            elapsed = float(result.get("elapsed_seconds", job["elapsed_seconds"]))
            if elapsed < 0 or elapsed == float("inf") or elapsed != elapsed:
                raise HistoryError("Search result elapsed_seconds must be finite and non-negative")
            connection.execute(
                "UPDATE jobs SET actual_fit_count=?, proposal_count=?, elapsed_seconds=MAX(elapsed_seconds, ?), updated_at=? WHERE job_id=?",
                (int(result.get("actual_fit_count", 0)), int(result.get("proposal_count", 0)), elapsed, now, job_id),
            )
            return self._append_event(connection, job_id, "search_finished", {"result": result})

    def set_job_status(self, job_id: str, status: str, *, reason: str | None = None) -> int:
        allowed = {"queued", "running", "pausing", "paused", "cancel_requested",
                   "cancelled", "completed", "failed"}
        if status not in allowed:
            raise HistoryError(f"Unsupported job status: {status}")
        now = _now()
        with self._transaction() as connection:
            job = connection.execute("SELECT status FROM jobs WHERE job_id=?", (job_id,)).fetchone()
            if job is None:
                raise HistoryError(f"Unknown job_id: {job_id}")
            sequence = self._append_event(
                connection, job_id, "job_status",
                {"from": job["status"], "to": status, "reason": reason},
            )
            connection.execute(
                "UPDATE jobs SET status=?, updated_at=? WHERE job_id=?",
                (status, now, job_id),
            )
        return sequence

    def request_control(self, job_id: str, action: str) -> int:
        if action not in {"pause", "resume", "cancel"}:
            raise HistoryError("Control action must be pause, resume, or cancel")
        now = _now()
        with self._transaction() as connection:
            job = connection.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
            if job is None:
                raise HistoryError(f"Unknown job_id: {job_id}")
            current = job["status"]
            if action == "pause":
                if current != "running":
                    raise HistoryError(f"Cannot pause job in state {current}")
                status, request = "pausing", "pause"
            elif action == "resume":
                if current != "paused":
                    raise HistoryError(f"Cannot resume job in state {current}")
                status, request = "queued", None
            else:
                if current in {"completed", "cancelled", "failed"}:
                    raise HistoryError(f"Cannot cancel job in state {current}")
                status, request = "cancel_requested", "cancel"
            sequence = self._append_event(
                connection, job_id, f"control_{action}",
                {"from": current, "to": status},
            )
            connection.execute(
                "UPDATE jobs SET status=?, control_request=?, updated_at=? WHERE job_id=?",
                (status, request, now, job_id),
            )
        return sequence

    def control_request(self, job_id: str) -> str | None:
        job = self.get_job(job_id)
        if job is None:
            raise HistoryError(f"Unknown job_id: {job_id}")
        return job["control_request"]

    def consume_test_permission(self, *, job_id: str, final_session_id: str) -> bool:
        """Persist test consumption before touching test data; false means cached."""
        final_session_id = _required_text(final_session_id, "final_session_id")
        now = _now()
        with self._transaction() as connection:
            if connection.execute("SELECT 1 FROM jobs WHERE job_id=?", (job_id,)).fetchone() is None:
                raise HistoryError(f"Unknown job_id: {job_id}")
            permit = connection.execute(
                "SELECT * FROM test_permissions WHERE job_id=?", (job_id,)
            ).fetchone()
            if permit is not None:
                if permit["final_session_id"] != final_session_id:
                    raise HistoryError("The one test permission is already bound to another final session")
                if permit["state"] == "completed":
                    return False
                raise HistoryError("The test permission was consumed; failed/interrupted tests cannot be retried")
            connection.execute(
                "INSERT INTO test_permissions(job_id, final_session_id, state, consumed_at) "
                "VALUES (?, ?, 'consumed', ?)",
                (job_id, final_session_id, now),
            )
            self._append_event(
                connection, job_id, "test_permission_consumed",
                {"final_session_id": final_session_id},
            )
        return True

    def finish_test(
        self,
        *,
        job_id: str,
        final_session_id: str,
        result: Any = None,
        error: str | None = None,
    ) -> int:
        state = "failed" if error is not None else "completed"
        result_json = None if result is None else _json_text(result, field="test result")
        now = _now()
        with self._transaction() as connection:
            permit = connection.execute(
                "SELECT * FROM test_permissions WHERE job_id=?", (job_id,)
            ).fetchone()
            if permit is None or permit["final_session_id"] != final_session_id:
                raise HistoryError("No matching consumed test permission exists")
            if permit["state"] != "consumed":
                raise HistoryError("Test permission is already finalized and cannot be changed")
            sequence = self._append_event(
                connection, job_id, f"final_test_{state}",
                {"final_session_id": final_session_id, "error": error},
            )
            connection.execute(
                "UPDATE test_permissions SET state=?, finished_at=?, result_json=?, error_text=? "
                "WHERE job_id=?",
                (state, now, result_json, error, job_id),
            )
        return sequence

    def get_test_permission(self, job_id: str) -> dict[str, Any] | None:
        with self._reader() as connection:
            row = connection.execute(
                "SELECT * FROM test_permissions WHERE job_id=?", (job_id,)
            ).fetchone()
        return _row_dict(row)

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        with self._reader() as connection:
            row = connection.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
        return _row_dict(row)

    def list_jobs(self, *, status: str | None = None) -> list[dict[str, Any]]:
        with self._reader() as connection:
            if status is None:
                rows = connection.execute("SELECT * FROM jobs ORDER BY created_at, job_id").fetchall()
            else:
                rows = connection.execute(
                    "SELECT * FROM jobs WHERE status=? ORDER BY created_at, job_id", (status,)
                ).fetchall()
        return [dict(row) for row in rows]

    def get_trial(self, job_id: str, trial_id: str) -> dict[str, Any] | None:
        with self._reader() as connection:
            row = connection.execute(
                "SELECT * FROM trials WHERE job_id=? AND trial_id=?", (job_id, trial_id)
            ).fetchone()
        return _row_dict(row)

    def list_trials(self, job_id: str) -> list[dict[str, Any]]:
        with self._reader() as connection:
            rows = connection.execute(
                "SELECT * FROM trials WHERE job_id=? ORDER BY event_sequence, trial_id",
                (job_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def list_events(self, job_id: str, *, after_sequence: int = 0) -> list[dict[str, Any]]:
        with self._reader() as connection:
            rows = connection.execute(
                "SELECT * FROM events WHERE job_id=? AND sequence>? ORDER BY sequence",
                (job_id, after_sequence),
            ).fetchall()
        return [dict(row) for row in rows]

    def latest_sequence(self) -> int:
        with self._reader() as connection:
            return int(connection.execute("SELECT COALESCE(MAX(sequence), 0) FROM events").fetchone()[0])
