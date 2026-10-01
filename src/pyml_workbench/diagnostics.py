"""Bounded, per-process JSONL diagnostics for PY-ML.

Importing this module has no filesystem or process-global side effects. A
``LogSession`` is created explicitly by the application or worker owner.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import importlib.metadata
import json
import os
from pathlib import Path
import re
import stat
import threading
import time
import traceback as traceback_module
import uuid
from typing import Any, Iterable, Iterator, Mapping


SCHEMA_VERSION = 1
DEFAULT_ACTIVE_BYTES = 2 * 1024 * 1024
DEFAULT_BACKUP_COUNT = 4
DEFAULT_PROCESS_BYTES = DEFAULT_ACTIVE_BYTES * (DEFAULT_BACKUP_COUNT + 1)
DEFAULT_ROOT_BYTES = 100 * 1024 * 1024
DEFAULT_RETENTION_DAYS = 30
_MANIFEST_LIMIT = 64 * 1024
_SESSION_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")
_NONCE_RE = re.compile(r"^[0-9a-f]{32}$")
_ERROR_ID_RE = re.compile(r"^[0-9a-f]{32}$")
_SAFE_LABEL_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_SESSION_LOCK_MARKER = b"PYML-WORKBENCH-SESSION-LOCK-V1\n"
_ROOT_LOCK_MARKER = b"PYML-WORKBENCH-ROOT-LOCK-V1\n"
_ROOT_LOCK_NAME = ".pyml-workbench-root-budget.lock"
_EVENT_KEYS = frozenset(
    {
        "schema",
        "timestamp_utc",
        "level",
        "event",
        "session_id",
        "process_id",
        "process_nonce",
        "role",
        "version",
        "stage",
        "model",
        "job_id",
        "error_id",
        "error_type",
        "message",
        "traceback",
    }
)
_CONTEXT_KEYS = frozenset({"stage", "model", "job_id"})
_ACTIVE_SESSION_KEYS: set[str] = set()
_ACTIVE_SESSION_GUARD = threading.RLock()


class DiagnosticsError(ValueError):
    """Raised for invalid diagnostics paths or manifests."""


@dataclass(frozen=True)
class ErrorRecord:
    error_id: str
    timestamp_utc: str
    error_type: str
    message: str
    traceback: str | None
    stage: str | None
    model: str | None
    job_id: str | None
    process_id: int
    session_id: str
    process_nonce: str
    role: str
    version: str | None
    log_path: str | None

    def to_dict(self) -> dict[str, Any]:
        """Return the fixed event fields without adding arbitrary context."""
        return {
            "schema": SCHEMA_VERSION,
            "timestamp_utc": self.timestamp_utc,
            "level": "ERROR",
            "event": "error",
            "session_id": self.session_id,
            "process_id": self.process_id,
            "process_nonce": self.process_nonce,
            "role": self.role,
            "version": self.version,
            "stage": self.stage,
            "model": self.model,
            "job_id": self.job_id,
            "error_id": self.error_id,
            "error_type": self.error_type,
            "message": self.message,
            "traceback": self.traceback,
        }


@dataclass(frozen=True)
class CleanupResult:
    removed_sessions: tuple[str, ...] = ()
    removed_files: tuple[str, ...] = ()
    removed_bytes: int = 0
    retained_active_sessions: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()


@dataclass(frozen=True)
class _Manifest:
    session_dir: Path
    session_id: str
    process_id: int
    process_nonce: str
    role: str
    data: dict[str, Any]


class _LockBusy(OSError):
    pass


class _ByteRangeLock:
    """One-byte OS lock; Windows uses the standard-library msvcrt API."""

    def __init__(
        self,
        path: Path,
        *,
        blocking: bool,
        create: bool,
        marker: bytes,
    ) -> None:
        self.path = path
        self.fd: int | None = None
        self._locked = False
        flags = os.O_RDWR | getattr(os, "O_BINARY", 0)
        created = False
        try:
            if create:
                try:
                    self.fd = os.open(path, flags | os.O_CREAT | os.O_EXCL, 0o600)
                    created = True
                except FileExistsError:
                    self.fd = os.open(path, flags)
            else:
                self.fd = os.open(path, flags)
            self._lock(blocking)
            self._locked = True
            if created:
                os.lseek(self.fd, 0, os.SEEK_SET)
                os.write(self.fd, marker)
                os.fsync(self.fd)
            else:
                os.lseek(self.fd, 0, os.SEEK_SET)
                found = os.read(self.fd, len(marker))
                if found != marker:
                    raise DiagnosticsError("lock file marker is invalid")
        except Exception:
            self.close()
            raise

    def _lock(self, blocking: bool) -> None:
        assert self.fd is not None
        os.lseek(self.fd, 0, os.SEEK_SET)
        try:
            if os.name == "nt":
                import msvcrt

                mode = msvcrt.LK_LOCK if blocking else msvcrt.LK_NBLCK
                msvcrt.locking(self.fd, mode, 1)
            else:
                import fcntl

                operation = fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB)
                fcntl.flock(self.fd, operation)
        except OSError as exc:
            if not blocking:
                raise _LockBusy(str(exc)) from exc
            raise

    def close(self) -> None:
        fd, self.fd = self.fd, None
        if fd is None:
            return
        if self._locked:
            try:
                os.lseek(fd, 0, os.SEEK_SET)
                if os.name == "nt":
                    import msvcrt

                    msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(fd, fcntl.LOCK_UN)
            except OSError:
                pass
            self._locked = False
        try:
            os.close(fd)
        except OSError:
            pass

    def __enter__(self) -> "_ByteRangeLock":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _utc_text(value: datetime | None = None) -> str:
    value = value or _utc_now()
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def _parse_utc(value: object) -> datetime | None:
    if not isinstance(value, str):
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return None
    return parsed.astimezone(timezone.utc)


def _is_reparse_or_symlink(path: Path) -> bool:
    try:
        info = os.lstat(path)
    except FileNotFoundError:
        return False
    if stat.S_ISLNK(info.st_mode):
        return True
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return bool(getattr(info, "st_file_attributes", 0) & reparse_flag)


def _check_path_chain(path: Path, *, permit_missing: bool) -> Path:
    absolute = Path(os.path.abspath(os.fspath(path)))
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current = current / part
        if os.path.lexists(current):
            if _is_reparse_or_symlink(current):
                raise DiagnosticsError("diagnostics paths cannot cross symlinks or reparse points")
        elif not permit_missing:
            raise FileNotFoundError(current)
    return absolute


def _ensure_plain_directory(path: Path) -> Path:
    absolute = _check_path_chain(path, permit_missing=True)
    absolute.mkdir(parents=True, exist_ok=True)
    if _is_reparse_or_symlink(absolute) or not absolute.is_dir():
        raise DiagnosticsError("diagnostics root must be a regular directory")
    return absolute.resolve(strict=True)


def _plain_regular_file(path: Path, root: Path) -> bool:
    try:
        absolute = _check_path_chain(path, permit_missing=False)
        root_real = root.resolve(strict=True)
        file_real = absolute.resolve(strict=True)
        file_real.relative_to(root_real)
        info = os.lstat(absolute)
    except (OSError, ValueError, DiagnosticsError):
        return False
    return stat.S_ISREG(info.st_mode) and not _is_reparse_or_symlink(absolute)


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _safe_text(value: object, *, default: str = "", maximum: int = 4096) -> str:
    try:
        text = value if isinstance(value, str) else str(value)
    except Exception:
        text = default
    return text[:maximum]


def _safe_context(context: Mapping[str, Any] | None) -> dict[str, str | None]:
    selected: dict[str, str | None] = {key: None for key in _CONTEXT_KEYS}
    if not isinstance(context, Mapping):
        return selected
    for key in _CONTEXT_KEYS:
        value = context.get(key)
        if value is None:
            continue
        if isinstance(value, (str, int, float)) and not isinstance(value, bool):
            selected[key] = _safe_text(value, maximum=256)
    return selected


def _app_version() -> str | None:
    try:
        value = importlib.metadata.version("pyml-workbench")
    except importlib.metadata.PackageNotFoundError:
        value = os.environ.get("PYML_VERSION")
    if not isinstance(value, str) or not value or len(value) > 64:
        return None
    if any(ord(ch) < 32 for ch in value):
        return None
    return value


def _default_log_root() -> Path:
    override = os.environ.get("PYML_LOG_ROOT")
    if override:
        return Path(override)
    local_app_data = os.environ.get("LOCALAPPDATA")
    if local_app_data:
        return Path(local_app_data) / "PyMLWorkbench" / "logs"
    return Path.home() / "AppData" / "Local" / "PyMLWorkbench" / "logs"


def _session_key(path: Path) -> str:
    return os.path.normcase(os.path.abspath(os.fspath(path)))


def _safe_manifest_text(path: Path) -> dict[str, Any] | None:
    try:
        if not path.is_file() or _is_reparse_or_symlink(path):
            return None
        if path.stat().st_size > _MANIFEST_LIMIT:
            return None
        raw = path.read_text(encoding="utf-8")
        payload = json.loads(raw, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError, TypeError):
        return None
    return payload if isinstance(payload, dict) else None


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise ValueError(f"invalid JSON numeric constant: {value}")


def _validated_manifest(session_dir: Path, root: Path, expected_id: str | None = None) -> _Manifest | None:
    if _is_reparse_or_symlink(session_dir) or not session_dir.is_dir():
        return None
    manifest_path = session_dir / "manifest.json"
    if not _plain_regular_file(manifest_path, root):
        return None
    data = _safe_manifest_text(manifest_path)
    if data is None:
        return None
    required = {
        "schema",
        "owner",
        "session_id",
        "process",
        "role",
        "created_files",
        "created_at_utc",
        "updated_at_utc",
    }
    if set(data) != required:
        return None
    if type(data.get("schema")) is not int or data["schema"] != SCHEMA_VERSION:
        return None
    if data.get("owner") != "pyml-workbench":
        return None
    session_id = data.get("session_id")
    if not isinstance(session_id, str) or not _SESSION_ID_RE.fullmatch(session_id):
        return None
    if session_dir.name != session_id or (expected_id is not None and expected_id != session_id):
        return None
    process = data.get("process")
    if not isinstance(process, dict) or set(process) != {"pid", "nonce"}:
        return None
    process_id = process.get("pid")
    nonce = process.get("nonce")
    if type(process_id) is not int or process_id <= 0:
        return None
    if not isinstance(nonce, str) or not _NONCE_RE.fullmatch(nonce):
        return None
    role = data.get("role")
    if not isinstance(role, str) or not _SAFE_LABEL_RE.fullmatch(role):
        return None
    if _parse_utc(data.get("created_at_utc")) is None or _parse_utc(data.get("updated_at_utc")) is None:
        return None
    files = data.get("created_files")
    if not isinstance(files, list) or len(files) != 8:
        return None
    entries: dict[str, str] = {}
    for item in files:
        if not isinstance(item, dict) or set(item) != {"path", "kind"}:
            return None
        name, kind = item.get("path"), item.get("kind")
        if not isinstance(name, str) or not isinstance(kind, str):
            return None
        if name in entries or Path(name).name != name or "/" in name or "\\" in name or name in {".", ".."}:
            return None
        entries[name] = kind
    base = f"process-{process_id}-{nonce}.jsonl"
    expected_files = {
        "active.lock": "lock",
        "manifest.json": "manifest",
        "manifest.tmp": "temp",
        base: "jsonl",
        **{f"{base}.{index}": "jsonl" for index in range(1, DEFAULT_BACKUP_COUNT + 1)},
    }
    if entries != expected_files:
        return None
    return _Manifest(session_dir, session_id, process_id, nonce, role, data)


def _manifest_files(manifest: _Manifest, root: Path, *, include_control: bool = False) -> list[Path]:
    files = manifest.data["created_files"]
    results: list[Path] = []
    for entry in files:
        kind = entry["kind"]
        if kind not in {"jsonl", "manifest"} and not include_control:
            continue
        path = manifest.session_dir / entry["path"]
        if not os.path.lexists(path):
            continue
        if not _plain_regular_file(path, root):
            return []
        results.append(path)
    return results


def _enumerate_manifests(root: Path) -> list[_Manifest]:
    if not os.path.lexists(root) or _is_reparse_or_symlink(root) or not root.is_dir():
        return []
    try:
        children = list(root.iterdir())
    except OSError:
        return []
    result: list[_Manifest] = []
    for child in children:
        if not _SESSION_ID_RE.fullmatch(child.name):
            continue
        manifest = _validated_manifest(child, root, child.name)
        if manifest is not None:
            result.append(manifest)
    return result


def owned_log_files(log_root: os.PathLike[str] | str, session_id: str | None = None) -> tuple[Path, ...]:
    """Return only registered, regular app logs and manifests under ``log_root``.

    Session directories are enumerated one level deep. No glob or filename
    prefix alone is treated as proof of ownership.
    """
    try:
        root = _check_path_chain(Path(log_root), permit_missing=False).resolve(strict=True)
    except (OSError, ValueError, DiagnosticsError):
        return ()
    if _is_reparse_or_symlink(root) or not root.is_dir():
        return ()
    manifests: list[_Manifest]
    if session_id is not None:
        if not isinstance(session_id, str) or not _SESSION_ID_RE.fullmatch(session_id):
            return ()
        manifests = []
        manifest = _validated_manifest(root / session_id, root, session_id)
        if manifest is not None:
            manifests.append(manifest)
    else:
        manifests = _enumerate_manifests(root)
    result: list[Path] = []
    for manifest in manifests:
        result.extend(_manifest_files(manifest, root))
    return tuple(sorted(result, key=lambda item: (item.parent.name, item.name)))


def _lock_root(root: Path) -> _ByteRangeLock:
    path = root / _ROOT_LOCK_NAME
    if os.path.lexists(path) and _is_reparse_or_symlink(path):
        raise DiagnosticsError("root budget lock cannot be a symlink or reparse point")
    return _ByteRangeLock(path, blocking=True, create=True, marker=_ROOT_LOCK_MARKER)


def _session_lock_is_active(manifest: _Manifest) -> bool:
    key = _session_key(manifest.session_dir)
    with _ACTIVE_SESSION_GUARD:
        if key in _ACTIVE_SESSION_KEYS:
            return True
    lock_path = manifest.session_dir / "active.lock"
    if not _plain_regular_file(lock_path, manifest.session_dir.parent):
        return True
    try:
        lock = _ByteRangeLock(
            lock_path,
            blocking=False,
            create=False,
            marker=_SESSION_LOCK_MARKER,
        )
    except (_LockBusy, OSError, DiagnosticsError):
        return True
    lock.close()
    return False


def _root_usage(root: Path) -> int:
    total = 0
    for manifest in _enumerate_manifests(root):
        for path in _manifest_files(manifest, root, include_control=True):
            try:
                total += path.stat().st_size
            except OSError:
                continue
    return total


def _session_newest_mtime(manifest: _Manifest, root: Path) -> float:
    mtimes: list[float] = []
    for path in _manifest_files(manifest, root):
        try:
            mtimes.append(path.stat().st_mtime)
        except OSError:
            pass
    return max(mtimes, default=0.0)


def _delete_closed_session(manifest: _Manifest, root: Path) -> tuple[list[str], int]:
    key = _session_key(manifest.session_dir)
    with _ACTIVE_SESSION_GUARD:
        if key in _ACTIVE_SESSION_KEYS:
            return [], 0
    lock_path = manifest.session_dir / "active.lock"
    if not _plain_regular_file(lock_path, root):
        return [], 0
    try:
        lock = _ByteRangeLock(
            lock_path,
            blocking=False,
            create=False,
            marker=_SESSION_LOCK_MARKER,
        )
    except (_LockBusy, OSError, DiagnosticsError):
        return [], 0
    removed: list[str] = []
    removed_bytes = 0
    try:
        # Re-read ownership after acquiring the OS lock; never trust a stale scan.
        fresh = _validated_manifest(manifest.session_dir, root, manifest.session_id)
        if fresh is None:
            return [], 0
        registered = list(fresh.data["created_files"])
        paths: list[tuple[Path, str]] = []
        for entry in registered:
            path = fresh.session_dir / entry["path"]
            if not os.path.lexists(path):
                continue
            if not _plain_regular_file(path, root):
                return [], 0
            try:
                size = path.stat().st_size
            except OSError:
                return [], 0
            paths.append((path, size))
        # The active lock is released immediately before its unlink. The unique
        # directory still exists, so a new session cannot claim this ID.
        lock.close()
        for path, size in paths:
            if not _plain_regular_file(path, root):
                continue
            try:
                path.unlink()
            except OSError:
                continue
            removed.append(str(path))
            removed_bytes += size
        try:
            fresh.session_dir.rmdir()
        except OSError:
            pass
    finally:
        lock.close()
    return removed, removed_bytes


def _cleanup_under_root_lock(
    root: Path,
    *,
    now: datetime,
    retention_days: int,
    max_root_bytes: int,
) -> CleanupResult:
    cutoff = now - timedelta(days=retention_days)
    removed_sessions: list[str] = []
    removed_files: list[str] = []
    removed_bytes = 0
    retained_active: list[str] = []
    errors: list[str] = []

    candidates = _enumerate_manifests(root)
    for manifest in candidates:
        if _session_lock_is_active(manifest):
            retained_active.append(manifest.session_id)
            continue
        newest = _session_newest_mtime(manifest, root)
        if newest and datetime.fromtimestamp(newest, timezone.utc) < cutoff:
            paths, size = _delete_closed_session(manifest, root)
            if paths:
                removed_sessions.append(manifest.session_id)
                removed_files.extend(paths)
                removed_bytes += size

    usage = _root_usage(root)
    if usage > max_root_bytes:
        candidates = sorted(
            _enumerate_manifests(root),
            key=lambda item: (_session_newest_mtime(item, root), item.session_id),
        )
        for manifest in candidates:
            if usage <= max_root_bytes:
                break
            if _session_lock_is_active(manifest):
                if manifest.session_id not in retained_active:
                    retained_active.append(manifest.session_id)
                continue
            paths, size = _delete_closed_session(manifest, root)
            if paths:
                removed_sessions.append(manifest.session_id)
                removed_files.extend(paths)
                removed_bytes += size
                usage = max(0, usage - size)
        if usage > max_root_bytes:
            errors.append("owned log root is over budget because remaining sessions are active or unverifiable")
    return CleanupResult(
        tuple(removed_sessions),
        tuple(removed_files),
        removed_bytes,
        tuple(sorted(set(retained_active))),
        tuple(errors),
    )


def cleanup_owned_logs(
    log_root: os.PathLike[str] | str,
    *,
    now: datetime | None = None,
    retention_days: int = DEFAULT_RETENTION_DAYS,
    max_root_bytes: int = DEFAULT_ROOT_BYTES,
) -> CleanupResult:
    """Prune only registered sessions whose byte-range lock is free."""
    if type(retention_days) is not int or retention_days < 0:
        raise ValueError("retention_days must be a non-negative integer")
    if type(max_root_bytes) is not int or max_root_bytes < 1:
        raise ValueError("max_root_bytes must be a positive integer")
    timestamp = now or _utc_now()
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    root = _ensure_plain_directory(Path(log_root))
    with _lock_root(root):
        return _cleanup_under_root_lock(
            root,
            now=timestamp.astimezone(timezone.utc),
            retention_days=retention_days,
            max_root_bytes=max_root_bytes,
        )


def _event_from_json(value: Any, *, session_id: str) -> dict[str, Any] | None:
    if not isinstance(value, dict) or set(value) != _EVENT_KEYS:
        return None
    if type(value.get("schema")) is not int or value["schema"] != SCHEMA_VERSION:
        return None
    if value.get("session_id") != session_id:
        return None
    if type(value.get("process_id")) is not int or value["process_id"] <= 0:
        return None
    for key in ("timestamp_utc", "level", "event", "process_nonce", "role"):
        if not isinstance(value.get(key), str):
            return None
    if _parse_utc(value["timestamp_utc"]) is None:
        return None
    for key in ("version", "stage", "model", "job_id", "error_id", "error_type", "message", "traceback"):
        if value.get(key) is not None and not isinstance(value.get(key), str):
            return None
    return value


def read_records(
    log_root: os.PathLike[str] | str,
    *,
    session_id: str | None = None,
    error_ids: Iterable[str] | None = None,
    level: str | None = None,
    event: str | None = None,
    since_utc: datetime | str | None = None,
    until_utc: datetime | str | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Read and filter validated JSONL records from registered sessions."""
    owned = owned_log_files(log_root, session_id)
    selected_ids = None if error_ids is None else {str(item) for item in error_ids}
    if selected_ids == set():
        return []
    since = _parse_utc(since_utc) if isinstance(since_utc, str) else since_utc
    until = _parse_utc(until_utc) if isinstance(until_utc, str) else until_utc
    if isinstance(since_utc, str) and since is None:
        return []
    if isinstance(until_utc, str) and until is None:
        return []
    if isinstance(since, datetime) and since.tzinfo is None:
        since = since.replace(tzinfo=timezone.utc)
    if isinstance(until, datetime) and until.tzinfo is None:
        until = until.replace(tzinfo=timezone.utc)
    requested_level = level.upper() if isinstance(level, str) else None
    requested_event = event.casefold() if isinstance(event, str) else None
    records: list[dict[str, Any]] = []
    for path in owned:
        if path.name == "manifest.json" or ".jsonl" not in path.name:
            continue
        found_session = path.parent.name
        try:
            with path.open("r", encoding="utf-8", errors="strict") as stream:
                for line in stream:
                    try:
                        payload = json.loads(line, object_pairs_hook=_unique_object, parse_constant=_reject_constant)
                    except (json.JSONDecodeError, UnicodeError, ValueError):
                        continue
                    row = _event_from_json(payload, session_id=found_session)
                    if row is None:
                        continue
                    if selected_ids is not None and row.get("error_id") not in selected_ids:
                        continue
                    if requested_level is not None and row["level"].upper() != requested_level:
                        continue
                    if requested_event is not None and row["event"].casefold() != requested_event:
                        continue
                    stamp = _parse_utc(row["timestamp_utc"])
                    assert stamp is not None
                    if isinstance(since, datetime) and stamp < since.astimezone(timezone.utc):
                        continue
                    if isinstance(until, datetime) and stamp > until.astimezone(timezone.utc):
                        continue
                    records.append(row)
        except (OSError, UnicodeError):
            continue
    records.sort(key=lambda row: row["timestamp_utc"])
    if limit is not None:
        if type(limit) is not int or limit < 1:
            return []
        records = records[-limit:]
    return records


class LogSession:
    """Thread-safe JSONL logger and cleanup owner for a single process."""

    def __init__(
        self,
        log_root: os.PathLike[str] | str | None = None,
        *,
        session_id: str | None = None,
        role: str = "gui",
        context: Mapping[str, Any] | None = None,
        active_bytes: int = DEFAULT_ACTIVE_BYTES,
        backup_count: int = DEFAULT_BACKUP_COUNT,
        root_bytes: int = DEFAULT_ROOT_BYTES,
        retention_days: int = DEFAULT_RETENTION_DAYS,
    ) -> None:
        self.log_root = Path(log_root) if log_root is not None else _default_log_root()
        self.session_id = session_id or os.environ.get("PYML_LOG_SESSION_ID") or uuid.uuid4().hex
        self.process_id = os.getpid()
        self.process_nonce = uuid.uuid4().hex
        self.role = role
        self.version = _app_version()
        self.context = _safe_context(context)
        self.active_bytes = active_bytes
        self.backup_count = backup_count
        self.root_bytes = root_bytes
        self.retention_days = retention_days
        self.session_dir: Path | None = None
        self.log_dir: Path | None = None
        self.log_path: Path | None = None
        self.available = False
        self.last_error: str | None = None
        self._active_lock: _ByteRangeLock | None = None
        self._mutex = threading.RLock()
        self._closed = False
        self._last_cleanup = 0.0
        self._created_paths: list[Path] = []

        if not self._valid_options():
            self.last_error = "invalid diagnostics session settings"
            return
        root_lock: _ByteRangeLock | None = None
        try:
            self.log_root = _ensure_plain_directory(self.log_root)
            self.log_dir = self.log_root
            root_lock = _lock_root(self.log_root)
            session_dir = self.log_root / self.session_id
            if os.path.lexists(session_dir):
                raise FileExistsError("session_id already exists; each process needs a unique session")
            session_dir.mkdir()
            self._created_paths.append(session_dir)
            if _is_reparse_or_symlink(session_dir):
                raise DiagnosticsError("session directory cannot be a reparse point")
            self.session_dir = session_dir.resolve(strict=True)
            active_lock_path = self.session_dir / "active.lock"
            self._created_paths.append(active_lock_path)
            self._active_lock = _ByteRangeLock(
                active_lock_path,
                blocking=False,
                create=True,
                marker=_SESSION_LOCK_MARKER,
            )
            self._created_paths.append(active_lock_path)

            base_name = self._active_name()
            log_names = [base_name] + [f"{base_name}.{index}" for index in range(1, self.backup_count + 1)]
            for name in log_names:
                path = self.session_dir / name
                fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0), 0o600)
                os.fsync(fd)
                os.close(fd)
                self._created_paths.append(path)
            self.log_path = self.session_dir / base_name
            manifest = self._build_manifest(log_names)
            manifest_path = self.session_dir / "manifest.json"
            manifest_tmp = self.session_dir / "manifest.tmp"
            with manifest_tmp.open("xb") as stream:
                stream.write(_canonical_json(manifest))
                stream.flush()
                os.fsync(stream.fileno())
            self._created_paths.append(manifest_tmp)
            os.replace(manifest_tmp, manifest_path)
            self._created_paths.append(manifest_path)
            self._fsync_directory(self.session_dir)
            key = _session_key(self.session_dir)
            with _ACTIVE_SESSION_GUARD:
                _ACTIVE_SESSION_KEYS.add(key)
            self.available = True
            self._last_cleanup = time.monotonic()
            _cleanup_under_root_lock(
                self.log_root,
                now=_utc_now(),
                retention_days=self.retention_days,
                max_root_bytes=self.root_bytes,
            )
        except Exception as exc:
            self.last_error = self._error_summary(exc)
            self.available = False
            self._rollback_setup()
        finally:
            if root_lock is not None:
                root_lock.close()

    def _valid_options(self) -> bool:
        if not isinstance(self.session_id, str) or not _SESSION_ID_RE.fullmatch(self.session_id):
            return False
        if not isinstance(self.role, str) or not _SAFE_LABEL_RE.fullmatch(self.role):
            return False
        if type(self.active_bytes) is not int or self.active_bytes < 1024:
            return False
        if type(self.backup_count) is not int or self.backup_count != DEFAULT_BACKUP_COUNT:
            return False
        if type(self.root_bytes) is not int or self.root_bytes < 1:
            return False
        if type(self.retention_days) is not int or self.retention_days < 0:
            return False
        return True

    def _error_summary(self, exc: Exception) -> str:
        return f"{type(exc).__name__}: {_safe_text(exc, maximum=512)}"

    def _rollback_setup(self) -> None:
        if self.session_dir is None:
            return
        if self._active_lock is not None:
            self._active_lock.close()
            self._active_lock = None
        with _ACTIVE_SESSION_GUARD:
            _ACTIVE_SESSION_KEYS.discard(_session_key(self.session_dir))
        # Remove only paths this constructor created, after verifying none was
        # replaced with a link/reparse point.
        for path in reversed(self._created_paths):
            try:
                if path.is_dir() and not _is_reparse_or_symlink(path):
                    path.rmdir()
                elif os.path.lexists(path) and not _is_reparse_or_symlink(path):
                    path.unlink()
            except OSError:
                pass
        try:
            self.session_dir.rmdir()
        except OSError:
            pass
        self.session_dir = None
        self.log_path = None

    def _active_name(self) -> str:
        return f"process-{self.process_id}-{self.process_nonce}.jsonl"

    def _build_manifest(self, log_names: list[str]) -> dict[str, Any]:
        created = [
            {"path": "active.lock", "kind": "lock"},
            *({"path": name, "kind": "jsonl"} for name in log_names),
            {"path": "manifest.json", "kind": "manifest"},
            {"path": "manifest.tmp", "kind": "temp"},
        ]
        stamp = _utc_text()
        return {
            "schema": SCHEMA_VERSION,
            "owner": "pyml-workbench",
            "session_id": self.session_id,
            "process": {"pid": self.process_id, "nonce": self.process_nonce},
            "role": self.role,
            "created_files": created,
            "created_at_utc": stamp,
            "updated_at_utc": stamp,
        }

    @staticmethod
    def _fsync_directory(path: Path) -> None:
        if os.name == "nt":
            return
        try:
            fd = os.open(path, os.O_RDONLY)
        except OSError:
            return
        try:
            os.fsync(fd)
        except OSError:
            pass
        finally:
            os.close(fd)

    def _record_base(self, *, context: Mapping[str, Any] | None = None) -> dict[str, Any]:
        selected = dict(self.context)
        selected.update(_safe_context(context))
        return {
            "schema": SCHEMA_VERSION,
            "timestamp_utc": _utc_text(),
            "level": "INFO",
            "event": "event",
            "session_id": self.session_id,
            "process_id": self.process_id,
            "process_nonce": self.process_nonce,
            "role": self.role,
            "version": self.version,
            "stage": selected["stage"],
            "model": selected["model"],
            "job_id": selected["job_id"],
            "error_id": None,
            "error_type": None,
            "message": None,
            "traceback": None,
        }

    def _persist(self, row: dict[str, Any]) -> None:
        try:
            line = _canonical_json(row) + b"\n"
            if len(line) > self.active_bytes:
                raise OSError("diagnostic record exceeds the active log size limit")
            with self._mutex:
                if self._closed or not self.available or self.session_dir is None or self.log_path is None:
                    return
                root = self.log_root
                with _lock_root(root):
                    active_size = self.log_path.stat().st_size
                    if active_size + len(line) > self.active_bytes:
                        self._rotate()
                    usage = _root_usage(root)
                    if usage + len(line) > self.root_bytes:
                        _cleanup_under_root_lock(
                            root,
                            now=_utc_now(),
                            retention_days=self.retention_days,
                            max_root_bytes=self.root_bytes - len(line),
                        )
                        usage = _root_usage(root)
                    if usage + len(line) > self.root_bytes:
                        raise OSError("diagnostics log root is at its owned storage budget")
                    self._append_line(self.log_path, line)
                    self._last_cleanup = time.monotonic()
        except Exception as exc:
            self.available = False
            self.last_error = self._error_summary(exc)

    def _append_line(self, path: Path, line: bytes) -> None:
        if not _plain_regular_file(path, self.log_root):
            raise DiagnosticsError("active log file is missing or is not a regular owned file")
        fd = os.open(path, os.O_WRONLY | os.O_APPEND | getattr(os, "O_BINARY", 0))
        try:
            with os.fdopen(fd, "ab", closefd=False) as stream:
                stream.write(line)
                stream.flush()
                os.fsync(stream.fileno())
        finally:
            os.close(fd)

    def _rotate(self) -> None:
        assert self.session_dir is not None and self.log_path is not None
        base_name = self._active_name()
        paths = [self.session_dir / f"{base_name}.{index}" for index in range(1, self.backup_count + 1)]
        for path in [self.log_path, *paths]:
            if os.path.lexists(path) and not _plain_regular_file(path, self.log_root):
                raise DiagnosticsError("rotation target is not a registered regular log file")
        oldest = paths[-1]
        if oldest.exists():
            oldest.unlink()
        for index in range(self.backup_count - 1, 0, -1):
            source, target = paths[index - 1], paths[index]
            if source.exists():
                os.replace(source, target)
        if self.log_path.exists():
            os.replace(self.log_path, paths[0])
        fd = os.open(
            self.log_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0),
            0o600,
        )
        os.fsync(fd)
        os.close(fd)
        self._fsync_directory(self.session_dir)

    @staticmethod
    def _record_error_fields(exc: BaseException) -> tuple[str, str, str]:
        error_type = type(exc).__name__
        try:
            message = str(exc)
        except Exception:
            message = "<exception message unavailable>"
        try:
            full_traceback = "".join(
                traceback_module.format_exception(type(exc), exc, exc.__traceback__)
            )
        except Exception:
            full_traceback = f"{error_type}: {message}"
        return error_type, message, full_traceback

    def _error_record(
        self,
        error_type: str,
        message: str,
        traceback_text: str | None,
        *,
        context: Mapping[str, Any] | None,
        error_id: str | None = None,
    ) -> ErrorRecord:
        selected = dict(self.context)
        selected.update(_safe_context(context))
        identity = error_id if isinstance(error_id, str) and _ERROR_ID_RE.fullmatch(error_id) else uuid.uuid4().hex
        safe_traceback = (
            traceback_text
            if traceback_text is None or isinstance(traceback_text, str)
            else _safe_text(traceback_text, maximum=2**31 - 1)
        )
        try:
            safe_message = message if isinstance(message, str) else str(message)
        except Exception:
            safe_message = "<message unavailable>"
        return ErrorRecord(
            error_id=identity,
            timestamp_utc=_utc_text(),
            error_type=_safe_text(error_type, default="UnknownError", maximum=256),
            message=safe_message,
            traceback=safe_traceback,
            stage=selected["stage"],
            model=selected["model"],
            job_id=selected["job_id"],
            process_id=self.process_id,
            session_id=self.session_id,
            process_nonce=self.process_nonce,
            role=self.role,
            version=self.version,
            log_path=str(self.log_path) if self.log_path is not None else None,
        )

    def _persist_error(self, record: ErrorRecord, *, event: str = "error") -> None:
        row = self._record_base()
        row.update(
            {
                "timestamp_utc": record.timestamp_utc,
                "level": "ERROR",
                "event": event,
                "stage": record.stage,
                "model": record.model,
                "job_id": record.job_id,
                "error_id": record.error_id,
                "error_type": record.error_type,
                "message": record.message,
                "traceback": record.traceback,
            }
        )
        self._persist(row)

    def record_error(
        self,
        exc: BaseException,
        *,
        context: Mapping[str, Any] | None = None,
    ) -> ErrorRecord:
        """Persist an exception traceback and return its stable identifier."""
        error_type, message, full_traceback = self._record_error_fields(exc)
        record = self._error_record(
            error_type,
            message,
            full_traceback,
            context=context,
        )
        self._persist_error(record)
        return record

    def record_external_error(
        self,
        error_type: str,
        message: str,
        traceback_text: str | None = None,
        *,
        context: Mapping[str, Any] | None = None,
        error_id: str | None = None,
    ) -> ErrorRecord:
        """Persist stderr or another process's already-captured error details."""
        record = self._error_record(
            error_type,
            message,
            traceback_text,
            context=context,
            error_id=error_id,
        )
        self._persist_error(record)
        return record

    def record_event(
        self,
        event: str,
        *,
        level: str = "INFO",
        message: str | None = None,
        context: Mapping[str, Any] | None = None,
    ) -> None:
        if not isinstance(event, str) or not _SAFE_LABEL_RE.fullmatch(event):
            return
        allowed_levels = {"DEBUG", "INFO", "WARNING", "WARN", "ERROR", "CRITICAL"}
        chosen_level = level.upper() if isinstance(level, str) else "INFO"
        if chosen_level not in allowed_levels:
            chosen_level = "INFO"
        if chosen_level == "WARN":
            chosen_level = "WARNING"
        row = self._record_base(context=context)
        row.update(
            {
                "event": event,
                "level": chosen_level,
                "message": None if message is None else _safe_text(message),
            }
        )
        self._persist(row)

    def record_process_exit(
        self,
        returncode: int | None,
        *,
        context: Mapping[str, Any] | None = None,
        error_id: str | None = None,
    ) -> ErrorRecord | None:
        if returncode == 0:
            self.record_event("process_exit", message="process exited successfully", context=context)
            return None
        message = "Worker process exited" if returncode is None else f"Worker process exited with code {returncode}"
        record = self._error_record(
            "WorkerProcessExit",
            message,
            None,
            context=context,
            error_id=error_id,
        )
        self._persist_error(record, event="process_exit")
        return record

    def record_cancelled(self, *, context: Mapping[str, Any] | None = None) -> None:
        self.record_event("cancelled", level="INFO", message="operation cancelled", context=context)

    def close(self) -> None:
        with self._mutex:
            if self._closed:
                return
            self._closed = True
            self.available = False
            if self._active_lock is not None:
                self._active_lock.close()
                self._active_lock = None
            if self.session_dir is not None:
                with _ACTIVE_SESSION_GUARD:
                    _ACTIVE_SESSION_KEYS.discard(_session_key(self.session_dir))

    def __enter__(self) -> "LogSession":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


__all__ = [
    "CleanupResult",
    "DEFAULT_ACTIVE_BYTES",
    "DEFAULT_BACKUP_COUNT",
    "DEFAULT_PROCESS_BYTES",
    "DEFAULT_RETENTION_DAYS",
    "DEFAULT_ROOT_BYTES",
    "DiagnosticsError",
    "ErrorRecord",
    "LogSession",
    "SCHEMA_VERSION",
    "cleanup_owned_logs",
    "owned_log_files",
    "read_records",
]
