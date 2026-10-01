"""Build a user-triggered, allowlisted archive of owned diagnostics.

This module has no import-time filesystem effects and imports the diagnostics
service lazily. It never accepts caller-provided log paths: files must be
registered by the application's per-session manifest and returned by
``diagnostics.owned_log_files``.
"""
from __future__ import annotations

import json
import os
import platform
import re
import stat
import sys
import tempfile
import zipfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Iterable


MAX_ARCHIVE_BYTES = 20 * 1024 * 1024
MAX_SESSION_LOG_BYTES = 10 * 1024 * 1024
MAX_SESSION_LOG_FILES = 5
MAX_MANIFEST_BYTES = 64 * 1024
MANIFEST_KEYS = frozenset(
    {
        "schema",
        "owner",
        "session_id",
        "process",
        "role",
        "created_files",
        "created_at_utc",
        "updated_at_utc",
    }
)
PROCESS_KEYS = frozenset({"pid", "nonce"})
CREATED_FILE_KEYS = frozenset({"path", "kind"})
DIAGNOSTIC_KEYS = frozenset(
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
SUMMARY_KEYS = (
    "schema",
    "timestamp_utc",
    "level",
    "event",
    "session_id",
    "role",
    "version",
    "stage",
    "model",
    "job_id",
    "error_id",
    "error_type",
    "message",
)
_SESSION_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")
_NONCE_RE = re.compile(r"^[0-9a-f]{32}$")
_ROLE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")
_ERROR_ID_RE = re.compile(r"^[0-9a-f]{32}$")
_JSONL_NAME_RE = re.compile(r"^.+\.jsonl(?:\.[1-4])?$")

_CREDENTIAL_ASSIGNMENT_RE = re.compile(
    r"(?i)([\"']?(?:authorization|proxy-authorization|api[_-]?key|access[_-]?token|"
    r"refresh[_-]?token|client[_-]?secret|secret[_-]?access[_-]?key|"
    r"private[_-]?key|password|passwd|secret|token)[\"']?"
    r"\s*[:=]\s*)(\"[^\"]*\"|'[^']*'|[^\s,;]+)"
)
_AUTH_SCHEME_RE = re.compile(r"(?i)\b(Bearer|Basic)\s+[A-Za-z0-9._~+/=-]+")
_URL_CREDENTIALS_RE = re.compile(r"(?i)(https?://)[^/\s:@]+:[^/\s@]+@")
_KNOWN_TOKEN_RE = re.compile(
    r"\b(?:sk-[A-Za-z0-9_-]{16,}|sk_(?:live|test)_[A-Za-z0-9]{12,}|"
    r"rk_(?:live|test)_[A-Za-z0-9]{12,}|gh[pousr]_[A-Za-z0-9_]{20,}|"
    r"github_pat_[A-Za-z0-9_]{20,}|eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\."
    r"[A-Za-z0-9_-]{10,}|xox[baprs]-[A-Za-z0-9-]{10,}|ya29\.[A-Za-z0-9_-]{20,}|"
    r"AKIA[A-Z0-9]{16}|ASIA[A-Z0-9]{16}|AIza[A-Za-z0-9_-]{35}|npm_[A-Za-z0-9]{36})\b"
)
_PRIVATE_KEY_RE = re.compile(
    r"(?is)-----BEGIN (?:RSA |EC |DSA |OPENSSH |PGP )?PRIVATE KEY(?: BLOCK)?-----"
    r".*?-----END (?:RSA |EC |DSA |OPENSSH |PGP )?PRIVATE KEY(?: BLOCK)?-----"
)
_QUOTED_LOCAL_PATH_RE = re.compile(
    r"(?i)([\"'])(?:[a-z]:\\|\\\\|/(?!/))[^\"']*\1"
)
_WINDOWS_PATH_RE = re.compile(
    r"(?i)(?<![A-Za-z0-9])(?:[a-z]:\\|\\\\)[^\"'<>|,;\r\n\)\]}]+"
)
_POSIX_PATH_RE = re.compile(
    r"(?<![A-Za-z0-9+./-])/(?!/)[^\"'<>|,;\r\n\)\]}]+"
)
_HOME_PATH_RE = re.compile(r"(?<!\S)~(?:[/\\][^\s\"'<>|,;\)\]}]*)?")


class DiagnosticArchiveError(ValueError):
    """Raised when an archive request or an owned diagnostic source is unsafe."""


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DiagnosticArchiveError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise DiagnosticArchiveError(f"Non-finite JSON value is not allowed: {value}")


def _parse_json(data: bytes, *, label: str) -> Any:
    try:
        return json.loads(
            data.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except DiagnosticArchiveError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise DiagnosticArchiveError(f"{label} is not valid UTF-8 JSON") from exc


def _is_link_or_reparse(path: Path) -> bool:
    try:
        info = path.lstat()
    except OSError as exc:
        raise DiagnosticArchiveError(f"Could not inspect path: {path}") from exc
    if stat.S_ISLNK(info.st_mode):
        return True
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x400)
    return bool(getattr(info, "st_file_attributes", 0) & reparse_flag)


def _assert_no_link_components(path: Path, *, include_leaf: bool = True) -> None:
    absolute = Path(os.path.abspath(os.fspath(path)))
    components = list(reversed(absolute.parents)) + [absolute]
    if not include_leaf:
        components = components[:-1]
    for component in components:
        try:
            component.lstat()
        except FileNotFoundError as exc:
            raise DiagnosticArchiveError(f"Path does not exist: {component}") from exc
        except OSError as exc:
            raise DiagnosticArchiveError(f"Could not inspect path: {component}") from exc
        if _is_link_or_reparse(component):
            raise DiagnosticArchiveError(f"Links and reparse paths are not allowed: {component}")


def _checked_log_root(log_root: str | os.PathLike[str]) -> Path:
    try:
        raw = os.fspath(log_root)
    except TypeError as exc:
        raise DiagnosticArchiveError("log_root must be a filesystem path") from exc
    if not isinstance(raw, str) or not raw or "\x00" in raw:
        raise DiagnosticArchiveError("log_root must be a non-empty text path")
    root = Path(os.path.abspath(raw))
    _assert_no_link_components(root)
    if not root.is_dir():
        raise DiagnosticArchiveError("log_root must be a directory")
    try:
        return root.resolve(strict=True)
    except OSError as exc:
        raise DiagnosticArchiveError("Could not resolve log_root") from exc


def _is_within(root: Path, path: Path) -> bool:
    try:
        return os.path.commonpath((str(root), str(path))) == str(root)
    except (ValueError, OSError):
        return False


def _safe_member_name(name: str) -> str:
    if not isinstance(name, str) or not name or "\x00" in name:
        raise DiagnosticArchiveError("Invalid registered log filename")
    if len(name) > 255 or any(ord(char) < 32 for char in name):
        raise DiagnosticArchiveError("Registered log filename contains unsafe characters")
    if "/" in name or "\\" in name or ":" in name:
        raise DiagnosticArchiveError("Registered log path must be a basename")
    if name in {".", ".."} or PurePosixPath(name).name != name:
        raise DiagnosticArchiveError("Registered log path is unsafe")
    if not _JSONL_NAME_RE.fullmatch(name):
        raise DiagnosticArchiveError("Only registered JSONL diagnostics may be archived")
    return name


def _validate_manifest(path: Path, *, session_id: str) -> tuple[set[str], Path]:
    _assert_no_link_components(path)
    info = path.lstat()
    if not stat.S_ISREG(info.st_mode):
        raise DiagnosticArchiveError("Session manifest must be a regular file")
    if info.st_size > MAX_MANIFEST_BYTES:
        raise DiagnosticArchiveError("Session manifest exceeds its size limit")
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise DiagnosticArchiveError("Could not read session manifest") from exc
    payload = _parse_json(data, label="Session manifest")
    if not isinstance(payload, dict) or frozenset(payload) != MANIFEST_KEYS:
        raise DiagnosticArchiveError("Session manifest has unknown or missing fields")
    if type(payload["schema"]) is not int or payload["schema"] != 1:
        raise DiagnosticArchiveError("Unsupported session manifest schema")
    if payload["owner"] != "pyml-workbench" or payload["session_id"] != session_id:
        raise DiagnosticArchiveError("Session manifest owner or ID does not match")
    process = payload["process"]
    if (
        not isinstance(process, dict)
        or frozenset(process) != PROCESS_KEYS
        or type(process["pid"]) is not int
        or process["pid"] <= 0
        or not isinstance(process["nonce"], str)
        or not _NONCE_RE.fullmatch(process["nonce"])
    ):
        raise DiagnosticArchiveError("Session manifest process identity is invalid")
    if not isinstance(payload["role"], str) or not _ROLE_RE.fullmatch(payload["role"]):
        raise DiagnosticArchiveError("Session manifest role is invalid")
    for key in ("created_at_utc", "updated_at_utc"):
        if not _valid_utc_timestamp(payload[key]):
            raise DiagnosticArchiveError(f"Session manifest {key} is invalid")

    entries = payload["created_files"]
    if not isinstance(entries, list):
        raise DiagnosticArchiveError("Session manifest created_files must be an array")
    registered: dict[str, str] = {}
    for entry in entries:
        if not isinstance(entry, dict) or frozenset(entry) != CREATED_FILE_KEYS:
            raise DiagnosticArchiveError("Session manifest file entry is invalid")
        name = entry["path"]
        kind = entry["kind"]
        if (
            not isinstance(name, str)
            or not name
            or "/" in name
            or "\\" in name
            or ":" in name
            or name in {".", ".."}
            or PurePosixPath(name).name != name
            or not isinstance(kind, str)
            or kind not in {"jsonl", "manifest", "lock", "temp"}
            or name in registered
        ):
            raise DiagnosticArchiveError("Session manifest contains an unsafe file entry")
        if kind == "jsonl":
            _safe_member_name(name)
        registered[name] = kind
    base_name = f"process-{process['pid']}-{process['nonce']}.jsonl"
    expected = {
        "active.lock": "lock",
        "manifest.json": "manifest",
        "manifest.tmp": "temp",
        base_name: "jsonl",
        **{f"{base_name}.{index}": "jsonl" for index in range(1, MAX_SESSION_LOG_FILES)},
    }
    if registered != expected:
        raise DiagnosticArchiveError("Session manifest file registry is invalid")
    jsonl_names = {name for name, kind in registered.items() if kind == "jsonl"}
    return jsonl_names, path


def _valid_utc_timestamp(value: Any) -> bool:
    if not isinstance(value, str) or not value:
        return False
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return parsed.tzinfo is not None and parsed.utcoffset() == timezone.utc.utcoffset(parsed)


def _read_checked_file(path: Path, *, root: Path, max_bytes: int) -> bytes:
    _assert_no_link_components(path)
    try:
        resolved = path.resolve(strict=True)
        info = path.lstat()
    except OSError as exc:
        raise DiagnosticArchiveError(f"Could not resolve registered file: {path}") from exc
    if not _is_within(root, resolved):
        raise DiagnosticArchiveError("Registered file escapes the application log root")
    if not stat.S_ISREG(info.st_mode):
        raise DiagnosticArchiveError("Registered log must be a regular file")
    if info.st_size > max_bytes:
        raise DiagnosticArchiveError("Registered log exceeds the archive input limit")
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise DiagnosticArchiveError(f"Could not open registered file: {path}") from exc
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise DiagnosticArchiveError("Registered log is not a regular file")
        if opened.st_size > max_bytes:
            raise DiagnosticArchiveError("Registered log exceeds the archive input limit")
        if hasattr(info, "st_ino") and info.st_ino and opened.st_ino != info.st_ino:
            raise DiagnosticArchiveError("Registered log changed while being opened")
        chunks: list[bytes] = []
        remaining = max_bytes + 1
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        data = b"".join(chunks)
        if len(data) > max_bytes:
            raise DiagnosticArchiveError("Registered log exceeds the archive input limit")
        after = os.fstat(descriptor)
        if after.st_size != opened.st_size or after.st_mtime_ns != opened.st_mtime_ns:
            raise DiagnosticArchiveError("Registered log changed during export")
        _assert_no_link_components(path)
        return data
    finally:
        os.close(descriptor)


def _redact_text(value: str) -> str:
    value = _PRIVATE_KEY_RE.sub("[REDACTED_PRIVATE_KEY]", value)
    value = _AUTH_SCHEME_RE.sub(r"\1 [REDACTED]", value)
    value = _CREDENTIAL_ASSIGNMENT_RE.sub(r"\1[REDACTED]", value)
    value = _URL_CREDENTIALS_RE.sub(r"\1[REDACTED]@", value)
    value = _KNOWN_TOKEN_RE.sub("[REDACTED]", value)
    value = _QUOTED_LOCAL_PATH_RE.sub('"[LOCAL_PATH]"', value)
    value = _WINDOWS_PATH_RE.sub("[LOCAL_PATH]", value)
    value = _POSIX_PATH_RE.sub("[LOCAL_PATH]", value)
    value = _HOME_PATH_RE.sub("[LOCAL_PATH]", value)
    return value


def _safe_record(record: Any, *, session_id: str) -> dict[str, Any]:
    if not isinstance(record, dict) or frozenset(record) != DIAGNOSTIC_KEYS:
        raise DiagnosticArchiveError("Diagnostic record has unknown or missing fields")
    if type(record["schema"]) is not int or record["schema"] != 1:
        raise DiagnosticArchiveError("Diagnostic record schema is invalid")
    if record.get("session_id") != session_id:
        raise DiagnosticArchiveError("Diagnostic record belongs to a different session")
    if any(not isinstance(record.get(key), str) for key in ("timestamp_utc", "level", "event")):
        raise DiagnosticArchiveError("Diagnostic record has invalid required fields")
    if type(record["process_id"]) is not int or record["process_id"] <= 0:
        raise DiagnosticArchiveError("Diagnostic process_id is invalid")
    if record["error_id"] is not None and (
        not isinstance(record["error_id"], str)
        or not _ERROR_ID_RE.fullmatch(record["error_id"])
    ):
        raise DiagnosticArchiveError("Diagnostic record has an invalid error_id")
    if record["event"] == "error" and record["error_id"] is None:
        raise DiagnosticArchiveError("Error records must have an error_id")
    result: dict[str, Any] = {}
    for key, value in record.items():
        if value is None:
            result[key] = None
        elif isinstance(value, str):
            result[key] = value if key in {"error_id", "session_id"} else _redact_text(value)
        elif type(value) is int and key == "process_id":
            result[key] = value
        elif type(value) is int and key == "schema":
            result[key] = value
        else:
            raise DiagnosticArchiveError(f"Diagnostic field {key} has an unsafe value")
    if result["traceback"] is not None and result["event"] not in {"error", "stack"}:
        raise DiagnosticArchiveError("Tracebacks are allowed only on error or stack records")
    return result


def _parse_log(data: bytes, *, session_id: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(data.splitlines(), start=1):
        if not line.strip():
            continue
        record = _parse_json(line, label=f"JSONL record {line_number}")
        rows.append(_safe_record(record, session_id=session_id))
    return rows


def _string_selection(value: Iterable[str] | None, *, field: str, pattern: re.Pattern[str]) -> tuple[str, ...] | None:
    if value is None:
        return None
    if isinstance(value, (str, bytes)):
        raise DiagnosticArchiveError(f"{field} must be an iterable of IDs, not a string")
    try:
        values = tuple(value)
    except TypeError as exc:
        raise DiagnosticArchiveError(f"{field} must be an iterable of IDs") from exc
    if not values:
        raise DiagnosticArchiveError(f"{field} selection cannot be empty")
    if any(not isinstance(item, str) or not pattern.fullmatch(item) for item in values):
        raise DiagnosticArchiveError(f"{field} contains an invalid ID")
    if len(set(values)) != len(values):
        raise DiagnosticArchiveError(f"{field} selection contains duplicates")
    return values


def _session_path(root: Path, session_id: str) -> Path:
    if not _SESSION_ID_RE.fullmatch(session_id):
        raise DiagnosticArchiveError("Session ID is not a safe directory name")
    path = root / session_id
    _assert_no_link_components(path)
    if not path.is_dir():
        raise DiagnosticArchiveError("Selected session directory is missing")
    resolved = path.resolve(strict=True)
    if not _is_within(root, resolved):
        raise DiagnosticArchiveError("Selected session escapes the application log root")
    return resolved


def _validate_registered_files(
    root: Path, session_id: str, returned: Iterable[str | os.PathLike[str]]
) -> tuple[tuple[Path, str], ...]:
    session_dir = _session_path(root, session_id)
    manifest_file = session_dir / "manifest.json"
    jsonl_names, _ = _validate_manifest(manifest_file, session_id=session_id)
    returned_paths: set[Path] = set()
    for item in returned:
        try:
            path = Path(item)
        except TypeError as exc:
            raise DiagnosticArchiveError("Log registry returned an invalid path") from exc
        _assert_no_link_components(path)
        resolved = path.resolve(strict=True)
        if not _is_within(session_dir, resolved) or resolved.parent != session_dir:
            raise DiagnosticArchiveError("Log registry returned a path outside its session")
        returned_paths.add(resolved)
    expected = {session_dir / name for name in jsonl_names}
    expected.add(manifest_file)
    if returned_paths != expected:
        raise DiagnosticArchiveError("Log registry paths do not match the session manifest")
    files: list[tuple[Path, str]] = []
    for path in sorted(expected):
        if path.name == "manifest.json":
            continue
        _safe_member_name(path.name)
        _assert_no_link_components(path)
        resolved = path.resolve(strict=True)
        if not _is_within(root, resolved) or resolved.parent != session_dir:
            raise DiagnosticArchiveError("Registered log path escapes the approved root")
        info = path.lstat()
        if not stat.S_ISREG(info.st_mode):
            raise DiagnosticArchiveError("Registered log is not a regular file")
        files.append((path, path.name))
    if sum(path.stat().st_size for path, _name in files) > MAX_SESSION_LOG_BYTES:
        raise DiagnosticArchiveError("Session logs exceed the 10 MiB process limit")
    return tuple(files)


def _dump_json(value: Any) -> bytes:
    try:
        return (json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise DiagnosticArchiveError("Could not serialize sanitized archive data") from exc


def _version_manifest(app_version: str | None) -> dict[str, Any]:
    if app_version is None:
        try:
            from importlib.metadata import PackageNotFoundError, version

            try:
                app_version = version("pyml-workbench")
            except PackageNotFoundError:
                app_version = "unknown"
        except Exception:
            app_version = "unknown"
    if not isinstance(app_version, str) or not app_version or len(app_version) > 128:
        raise DiagnosticArchiveError("app_version must be a short non-empty string")
    return {
        "schema": 1,
        "app_version": _redact_text(app_version),
        "python_version": platform.python_version(),
        "platform": sys.platform,
    }


def _write_exclusive(destination: Path, data: bytes) -> Path:
    _assert_no_link_components(destination.parent)
    if not destination.parent.is_dir():
        raise DiagnosticArchiveError("Archive destination directory does not exist")
    if destination.exists() or destination.is_symlink():
        raise DiagnosticArchiveError("Archive destination already exists")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    try:
        descriptor = os.open(destination, flags, 0o600)
    except FileExistsError as exc:
        raise DiagnosticArchiveError("Archive destination already exists") from exc
    except OSError as exc:
        raise DiagnosticArchiveError("Could not create archive destination") from exc
    identity = os.fstat(descriptor)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = -1
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
    except OSError as exc:
        try:
            current = destination.lstat()
            if (
                stat.S_ISREG(current.st_mode)
                and (not identity.st_ino or current.st_ino == identity.st_ino)
            ):
                destination.unlink()
        except OSError:
            pass
        raise DiagnosticArchiveError("Could not write archive destination") from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)
    return destination


def export_diagnostics_zip(
    log_root: str | os.PathLike[str],
    destination: str | os.PathLike[str],
    *,
    session_ids: Iterable[str] | None = None,
    error_ids: Iterable[str] | None = None,
    app_version: str | None = None,
) -> Path:
    """Write a selected-session diagnostic ZIP without overwriting a file.

    At least one explicit session or error selection is required. With only
    ``session_ids``, all safe error summaries in those sessions are included;
    with ``error_ids``, sessions are resolved through the owned diagnostics API
    and only the selected error summaries/tracebacks are included.
    """
    selected_sessions = _string_selection(
        session_ids, field="session_ids", pattern=_SESSION_ID_RE
    )
    selected_errors = _string_selection(error_ids, field="error_ids", pattern=_ERROR_ID_RE)
    if selected_sessions is None and selected_errors is None:
        raise DiagnosticArchiveError("Select at least one session or error")

    root = _checked_log_root(log_root)
    try:
        destination_value = os.fspath(destination)
    except TypeError as exc:
        raise DiagnosticArchiveError("destination must be a filesystem path") from exc
    if not isinstance(destination_value, str) or "\x00" in destination_value:
        raise DiagnosticArchiveError("Archive destination is invalid")
    target = Path(os.path.abspath(destination_value))
    if target.name in {"", ".", ".."}:
        raise DiagnosticArchiveError("Archive destination is invalid")
    _assert_no_link_components(target.parent)
    if not target.parent.is_dir():
        raise DiagnosticArchiveError("Archive destination directory does not exist")
    if target.exists() or target.is_symlink():
        raise DiagnosticArchiveError("Archive destination already exists")

    # Importing the application logger has no side effects, but keep archive
    # creation independent of logger startup and optional GUI dependencies.
    try:
        from .diagnostics import owned_log_files, read_records
    except ImportError as exc:
        raise DiagnosticArchiveError("Diagnostics registry is unavailable") from exc

    if selected_sessions is None:
        try:
            candidate_records = read_records(root, error_ids=selected_errors)
        except Exception as exc:
            raise DiagnosticArchiveError("Could not resolve selected diagnostics") from exc
        found_errors: dict[str, dict[str, Any]] = {}
        for record in candidate_records:
            if (
                isinstance(record, dict)
                and record.get("event") == "error"
                and record.get("error_id") in selected_errors
            ):
                found_errors[record["error_id"]] = record
        missing = set(selected_errors or ()) - set(found_errors)
        if missing:
            raise DiagnosticArchiveError("One or more selected errors were not found")
        resolved_sessions = tuple(
            dict.fromkeys(
                record.get("session_id")
                for record in found_errors.values()
                if isinstance(record.get("session_id"), str)
            )
        )
    else:
        resolved_sessions = selected_sessions
        found_errors = {}

    if not resolved_sessions:
        raise DiagnosticArchiveError("The selection does not resolve to an owned session")
    if any(not isinstance(item, str) or not _SESSION_ID_RE.fullmatch(item) for item in resolved_sessions):
        raise DiagnosticArchiveError("Diagnostics contain an invalid session ID")

    sessions: dict[str, list[dict[str, Any]]] = {}
    total_source_bytes = 0
    for session_id in resolved_sessions:
        _session_path(root, session_id)
        try:
            records = read_records(
                root,
                session_id=session_id,
                error_ids=selected_errors,
            )
            returned_files = owned_log_files(root, session_id)
        except Exception as exc:
            raise DiagnosticArchiveError(f"Could not inspect owned session {session_id}") from exc
        registered = _validate_registered_files(root, session_id, returned_files)
        sanitized_rows: list[dict[str, Any]] = []
        for path, name in registered:
            try:
                source_size = path.stat().st_size
            except OSError as exc:
                raise DiagnosticArchiveError("Could not inspect a registered log") from exc
            total_source_bytes += source_size
            if total_source_bytes > MAX_ARCHIVE_BYTES:
                raise DiagnosticArchiveError("Selected log input exceeds the 20 MiB limit")
            data = _read_checked_file(path, root=root, max_bytes=MAX_ARCHIVE_BYTES)
            parsed_rows = _parse_log(data, session_id=session_id)
            sanitized_rows.extend(parsed_rows)
            archived_rows: list[dict[str, Any]] = []
            for row in parsed_rows:
                archive_row = dict(row)
                if (
                    selected_errors is not None
                    and archive_row.get("error_id") not in selected_errors
                ):
                    archive_row["traceback"] = None
                archived_rows.append(archive_row)
            sessions.setdefault(session_id, []).append(
                {
                    "member": f"logs/{session_id}/{name}",
                    "data": _dump_json_lines(archived_rows),
                }
            )

        # Cross-check the logger's selection API against registered log rows;
        # a stale or foreign record must never be summarized as this session.
        api_errors: dict[str, dict[str, Any]] = {}
        for record in records:
            if not isinstance(record, dict) or record.get("event") != "error":
                continue
            error_id = record.get("error_id")
            if error_id is None or (
                selected_errors is not None and error_id not in selected_errors
            ):
                continue
            safe = _safe_record(record, session_id=session_id)
            api_errors[error_id] = safe
        row_errors = {
            row.get("error_id"): row
            for row in sanitized_rows
            if row.get("event") == "error" and row.get("error_id") is not None
        }
        for error_id, safe in api_errors.items():
            if error_id not in row_errors:
                raise DiagnosticArchiveError("Selected error is absent from registered logs")
            if safe != row_errors[error_id]:
                raise DiagnosticArchiveError("Logger records do not match owned log contents")
        if selected_errors is not None:
            expected_for_session = {
                error_id
                for error_id, record in found_errors.items()
                if record.get("session_id") == session_id
            }
            expected_for_session.update(
                row.get("error_id")
                for row in sanitized_rows
                if row.get("event") == "error"
                and row.get("error_id") in selected_errors
            )
            if not expected_for_session.issubset(row_errors):
                raise DiagnosticArchiveError("One or more selected errors are missing from logs")

    error_summaries: list[dict[str, Any]] = []
    tracebacks: list[dict[str, str]] = []
    for session_id in resolved_sessions:
        for item in sessions[session_id]:
            rows = _parse_jsonl_bytes(item["data"], session_id=session_id)
            for row in rows:
                if row.get("event") != "error":
                    continue
                error_id = row.get("error_id")
                if selected_errors is not None and error_id not in selected_errors:
                    continue
                summary = {key: row[key] for key in SUMMARY_KEYS if key in row}
                error_summaries.append(summary)
                traceback_text = row.get("traceback")
                if isinstance(traceback_text, str) and traceback_text:
                    tracebacks.append(
                        {"error_id": error_id or "", "traceback": traceback_text}
                    )

    if selected_errors is not None:
        exported_ids = {item.get("error_id") for item in error_summaries}
        if not set(selected_errors).issubset(exported_ids):
            raise DiagnosticArchiveError("One or more selected errors are missing from the archive")

    archive_files: list[tuple[str, bytes]] = [
        ("manifest.json", _dump_json(_version_manifest(app_version))),
        (
            "summary.json",
            _dump_json({"schema": 1, "errors": error_summaries}),
        ),
        ("tracebacks.json", _dump_json({"schema": 1, "errors": tracebacks})),
    ]
    for session_id in resolved_sessions:
        archive_files.extend(
            (item["member"], item["data"]) for item in sessions[session_id]
        )

    raw_total = sum(len(data) for _, data in archive_files)
    if raw_total > MAX_ARCHIVE_BYTES:
        raise DiagnosticArchiveError("Sanitized archive contents exceed 20 MiB")
    try:
        with tempfile.SpooledTemporaryFile(max_size=1024 * 1024, mode="w+b") as memory_file:
            with zipfile.ZipFile(
                memory_file,
                mode="w",
                compression=zipfile.ZIP_DEFLATED,
                compresslevel=6,
                allowZip64=False,
            ) as archive:
                for name, data in archive_files:
                    archive.writestr(name, data)
            memory_file.seek(0, os.SEEK_END)
            archive_size = memory_file.tell()
            if archive_size > MAX_ARCHIVE_BYTES:
                raise DiagnosticArchiveError("ZIP archive exceeds the 20 MiB limit")
            memory_file.seek(0)
            zip_bytes = memory_file.read(MAX_ARCHIVE_BYTES + 1)
        if len(zip_bytes) > MAX_ARCHIVE_BYTES:
            raise DiagnosticArchiveError("ZIP archive exceeds the 20 MiB limit")
    except DiagnosticArchiveError:
        raise
    except (OSError, RuntimeError, zipfile.BadZipFile) as exc:
        raise DiagnosticArchiveError("Could not build diagnostic ZIP") from exc
    return _write_exclusive(target, zip_bytes)


def _dump_json_lines(rows: list[dict[str, Any]]) -> bytes:
    try:
        return b"".join(
            (json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n").encode("utf-8")
            for row in rows
        )
    except (TypeError, ValueError, UnicodeEncodeError) as exc:
        raise DiagnosticArchiveError("Could not serialize sanitized diagnostics") from exc


def _parse_jsonl_bytes(data: bytes, *, session_id: str) -> list[dict[str, Any]]:
    # Reparse the sanitized member so summary and traceback output is derived
    # from the exact bytes that will be written to the archive.
    return _parse_log(data, session_id=session_id)


__all__ = [
    "MAX_ARCHIVE_BYTES",
    "MAX_SESSION_LOG_BYTES",
    "DiagnosticArchiveError",
    "export_diagnostics_zip",
]
